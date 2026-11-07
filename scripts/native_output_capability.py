"""Isolated native-output adapter and direct-point capability control."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TAUS = (1., .8, .6, .4, .2)


class LowRank(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(8, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, 8, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


def adapt(flow):
    flow.requires_grad_(False)
    for layer in flow.transformer_layers[10:]:
        for name in ('global_qkv_proj', 'global_out_proj'):
            setattr(layer, name, LowRank(getattr(layer, name)))
    flow.final_mlp.requires_grad_(True)
    assert sum(p.numel() for p in flow.parameters() if p.requires_grad) == 443904


def args(batch, x, tau):
    counts = batch['points_per_part'].reshape(-1)
    part = F.pad(torch.cumsum(counts, 0), (1, 0)).int()
    return dict(x=x, timesteps=torch.tensor([tau], device=x.device),
        cond_coord=batch['pointclouds'], local_features=batch['features'], latent_features=None,
        scales=batch['scales'], anchor_indices=batch['anchor_indices'],
        cu_seqlens_batch=batch['cu_seqlens'].int(), cu_seqlens_part=part)


def prefix(flow, kw):
    from rectified_point_flow.utils.point_clouds import repeat_by_cu_seqlens
    scale = repeat_by_cu_seqlens(kw['scales'], kw['cu_seqlens_batch'])
    e = flow.encoding_manager(kw['x'], kw['cond_coord'], kw['local_features'], None, scale)
    e = flow._add_anchor_embedding(e, kw['anchor_indices'])
    for layer in flow.transformer_layers[:10]:
        e = layer(e, kw['timesteps'], kw['cu_seqlens_batch'], kw['cu_seqlens_part'])
    return e.detach()


def tail(flow, e, kw):
    for layer in flow.transformer_layers[10:]:
        e = layer(e, kw['timesteps'], kw['cu_seqlens_batch'], kw['cu_seqlens_part'])
    with torch.autocast('cuda', enabled=False):
        return flow.final_mlp(e.float())


def trainables(flow):
    return {n:p for n,p in flow.named_parameters() if p.requires_grad}


def frozen_digest(flow):
    h = hashlib.sha256()
    for n,p in flow.named_parameters():
        if not p.requires_grad:
            h.update(n.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Direct(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(39,256), nn.SiLU(), nn.Linear(256,512), nn.SiLU(),
            nn.Linear(512,512), nn.SiLU(), nn.Linear(512,3))

    def forward(self, batch):
        part = torch.arange(3, device=batch['pointclouds'].device).repeat_interleave(1000)
        ids = F.one_hot(part,3).float()
        scale = batch['scales'].log().expand(3000,1)
        return self.net(torch.cat([batch['pointclouds'], batch['features'], ids, scale], 1))


def poses(cond, output, centers, scale):
    """Native unweighted per-part SVD fit, float64 metric restoration."""
    out=[]
    for k in range(3):
        x = np.asarray(cond[k*1000:(k+1)*1000],np.float64)
        y = np.asarray(output[k*1000:(k+1)*1000],np.float64)
        u,_,v = np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)))
        R = v.T@np.diag([1.,1.,np.linalg.det(v.T@u.T)])@u.T
        T=np.eye(4); T[:3,:3]=R; T[:3,3]=scale*(y.mean(0)-R@x.mean(0))-R@centers[k]
        out.append(T)
    return np.array(out)


def errors(part, truth):
    out=[]
    for k in (1,2):
        p = np.linalg.inv(part[0])@part[k]
        delta = p[:3,:3].T@truth[k,:3,:3]
        sine = np.linalg.norm([delta[2,1]-delta[1,2],delta[0,2]-delta[2,0],delta[1,0]-delta[0,1]])/2
        cosine=(np.trace(delta)-1)/2
        out.append(dict(part=k,t_m=float(np.linalg.norm(p[:3,3]-truth[k,:3,3])),
            R_deg=float(np.degrees(np.arctan2(sine,cosine)))))
    return out


def gate(rows):
    return all(x['t_m'] <= .05 and x['R_deg'] <= .2 for x in rows)
