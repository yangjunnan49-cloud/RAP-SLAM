"""Isolated RLP2 motion-supervised original-point soft readout."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PAIRS=((0,1),(0,2),(1,2))

class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(521,128)
        self.project=nn.Linear(128,64)
        assert sum(p.numel() for p in self.parameters())==75072
    def forward(self,feature,common):
        feature=F.normalize(feature,dim=-1,eps=1e-8)
        if feature.shape[-1]==32:feature=F.pad(feature,(0,480))
        return F.normalize(self.project(F.silu(self.hidden(torch.cat((feature,common),-1)))),dim=-1,eps=1e-8)

def weights(a,b):
    s=a@b.T/.10
    return F.softmax(s,dim=1)*F.softmax(s.T,dim=1).T

def solve(source,target,raw):
    x,y,w=source.double(),target.double(),raw.double()
    mass=w.sum()
    if not torch.isfinite(mass) or float(mass.detach())<=0:raise ValueError('Invalid soft mass')
    w=w/mass
    cx=w.sum(0)@x;cy=w.sum(1)@y
    H=(x-cx).T@w.T@(y-cy)
    xx,xy,xz,yx,yy,yz,zx,zy,zz=H.reshape(-1).unbind()
    K=torch.stack([torch.stack(row) for row in (
        (xx+yy+zz,yz-zy,zx-xz,xy-yx),(yz-zy,xx-yy-zz,xy+yx,zx+xz),
        (zx-xz,xy+yx,-xx+yy-zz,yz+zy),(xy-yx,zx+xz,yz+zy,-xx-yy+zz))])
    ev=torch.linalg.eigvalsh(K.detach())
    gap=(ev[-1]-ev[-2])/ev.abs().max().clamp_min(1e-12)
    if not torch.isfinite(gap) or float(gap)<=1e-6:raise ValueError('Nonunique soft proper objective')
    u,s,vh=torch.linalg.svd(H)
    sign=torch.linalg.det(vh.T@u.T)
    R=vh.T@torch.diag(torch.stack((sign*0+1,sign*0+1,sign)))@u.T
    t=cy-R@cx
    if not torch.isfinite(R).all() or not torch.isfinite(t).all():raise FloatingPointError('Nonfinite pose')
    meta=dict(mass=mass.detach(),gap=gap,values=ev,marginal_target=w.sum(1).detach(),marginal_source=w.sum(0).detach())
    return R,t,meta

def loss(head,feature,common,points,truth):
    E=head(feature,common);terms=[];gaps=[]
    for k,(a,b) in enumerate(PAIRS):
        R,t,m=solve(points[b],points[a],weights(E[a],E[b]))
        td=torch.linalg.vector_norm(t-truth[k,:3,3])/.20
        rd=torch.linalg.matrix_norm(R-truth[k,:3,:3])/(np.sqrt(2.)*np.deg2rad(.2))
        terms.append(F.huber_loss(td,torch.zeros_like(td),delta=1.)+F.huber_loss(rd,torch.zeros_like(rd),delta=1.))
        gaps.append(float(m['gap']))
    result=torch.stack(terms).mean()
    if not torch.isfinite(result):raise FloatingPointError('Nonfinite loss')
    return result,dict(pair_losses=[float(v.detach()) for v in terms],gaps=gaps)

def common(z):
    roles=np.zeros((3000,1),np.float32);roles[:1000]=1
    constants=np.tile(np.array([np.log(float(z['scale'])),.2],np.float32),(3000,1))
    return np.concatenate((z['cond'],z['noise'],roles,constants),1).reshape(3,1000,9)

def encode_prediction(head,feature,channels,points):
    with torch.no_grad():
        e=head(feature,channels);raw=weights(e[0],e[1])
        R,t,m=solve(points[1],points[0],raw)
    raw_np=raw.cpu().numpy()
    pose=np.eye(4);pose[:3,:3]=R.cpu().numpy();pose[:3,3]=t.cpu().numpy()
    return dict(embedding=e[:2].cpu().numpy(),pose=pose,
        weight_sha256=np.array(hashlib.sha256(raw_np.tobytes()).hexdigest()),
        **{k:v.cpu().numpy() for k,v in m.items()})
