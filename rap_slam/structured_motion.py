"""Isolated F1 structured-motion sampler. No upstream or model monkeypatches."""
import numpy as np
import torch
from .rap_precision import ieee_matmul


def build_batch(backend, points, features):
    cond, scale, centers, counts, anchor = backend._normalize_parts(points)
    offsets = np.r_[0, np.cumsum(counts)]
    mask = np.zeros(len(cond), dtype=bool)
    mask[offsets[anchor]:offsets[anchor+1]] = True
    device = backend.device
    batch = dict(pointclouds=torch.as_tensor(cond, device=device),
                 pointclouds_gt=torch.as_tensor(cond, device=device),
                 features=torch.as_tensor(np.concatenate(features), device=device),
                 scales=torch.tensor([scale], dtype=torch.float32, device=device),
                 anchor_indices=torch.as_tensor(mask, device=device),
                 points_per_part=torch.as_tensor(counts[None], device=device),
                 cu_seqlens=torch.tensor([0,len(cond)], device=device))
    return batch, scale, np.asarray(centers), counts


def group_coordinates(points, motions, centers, scale, device):
    grouped=[]
    for k,(p,L) in enumerate(zip(points,motions)):
        c=centers[0 if k<3 else 3]
        grouped.append((p.astype(np.float64)@L[:3,:3].T+L[:3,3]-c)/scale)
    return torch.as_tensor(np.concatenate(grouped), dtype=torch.float32, device=device)


def project_groups(raw, source, split):
    from rectified_point_flow.procrustes import solve_procrustes
    projected=[]; poses=[]
    with torch.autocast(device_type=raw.device.type, enabled=False):
        for sl in [slice(0,split),slice(split,None)]:
            p,y=source[sl].float(),raw[sl].float()
            R,t=solve_procrustes(p,y)
            projected.append(p@R.T+t)
            poses.append((R,t))
    return torch.cat(projected),poses


def decode_groups(poses, centers, scale):
    transforms=[]
    for g,(R,t) in enumerate(poses):
        r=R.detach().double().cpu().numpy();v=t.detach().double().cpu().numpy()
        T=np.eye(4);T[:3,:3]=r;T[:3,3]=scale*v-r@centers[0 if g==0 else 3]
        transforms.append(T)
    return np.asarray(transforms)


def decode_parts(batch, raw, centers, scale):
    from rectified_point_flow.procrustes import fit_transformations
    with torch.autocast(device_type=raw.device.type, enabled=False):
        r,t=fit_transformations(batch['pointclouds'].float(),raw.float(),batch['points_per_part'],batch['cu_seqlens'].int())
    r=r[0].detach().double().cpu().numpy();t=t[0].detach().double().cpu().numpy()
    Ts=np.tile(np.eye(4),(len(r),1,1));Ts[:,:3,:3]=r
    Ts[:,:3,3]=t*scale-np.einsum('nij,nj->ni',r,centers)
    return Ts


def sample(model, batch, noise, group_source=None, split=None):
    """Native-equivalent Euler; optional group projection on raw clean endpoint."""
    from rectified_point_flow.procrustes import rigidify_prediction_with_procrustes
    _,cond,features,scales,_,anchor,counts,cu,parts=model._prepare_data(batch)
    assert model.inference_sampler=='euler' and model.inference_sampling_steps==5
    x=noise.clone();dt=1.0/model.inference_sampling_steps
    raw_steps=[];clean_steps=[];states=[]
    for step in range(model.inference_sampling_steps):
        t=1-step*dt
        v=model.flow_model(x=x,timesteps=torch.full((len(cu)-1,),t,device=x.device),
            cond_coord=cond,local_features=features,latent_features=None,scales=scales,
            anchor_indices=anchor,cu_seqlens_batch=cu,cu_seqlens_part=parts)
        raw=x-v*t
        if group_source is None:
            clean=rigidify_prediction_with_procrustes(raw,cond,counts,cu)
        else:
            clean,_=project_groups(raw,group_source,split)
        x=clean*(1-t+dt)+noise*(t-dt)
        raw_steps.append(raw.detach().clone());clean_steps.append(clean.detach().clone());states.append(x.detach().clone())
    return dict(end_point_trajectory=torch.stack(raw_steps),
                projected_clean_trajectory=torch.stack(clean_steps),trajectory=torch.stack(states))


def infer(backend,batch,noise,group_source=None,split=None):
    with ieee_matmul(),torch.inference_mode(),torch.autocast(device_type=backend.device.type,dtype=torch.float16):
        return sample(backend.model,batch,noise,group_source,split)