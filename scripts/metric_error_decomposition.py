"""Isolated algebraic accounting of camera segment translation errors."""
import numpy as np


def anchor(t):
    return np.linalg.inv(t[0]) @ t


def align(g, e):
    x=e[:, :3, 3];y=g[:, :3, 3]
    xc=x-x.mean(0);yc=y-y.mean(0)
    u, _, vt=np.linalg.svd(xc.T @ yc)
    d=np.eye(3);d[-1,-1]=np.linalg.det(vt.T @ u.T)
    q=np.eye(4);q[:3,:3]=vt.T @ d @ u.T
    q[:3,3]=y.mean(0)-q[:3,:3] @ x.mean(0)
    return q


def decompose(g, e, first, last, length):
    u=[];v=[]
    for a,b,l in zip(first,last,length):
        chord_g=g[b,:3,3]-g[a,:3,3]
        chord_e=e[b,:3,3]-e[a,:3,3]
        u.append(np.linalg.solve(e[b,:3,:3], e[a,:3,:3] @ np.linalg.solve(g[a,:3,:3],chord_g)-chord_g)/l)
        v.append(np.linalg.solve(e[b,:3,:3],chord_g-chord_e)/l)
    u=np.array(u);v=np.array(v)
    return dict(u=u,v=v,z=u+v)
