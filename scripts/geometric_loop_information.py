"""Isolated PFG geometry information; conditional curvature, not covariance."""
import numpy as np
from scipy.spatial import cKDTree


def skew(v):
    x,y,z=np.asarray(v,dtype=float)
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def adjoint(T):
    R=T[:3,:3];t=T[:3,3]
    A=np.zeros((6,6));A[:3,:3]=R;A[3:,3:]=R;A[3:,:3]=skew(t)@R
    return A


def surface_jacobians(T,p,q,ni,nj):
    """T maps q in j to p in i; right tangent [rotation, translation]."""
    R=T[:3,:3];t=T[:3,3]
    Jf=np.array([n@R@np.column_stack((-skew(x),np.eye(3))) for x,n in zip(q,ni)])
    reverse=(p-t)@R
    Jr=np.array([n@np.column_stack((skew(x),-np.eye(3))) for x,n in zip(reverse,nj)])
    return np.vstack((Jf,Jr)).reshape(-1,6)


def normalized_information(J,Sigma):
    H=J.T@J;H=(H+H.T)/2
    denom=float(np.trace(Sigma@H))
    if not np.isfinite(denom) or denom<=1e-20:
        raise ValueError('No measurable geometric support')
    Omega=6*H/denom
    lam,V=np.linalg.eigh(Omega)
    if lam[0]<-1e-10*max(1.,lam[-1]):raise ValueError('Non-PSD information')
    # Only roundoff-negative eigenvalues become zero; no positive eigenvalue floor.
    A=np.sqrt(np.maximum(lam,0))[:,None]*V.T
    return dict(H=H,Omega=Omega,A=A,rank=int(np.count_nonzero(lam>1e-10*lam[-1])))


def measured_normals(points,k=20):
    points=np.asarray(points,dtype=float);assert len(points)>=k
    tree=cKDTree(points);dist,idx=tree.query(points,k=k+1)
    normals=[];valid=[];neighbors=[]
    for row,x in enumerate(points):
        ids=idx[row,:k]
        if dist[row,k-1]==dist[row,k]:
            ids=np.asarray(tree.query_ball_point(x,np.nextafter(dist[row,k],np.inf)))
        ds=np.sum((points[ids]-x)**2,axis=1)
        ids=ids[np.lexsort((ids,ds))][:k]
        local=points[ids];centered=local-local.mean(axis=0)
        lam,V=np.linalg.eigh(centered.T@centered/k)
        # A line/duplicate cluster does not specify a unique plane normal.
        okay=bool(lam[1]>1e-12 and lam[1]-lam[0]>1e-10*max(1.,lam[-1]))
        normals.append(V[:,0]);valid.append(okay);neighbors.append(ids)
    return np.asarray(normals),np.asarray(valid),np.asarray(neighbors)


def nearest_indices(src,dst):
    tree=cKDTree(dst);dist,idx=tree.query(src,k=2)
    out=idx[:,0].copy()
    for a in np.flatnonzero(dist[:,0]==dist[:,1]):
        ids=np.asarray(tree.query_ball_point(src[a],np.nextafter(dist[a,1],np.inf)))
        ds=np.sum((dst[ids]-src[a])**2,axis=1)
        out[a]=ids[np.lexsort((ids,ds))[0]]
    return out


def geometry_information(T,p,q,normal_p,valid_p,normal_q,valid_q,Sigma):
    qp=q@T[:3,:3].T+T[:3,3]
    q_to_p=nearest_indices(qp,p);p_to_q=nearest_indices(p,qp)
    idsq=np.arange(len(q));idsp=q_to_p
    keep=(p_to_q[idsp]==idsq)&valid_p[idsp]&valid_q
    keep &= np.sum((p[idsp]-qp)**2,axis=1)<=1.0
    iq=idsq[keep];ip=idsp[keep]
    if len(iq)==0:return dict(supported=False,ip=ip,iq=iq,J=np.empty((0,6)))
    J=surface_jacobians(T,p[ip],q[iq],normal_p[ip],normal_q[iq])
    try:r=normalized_information(J,Sigma)
    except ValueError:return dict(supported=False,ip=ip,iq=iq,J=J)
    return dict(supported=True,ip=ip,iq=iq,J=J,**r)
