"""RAD1 isolated GT-free physical association decoder. No model imports."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

K = 3
MAX_MATCHES = 512
MAX_SEEDS = 32
MAX_CLIQUE = 64
COMPATIBILITY_M = 1.0
INLIER_M = 1.0
REFITS = 3


def apply(T, x):
    return np.asarray(x, float) @ T[:3, :3].T + T[:3, 3]


def fit(source, target):
    s, t = np.asarray(source, float), np.asarray(target, float)
    if len(s) < 3 or s.shape != t.shape or not np.isfinite([s, t]).all():
        return None
    a, b = s-s.mean(0), t-t.mean(0)
    for p in (a, b):
        v = np.linalg.svd(p, compute_uv=False)
        if v[0] <= 1e-12 or v[1] <= 1e-6*v[0]:
            return None
    u, spectrum, vt = np.linalg.svd(a.T @ b)
    sign = 1 if np.linalg.det(vt.T @ u.T) >= 0 else -1
    # Twice this quantity is the Horn objective's top eigenvalue gap.
    gap = 2*(spectrum[1]+sign*spectrum[2])
    if spectrum[0] <= 1e-12 or gap <= 1e-6*spectrum[0]:
        return None
    d = np.eye(3)
    d[2, 2] = 1 if np.linalg.det(vt.T @ u.T) >= 0 else -1
    R = vt.T @ d @ u.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t.mean(0)-R@s.mean(0)
    return T


def _top(d, ids):
    k = min(K, d.shape[1])
    indices = np.argpartition(d, k-1, axis=1)[:, :k]
    out = np.empty_like(indices)
    for row, a in enumerate(indices):
        cutoff = d[row, a].max()
        tied = np.flatnonzero(d[row] == cutoff)
        if len(tied) > np.count_nonzero(d[row, a] == cutoff):
            lower = np.flatnonzero(d[row] < cutoff)
            a = np.r_[lower, tied[np.argsort(ids[tied], kind='stable')][:k-len(lower)]]
        out[row] = a[np.lexsort((ids[a], d[row, a]))]
    return out


def associate(a, b, ids_a=None, ids_b=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1]
    assert np.isfinite(a).all() and np.isfinite(b).all()
    ids_a = np.arange(len(a)) if ids_a is None else np.asarray(ids_a)
    ids_b = np.arange(len(b)) if ids_b is None else np.asarray(ids_b)
    assert len(np.unique(ids_a)) == len(a) and len(np.unique(ids_b)) == len(b)
    d = cdist(a, b)
    forward, reverse = _top(d, ids_b), _top(d.T, ids_a)
    i = np.repeat(np.arange(len(a)), forward.shape[1])
    j = forward.ravel()
    good = np.any(reverse[j] == i[:, None], axis=1)
    i, j = i[good], j[good]
    order = np.lexsort((ids_b[j], ids_a[i], d[i, j]))[:MAX_MATCHES]
    pairs = np.c_[i[order], j[order]]
    return pairs, d[i[order], j[order]]


def solve(x, y, pairs, distances, fallback, ids_x=None, ids_y=None):
    n = len(pairs)
    info = dict(fallback=True, proposals=0, seed_rank=-1,
                clique=np.empty(0, dtype=np.int64), refits=np.zeros((REFITS, n), bool),
                final_inliers=np.zeros(n, bool))
    if n < 3:
        return np.asarray(fallback).copy(), info
    a, b = np.asarray(x, float)[pairs[:, 0]], np.asarray(y, float)[pairs[:, 1]]
    graph = np.abs(cdist(a, a)-cdist(b, b)) <= COMPATIBILITY_M
    graph &= (pairs[:, 0, None] != pairs[None, :, 0]) & (pairs[:, 1, None] != pairs[None, :, 1])
    ids_x = np.arange(len(x)) if ids_x is None else np.asarray(ids_x)
    ids_y = np.arange(len(y)) if ids_y is None else np.asarray(ids_y)
    rank = np.lexsort((ids_y[pairs[:, 1]], ids_x[pairs[:, 0]], distances, -graph.sum(1)))
    best = None
    for seed_rank, seed in enumerate(rank[:MAX_SEEDS]):
        clique = [int(seed)]
        remaining = graph[seed].copy()
        while remaining.any() and len(clique) < MAX_CLIQUE:
            q = int(rank[remaining[rank]][0])
            clique.append(q)
            remaining &= graph[q]
        T = fit(b[clique], a[clique])
        if T is None:
            continue
        masks = []
        for _ in range(REFITS):
            mask = np.linalg.norm(apply(T, b)-a, axis=1) <= INLIER_M
            T = fit(b[mask], a[mask])
            if T is None:
                break
            masks.append(mask)
        if T is None:
            continue
        assert len(masks) == REFITS
        residual = np.linalg.norm(apply(T, b)-a, axis=1)
        inliers = residual <= INLIER_M
        key = (-int(inliers.sum()), round(float(np.mean(np.minimum(residual, INLIER_M)**2)), 12), seed_rank)
        info['proposals'] += 1
        if best is None or key < best[0]:
            best = (key, T.copy(), seed_rank, np.asarray(clique), np.asarray(masks), inliers)
    if best is None:
        return np.asarray(fallback).copy(), info
    _, T, seed_rank, clique, masks, inliers = best
    info.update(fallback=False, seed_rank=seed_rank, clique=clique,
                refits=masks, final_inliers=inliers)
    return T, info


def geometry(x, y, T):
    a, b = np.asarray(x, float), apply(T, y)
    db = cKDTree(a).query(b, workers=1)[0]
    da = cKDTree(b).query(a, workers=1)[0]
    return np.array([.5*((da < 1).mean()+(db < 1).mean()), .5*(da.mean()+db.mean())])


def select(scores):
    return min(range(len(scores)), key=lambda k: (-float(scores[k, 0]), float(scores[k, 1]), k))
