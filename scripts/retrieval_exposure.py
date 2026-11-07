"""Isolated LC0 exposure reconstruction; no production retrieval change."""
import numpy as np
from rap_slam.scan_context import make_scan_context, make_ring_key, sc_distance
from rap_slam.real_dataset import kitti_vertical_correction


def descriptor(path):
    xyz = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    xyz = xyz[np.linalg.norm(xyz, axis=1) >= 1.5]
    xyz = kitti_vertical_correction(xyz, 0.205)
    sc = make_scan_context(xyz)
    return sc, make_ring_key(sc), len(xyz)


def exposure(kf, odom, sc, rk):
    kf = np.asarray(kf, dtype=np.int64)
    ai, bj = np.triu_indices(len(kf), 1)
    keep = kf[bj] - kf[ai] >= 200
    ai, bj = ai[keep], bj[keep]
    i, j = kf[ai], kf[bj]
    od = np.asarray([float(np.linalg.norm(odom[a, :3, 3] - odom[b, :3, 3])) for a, b in zip(i, j)])
    # Preserve the native one-dimensional float32 norm reduction.
    rd = np.asarray([float(np.linalg.norm(rk[b] - rk[a])) for a, b in zip(ai, bj)])
    rank = np.zeros(len(i), dtype=np.int32)
    sd = np.full(len(i), -1., dtype=np.float64)
    shift = np.full(len(i), -1, dtype=np.int16)
    for b in range(1, len(kf)):
        ids = np.flatnonzero(bj == b)
        ids = ids[np.argsort(rd[ids], kind='stable')]
        rank[ids] = np.arange(1, len(ids) + 1)
        for t in ids[:10]:
            sd[t], shift[t] = sc_distance(sc[ai[t]], sc[bj[t]])
    prox = np.flatnonzero(od < 15.)
    prox = prox[np.argsort(od[prox], kind='stable')]
    sp = np.flatnonzero((sd >= 0) & (sd < .25))
    sp = sp[np.lexsort((rank[sp], bj[sp], sd[sp]))]
    pc, cc = prox[:12], sp[:12]
    merged = list(dict.fromkeys([int(t) for t in np.r_[pc, cc]]))[:12]
    return dict(i=i, j=j, ai=ai, bj=bj, odom_distance=od, ring_distance=rd,
                ring_rank=rank, sc_distance=sd, sc_shift=shift,
                proximity_precap=prox, sc_precap=sp, proximity_cap=pc,
                sc_cap=cc, final=np.asarray(merged, dtype=np.int64),
                union_precap=np.union1d(prox, sp))


def distributed(pair_ids, i, j, gt_distance, budget=12):
    near = [int(t) for t in pair_ids if gt_distance[t] < 15.]
    chosen = {}
    for t in near:
        q = int(j[t])
        if q not in chosen or (gt_distance[t], i[t]) < (gt_distance[chosen[q]], i[chosen[q]]):
            chosen[q] = t
    ordered = [chosen[q] for q in sorted(chosen)]
    k = min(budget, len(ordered))
    return np.asarray([ordered[int(np.floor((a + .5) * len(ordered) / k))]
                       for a in range(k)], dtype=np.int64)

