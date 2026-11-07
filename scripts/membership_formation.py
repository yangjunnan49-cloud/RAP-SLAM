"""Fixed-criterion membership census and costs; contains no pose optimizer."""
import hashlib
import numpy as np

SIGMA = .10
SCALE = 2.
FIELDS = ('counts', 'means', 'seconds', 'keys')


def planes(stats, poses):
    active = stats['counts'] > 0
    mass = active/active.sum(1)[:, None]
    mu = stats['means']
    local = stats['seconds']-mu[..., :, None]*mu[..., None, :]
    r = poses[None, :, :3, :3]
    wm = np.einsum('nij,knj->kni', poses[:, :3, :3], mu)+poses[None, :, :3, 3]
    center = np.sum(mass[..., None]*wm, axis=1)
    d = wm-center[:, None]
    covariance = np.sum(mass[..., None, None]*(r@local@r.transpose(0, 1, 3, 2)+d[..., :, None]*d[..., None, :]), axis=1)
    values, vectors = np.linalg.eigh(covariance)
    return values, vectors[:, :, 0], center, wm, mass


def form(points, poses, seq, endpoints):
    n = len(points)
    world = [p@t[:3, :3].T+t[:3, 3] for p, t in zip(points, poses)]
    all_keys = np.concatenate([np.floor(p/SCALE).astype(np.int64) for p in world])
    node = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(points)])
    row = np.concatenate([np.arange(len(p)) for p in points])
    unique, inverse = np.unique(all_keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    bins = np.split(order, np.flatnonzero(np.diff(inverse[order]))+1)
    raw_counts = np.zeros((len(unique), n), np.int64)
    effective_counts = raw_counts.copy()
    reason = np.ones(len(unique), np.int8)  # 0 cross accepted, 1 support, 2 plane, 3 single visit.
    plane_fail_bits = np.zeros(len(unique), np.uint8)
    eigenvalues = np.full((len(unique), 3), np.nan)
    offset_max = np.full(len(unique), np.nan)
    accepted, records, members = [], [], []
    full_eligible = 0
    for c, ids in enumerate(bins):
        mu, second = np.zeros((n, 3)), np.zeros((n, 3, 3))
        per_scan = []
        for i in range(n):
            selected = row[ids[node[ids] == i]]
            raw_counts[c, i] = len(selected)
            if len(selected) < 6:
                continue
            p = points[i][selected]
            effective_counts[c, i], mu[i], second[i] = len(p), p.mean(0), p.T@p/len(p)
            per_scan.append((i, selected))
        count = effective_counts[c]
        if np.sum(count > 0) < 3 or count.sum() < 40:
            continue
        temporary = dict(counts=count[None], means=mu[None], seconds=second[None])
        value, normal, center, wm, _ = planes(temporary, poses)
        eigenvalues[c] = value[0]
        offset_max[c] = np.max(np.abs((wm[0]-center[0])@normal[0])[count > 0])
        flags = [not (value[0, 0]/max(value[0].sum(), 1e-15) < .025),
                 not (value[0, 1] > .005), not (value[0, 0] <= .015), not (offset_max[c] <= .20)]
        plane_fail_bits[c] = sum((1 << i) for i, failed in enumerate(flags) if failed)
        if any(flags):
            reason[c] = 2
            continue
        full_eligible += 1
        if not (count[:8].sum() > 0 and count[8:].sum() > 0):
            reason[c] = 3
            continue
        reason[c] = 0
        accepted.append(c)
        records.append((count.copy(), mu, second))
        members.append(per_scan)
    stats = dict(counts=np.zeros((0, n), np.int64), means=np.zeros((0, n, 3)), seconds=np.zeros((0, n, 3, 3)))
    if records:
        stats = {name: np.stack([p[i] for p in records]) for i, name in enumerate(FIELDS[:3])}
    stats['keys'] = unique[np.array(accepted, np.int64)]
    labels = [np.full(len(p), -1, np.int64) for p in points]
    for k, per_scan in enumerate(members):
        for i, ids in per_scan:
            labels[i][ids] = k
    payloads = [f'CV1|{seq}|{endpoints[0]}|{endpoints[1]}|2|'+','.join(str(int(x)) for x in key) for key in stats['keys']]
    numbers = np.array([int.from_bytes(hashlib.sha256(x.encode()).digest()[:4], 'big') for x in payloads], np.uint32)
    output = {**stats, 'holdout': numbers%4 == 0, 'hash_integers': numbers, 'candidate_keys': unique,
        'candidate_raw_counts': raw_counts, 'candidate_effective_counts': effective_counts,
        'candidate_reason': reason, 'candidate_plane_fail_bits': plane_fail_bits,
        'candidate_eigenvalues': eigenvalues, 'candidate_offset_max_m': offset_max,
        'accepted_candidate_ids': np.array(accepted, np.int64)}
    cursor = 0
    for i, p in enumerate(points):
        candidate = inverse[cursor:cursor+len(p)]
        cursor += len(p)
        output[f'candidate_id_{i}'] = candidate
        output[f'label_{i}'] = labels[i]
        point_reason = reason[candidate].copy()
        point_reason[(point_reason == 0) & (labels[i] < 0)] = 4  # Inactive <6-point scan in accepted voxel.
        output[f'point_reason_{i}'] = point_reason
    info = dict(candidate_voxels=len(unique), full_eligible_planes=full_eligible,
        accepted_cross_patches=len(accepted), rejected_support=int(np.sum(reason == 1)),
        rejected_plane=int(np.sum(reason == 2)), rejected_single_visit=int(np.sum(reason == 3)))
    return output, info


def subset(stats, mask):
    return {k: stats[k][mask] for k in FIELDS}


def describe(stats, mask):
    count = stats['counts'][mask].sum(0)
    return dict(patches=int(mask.sum()), per_scan_points=count.tolist(),
        per_visit_points=[int(count[:8].sum()), int(count[8:].sum())],
        per_visit_scans=[int(np.sum(count[:8] > 0)), int(np.sum(count[8:] > 0))])


def costs(stats, poses, mask):
    if not np.any(mask):
        return None
    values = np.maximum(planes(subset(stats, mask), poses)[0][:, 0], 0.)
    return dict(cost=float(.5*values.mean()/SIGMA**2), rms_m=float(np.sqrt(values.mean())),
                per_patch_variance_m2=values.tolist(), patches=len(values))


def gradient(stats, poses, mask):
    if not np.any(mask):
        return None
    values, normal, center, wm, mass = planes(subset(stats, mask), poses)
    residual = np.einsum('kni,ki->kn', wm-center[:, None], normal)
    weight = np.sum(mass[:, 8:]*residual[:, 8:], axis=1)
    world = np.mean(normal*weight[:, None], axis=0)/SIGMA**2
    return poses[-1, :3, :3].T@world


def intersection_stats(points, labels, shared, keys):
    # Diagnostic restriction only: retain nonempty original patches, renormalize scan mass.
    records, kept = [], []
    for k in range(len(keys)):
        n = len(points)
        count, mu, second = np.zeros(n, np.int64), np.zeros((n, 3)), np.zeros((n, 3, 3))
        for i, p in enumerate(points):
            q = p[(labels[i] == k) & shared[i]]
            if len(q):
                count[i], mu[i], second[i] = len(q), q.mean(0), q.T@q/len(q)
        if count.sum():
            records.append((count, mu, second))
            kept.append(k)
    result = dict(counts=np.zeros((0, len(points)), np.int64), means=np.zeros((0, len(points), 3)), seconds=np.zeros((0, len(points), 3, 3)))
    if records:
        result = {name: np.stack([p[i] for p in records]) for i, name in enumerate(FIELDS[:3])}
    result['keys'] = keys[np.array(kept, np.int64)]
    return result, np.array(kept, np.int64)
