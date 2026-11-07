"""Conventional shared-plane moment optimization for an isolated map pilot."""
import numpy as np
from scipy.spatial.transform import Rotation
from measured_map_reoptimization import inverse, retract

GEOMETRY_SIGMA = .10
PRIOR_T = .10
PRIOR_R = np.deg2rad(.15)


def skew(x):
    a, b, c = x
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def fit_planes(stats, poses):
    active = stats['counts'] > 0
    mass = active/active.sum(1)[:, None]
    means = stats['means']
    local_cov = stats['seconds']-means[..., :, None]*means[..., None, :]
    rotation = poses[None, :, :3, :3]
    world_mean = np.einsum('nij,knj->kni', poses[:, :3, :3], means)+poses[None, :, :3, 3]
    center = np.sum(mass[..., None]*world_mean, axis=1)
    difference = world_mean-center[:, None]
    world_cov = rotation@local_cov@rotation.transpose(0, 1, 3, 2)
    covariance = np.sum(mass[..., None, None]*(world_cov+difference[..., :, None]*difference[..., None, :]), axis=1)
    values, vectors = np.linalg.eigh(covariance)
    return values, vectors, center, world_mean, world_cov


def features(points, initial, scale):
    world = [p@t[:3, :3].T+t[:3, 3] for p, t in zip(points, initial)]
    keys = np.concatenate([np.floor(p/scale).astype(np.int64) for p in world])
    node = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(points)])
    rows = np.concatenate([np.arange(len(p)) for p in points])
    unique, group = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(group, kind='stable')
    splits = np.split(order, np.flatnonzero(np.diff(group[order]))+1)
    candidate, memberships, candidate_keys = [], [], []
    rejected = {'support': 0, 'plane': 0}
    n = len(points)
    for key, ids in zip(unique, splits):
        counts = np.zeros(n, np.int64)
        means = np.zeros((n, 3))
        seconds = np.zeros((n, 3, 3))
        members = []
        for i in range(n):
            selected = rows[ids[node[ids] == i]]
            if len(selected) < 6:
                continue
            p = points[i][selected]
            counts[i], means[i], seconds[i] = len(p), p.mean(0), p.T@p/len(p)
            members.append((i, selected))
        if np.sum(counts > 0) < 3 or counts.sum() < 40:
            rejected['support'] += 1
            continue
        stats = dict(counts=counts[None], means=means[None], seconds=seconds[None])
        value, vector, center, wm, _ = fit_planes(stats, initial)
        offset = (wm[0]-center[0])@vector[0, :, 0]
        good = value[0, 0]/max(value[0].sum(), 1e-15) < .025 and value[0, 1] > .005
        good = good and value[0, 0] <= .015 and np.max(abs(offset[counts > 0])) <= .20
        if not good:
            rejected['plane'] += 1
            continue
        candidate.append((counts, means, seconds))
        memberships.append(members)
        candidate_keys.append(key)
    labels = [np.full(len(p), -1, np.int64) for p in points]
    for patch, members in enumerate(memberships):
        for i, ids in members:
            labels[i][ids] = patch
    if not candidate:
        stats = dict(counts=np.zeros((0, n), np.int64), means=np.zeros((0, n, 3)), seconds=np.zeros((0, n, 3, 3)))
    else:
        stats = {name: np.stack([p[i] for p in candidate]) for i, name in enumerate(('counts', 'means', 'seconds'))}
    stats['keys'] = np.array(candidate_keys, dtype=np.int64).reshape(-1, 3)
    stats['labels'] = labels
    coverage = [int(np.sum(x >= 0)) for x in labels]
    info = dict(features=len(candidate), candidate_voxels=len(unique), rejected=rejected,
                per_scan_training_points=coverage, representation_valid=bool(len(candidate) >= 10 and min(coverage) >= 100))
    return stats, info


def plane_system(stats, poses):
    values, vectors, center, wm, wc = fit_planes(stats, poses)
    dimension = 6*(len(poses)-1)
    h = np.zeros((dimension, dimension))
    g = np.zeros(dimension)
    count = len(values)
    for k in range(count):
        normal = vectors[k, :, 0]
        tangent = vectors[k, :, 1:3].T
        hxx = np.zeros_like(h)
        hxf = np.zeros((dimension, 3))
        hff = np.zeros((3, 3))
        gx, gf = np.zeros(dimension), np.zeros(3)
        active = np.flatnonzero(stats['counts'][k] > 0)
        weight = 1/(count*len(active)*GEOMETRY_SIGMA**2)
        for i in active:
            mu, second = stats['means'][k, i], stats['seconds'][k, i]
            rotation, translation = poses[i, :3, :3], poses[i, :3, 3]
            nb = rotation.T@normal
            b = -skew(nb)
            l = np.vstack([tangent@rotation, np.zeros(3)])
            intercept = np.r_[tangent@(translation-center[k]), 1.]
            lp = l@mu
            rmean = normal@(wm[k, i]-center[k])
            cov = second-np.outer(mu, mu)
            hff += weight*(l@second@l.T+np.outer(lp, intercept)+np.outer(intercept, lp)+np.outer(intercept, intercept))
            gf += weight*(l@cov@nb+rmean*(lp+intercept))
            if i == 0:
                continue
            block = slice(6*(i-1), 6*i)
            bj = b@mu
            hh = np.block([[b@second@b.T, np.outer(bj, nb)], [np.outer(nb, bj), np.outer(nb, nb)]])
            hf = np.vstack([b@second@l.T+np.outer(bj, intercept), np.outer(nb, lp+intercept)])
            gg = np.r_[b@(cov@nb+rmean*mu), rmean*nb]
            hxx[block, block] += weight*hh
            hxf[block] += weight*hf
            gx[block] += weight*gg
        h += hxx-hxf@np.linalg.solve(hff, hxf.T)
        g += gx-hxf@np.linalg.solve(hff, gf)
    return (h+h.T)/2, g


def rotation_jacobian_inverse(phi, right):
    theta = np.linalg.norm(phi)
    k = skew(phi)
    coefficient = 1/12+theta**2/720 if theta < 1e-4 else 1/theta**2-(1+np.cos(theta))/(2*theta*np.sin(theta))
    return np.eye(3)+(.5 if right else -.5)*k+coefficient*k@k


def prior_edge(initial, poses, i, j):
    z = inverse(initial[i])@initial[j]
    relative = inverse(poses[i])@poses[j]
    error = inverse(z)@relative
    phi = Rotation.from_matrix(error[:3, :3]).as_rotvec()
    residual = np.r_[phi/PRIOR_R, error[:3, 3]/PRIOR_T]
    ja, jb = np.zeros((6, 6)), np.zeros((6, 6))
    ja[:3, :3] = -rotation_jacobian_inverse(phi, False)@z[:3, :3].T/PRIOR_R
    jb[:3, :3] = rotation_jacobian_inverse(phi, True)/PRIOR_R
    ja[3:, :3] = z[:3, :3].T@skew(relative[:3, 3])/PRIOR_T
    ja[3:, 3:] = -z[:3, :3].T/PRIOR_T
    jb[3:, 3:] = z[:3, :3].T@relative[:3, :3]/PRIOR_T
    return residual, ja, jb


def prior_system(initial, poses):
    dimension = 6*(len(poses)-1)
    h, g = np.zeros((dimension, dimension)), np.zeros(dimension)
    for i in range(len(poses)-1):
        j = i+1
        residual, ja, jb = prior_edge(initial, poses, i, j)
        full = np.zeros((6, dimension))
        if i:
            full[:, 6*(i-1):6*i] = ja
        full[:, 6*(j-1):6*j] = jb
        h += full.T@full/(len(poses)-1)
        g += full.T@residual/(len(poses)-1)
    return h, g


def cost(stats, poses, initial, prior):
    values = fit_planes(stats, poses)[0]
    geometry = .5*float(np.mean(np.maximum(values[:, 0], 0.)))/GEOMETRY_SIGMA**2
    motion = 0.
    if prior:
        motion = .5*np.mean([np.sum(prior_edge(initial, poses, i, i+1)[0]**2) for i in range(len(poses)-1)])
    return float(geometry+motion), dict(geometry=geometry, motion=float(motion))


def optimize(stats, initial, prior, start=None):
    poses = initial.copy() if start is None else start.copy()
    trace = []
    for iteration in range(24):
        before, _ = cost(stats, poses, initial, prior)
        h, g = plane_system(stats, poses)
        if prior:
            hp, gp = prior_system(initial, poses)
            h, g = h+hp, g+gp
        diagonal = np.maximum(np.diag(h), 1e-8)
        step = -np.linalg.solve(h+np.diag(1e-3*diagonal), g).reshape(-1, 6)
        for delta in step:
            delta /= max(1., np.linalg.norm(delta[:3])/np.deg2rad(1.), np.linalg.norm(delta[3:])/.20)
        accepted, after = 0., before
        for scale in (1., .5, .25, .125, .0625):
            trial = poses.copy()
            for i, delta in enumerate(step, start=1):
                trial[i] = retract(poses[i], scale*delta)
            current, _ = cost(stats, trial, initial, prior)
            if current <= before+1e-13:
                poses, after, accepted = trial, current, scale
                break
        eigenvalues = np.linalg.eigvalsh(h/np.sqrt(diagonal[:, None]*diagonal[None, :]))
        trace.append(dict(iteration=iteration, before=before, after=after, accepted_scale=accepted,
                          gradient_inf=float(np.max(abs(g))), normalized_hessian_min=float(eigenvalues[0]),
                          normalized_hessian_max=float(eigenvalues[-1])))
    corrections = np.stack([inverse(a)@b for a, b in zip(initial, poses)])
    t = float(np.max(np.linalg.norm(corrections[:, :3, 3], axis=1)))
    r = float(np.max(Rotation.from_matrix(corrections[:, :3, :3]).magnitude())*180/np.pi)
    return poses, trace, dict(max_translation_m=t, max_rotation_deg=r, valid=bool(t <= 1. and r <= 5.))


def probe_labels(points, initial, stats, scale):
    mapping = {tuple(key): k for k, key in enumerate(stats['keys'])}
    result = []
    for i, p in enumerate(points):
        world = p@initial[i, :3, :3].T+initial[i, :3, 3]
        labels = np.array([mapping.get(tuple(key), -1) for key in np.floor(world/scale).astype(np.int64)])
        good = labels >= 0
        labels[good & (stats['counts'][np.maximum(labels, 0), i] == 0)] = -1
        result.append(labels)
    return result


def probe(points, labels, stats, poses):
    values, vectors, center, wm, wc = fit_planes(stats, poses)
    squares = []
    node_counts = []
    for i, (p, group) in enumerate(zip(points, labels)):
        good = group >= 0
        world = p[good]@poses[i, :3, :3].T+poses[i, :3, 3]
        residual = np.sum(vectors[group[good], :, 0]*(world-center[group[good]]), axis=1)
        squares.append(residual**2)
        node_counts.append(int(good.sum()))
    loo = []
    for k in range(len(values)):
        active = np.flatnonzero(stats['counts'][k] > 0)
        for i in active:
            others = active[active != i]
            c = wm[k, others].mean(0)
            d = wm[k, others]-c
            covariance = (wc[k, others]+d[..., :, None]*d[..., None, :]).mean(0)
            _, v = np.linalg.eigh(covariance)
            n = v[:, 0]
            loo.append(float(n@wc[k, i]@n+(n@(wm[k, i]-c))**2))
    all_squared = np.concatenate(squares)
    return dict(odd_point_rms_m=float(np.sqrt(all_squared.mean())), odd_point_count=len(all_squared),
                per_scan_odd_point_count=node_counts, leave_one_scan_out_rms_m=float(np.sqrt(np.mean(loo))),
                leave_one_scan_out_records=len(loo), training_feature_count=len(values))
