"""Joint measured point-plane optimization for an isolated development pilot."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def proper(T):
    out = np.array(T, dtype=float, copy=True)
    u, _, vh = np.linalg.svd(out[:3, :3])
    out[:3, :3] = u @ np.diag([1., 1., np.linalg.det(u @ vh)]) @ vh
    return out


def inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def retract(T, delta):
    # Product retraction: exact rigid update; derivative at zero is right tangent.
    update = np.eye(4)
    update[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix()
    update[:3, 3] = delta[3:]
    return T @ update


def normals(points):
    _, ids = cKDTree(points).query(points, k=20, workers=1)
    patches = points[ids]
    centered = patches - patches.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered) / 20.
    values, vectors = np.linalg.eigh(covariance)
    valid = (values[:, 0] / np.maximum(values.sum(1), 1e-15) < .05) & (values[:, 1] > 1e-6)
    return vectors[:, :, 0], valid


def edge_list(count):
    return [(a, b) for a in range(count) for b in range(count) if 1 <= abs(a-b) <= 2]


def associations(points, normal, valid, poses):
    trees = [cKDTree(p) for p in points]
    out = []
    for a, b in edge_list(len(points)):
        rel = inverse(poses[a]) @ poses[b]
        source = points[b] @ rel[:3, :3].T + rel[:3, 3]
        distances, ia = trees[a].query(source, k=1, workers=1)
        aligned_normals = normal[b] @ rel[:3, :3].T
        good = (distances < 1.) & valid[b] & valid[a][ia]
        good &= np.abs(np.einsum('ij,ij->i', aligned_normals, normal[a][ia])) >= np.cos(np.deg2rad(30.))
        ib = np.flatnonzero(good)
        if len(ib) < 50:
            raise ValueError(f'Insufficient measured support on directed edge {a}/{b}: {len(ib)}')
        out.append((a, b, ia[ib], ib))
    return out


def residual_jacobian(points, normal, poses, match):
    a, b, ia, ib = match
    rel = inverse(poses[a]) @ poses[b]
    p = points[b][ib]
    z = p @ rel[:3, :3].T + rel[:3, 3]
    n = normal[a][ia]
    residual = np.einsum('ij,ij->i', n, z-points[a][ia])
    nb = n @ rel[:3, :3]
    ja = np.concatenate((np.cross(n, z), -n), axis=1)
    jb = np.concatenate((np.cross(p, nb), nb), axis=1)
    return residual, ja, jb


def cost(points, normal, poses, matches):
    result = 0.
    for match in matches:
        r, _, _ = residual_jacobian(points, normal, poses, match)
        result += float(np.mean(.20**2*(np.sqrt(1+(r/.20)**2)-1)))
    return result


def system(points, normal, poses, matches):
    dimensions = 6*(len(poses)-1)
    h = np.zeros((dimensions, dimensions))
    g = np.zeros(dimensions)
    for match in matches:
        a, b = match[:2]
        r, ja, jb = residual_jacobian(points, normal, poses, match)
        weights = 1/np.sqrt(1+(r/.20)**2)/len(r)
        parts = [(node, jac) for node, jac in ((a, ja), (b, jb)) if node > 0]
        for node, jac in parts:
            block = slice(6*(node-1), 6*node)
            g[block] += jac.T @ (weights*r)
            for node2, jac2 in parts:
                block2 = slice(6*(node2-1), 6*node2)
                h[block, block2] += jac.T @ (weights[:, None]*jac2)
    return h, g


def optimize(points, normal, valid, initial, refresh, matches=None):
    poses = initial.copy()
    frozen = associations(points, normal, valid, initial) if matches is None else matches
    trace = []
    for outer in range(6):
        current = associations(points, normal, valid, poses) if refresh else frozen
        for inner in range(4):
            before = cost(points, normal, poses, current)
            h, g = system(points, normal, poses, current)
            diagonal = np.maximum(np.diag(h), 1e-8)
            step = -np.linalg.solve(h+np.diag(1e-3*diagonal), g).reshape(-1, 6)
            for delta in step:
                scale = max(1., np.linalg.norm(delta[:3])/np.deg2rad(1.), np.linalg.norm(delta[3:])/.20)
                delta /= scale
            accepted = 0.
            after = before
            for scale in (1., .5, .25, .125, .0625):
                trial = poses.copy()
                for node, delta in enumerate(step, start=1):
                    trial[node] = retract(poses[node], scale*delta)
                trial_cost = cost(points, normal, trial, current)
                if trial_cost <= before+1e-14:
                    poses, after, accepted = trial, trial_cost, scale
                    break
            eig = np.linalg.eigvalsh(h/np.sqrt(diagonal[:, None]*diagonal[None, :]))
            trace.append(dict(outer=outer, inner=inner, before=before, after=after,
                              accepted_scale=accepted, counts=[len(x[2]) for x in current],
                              normalized_hessian_min=float(eig[0]), normalized_hessian_max=float(eig[-1])))
    correction = np.array([inverse(a) @ b for a, b in zip(initial, poses)])
    max_t = float(np.max(np.linalg.norm(correction[:, :3, 3], axis=1)))
    max_r = float(np.max(Rotation.from_matrix(correction[:, :3, :3]).magnitude())*180/np.pi)
    if max_t > 1. or max_r > 5.:
        raise ValueError(f'Correction bound exceeded: {max_t} m/{max_r} deg')
    return poses, trace, dict(max_translation_correction_m=max_t, max_rotation_correction_deg=max_r)


def probe(points, poses):
    means, p95, coverage = [], [], []
    trees = [cKDTree(p) for p in points]
    for a, b in edge_list(len(points)):
        rel = inverse(poses[a]) @ poses[b]
        moved = points[b] @ rel[:3, :3].T + rel[:3, 3]
        distance, _ = trees[a].query(moved, k=1, workers=1)
        means.append(float(distance.mean()))
        p95.append(float(np.quantile(distance, .95)))
        coverage.append(float(np.mean(distance < 1.)))
    return dict(mean_nn_m=float(np.mean(means)), mean_pair_p95_m=float(np.mean(p95)),
                mean_1m_coverage=float(np.mean(coverage)), pair_mean_nn_m=means)
