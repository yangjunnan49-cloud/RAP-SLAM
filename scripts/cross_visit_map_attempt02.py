"""Isolated CV1 two-visit shared-plane solver; no fabricated motion-gap edge."""
import numpy as np
from scipy.spatial.transform import Rotation
import shared_plane_map as plane
from measured_map_reoptimization import inverse, retract


def edges(n, split=8):
    # A endpoint is node0; its other nodes are chronological1..split-1.
    return [(i, i+1) for i in range(1, split-1)]+[(split-1, 0)]+[(i, i+1) for i in range(split, n-1)]


def prior_cost(initial, poses, split=8):
    pairs = edges(len(poses), split)
    return .5*float(np.mean([np.sum(plane.prior_edge(initial, poses, i, j)[0]**2) for i, j in pairs]))


def prior_system(initial, poses, split=8):
    pairs = edges(len(poses), split)
    size = 6*(len(poses)-1)
    h, g = np.zeros((size, size)), np.zeros(size)
    for i, j in pairs:
        residual, ja, jb = plane.prior_edge(initial, poses, i, j)
        jac = np.zeros((6, size))
        for node, block in ((i, ja), (j, jb)):
            if node:
                jac[:, 6*(node-1):6*node] = block
        h += jac.T@jac/len(pairs)
        g += jac.T@residual/len(pairs)
    return h, g


def mapping(poses, split=8):
    result = np.zeros((6*(len(poses)-1), 6))
    endpoint = poses[-1]
    for i in range(split, len(poses)):
        local = inverse(endpoint)@poses[i]
        r = local[:3, :3].T
        adjoint = np.block([[r, np.zeros((3, 3))], [-r@plane.skew(local[:3, 3]), r]])
        result[6*(i-1):6*i] = adjoint
    return result


def system(stats, initial, poses, arm, split=8):
    h, g = plane.plane_system(stats, poses)
    if arm == 'F':
        m = mapping(poses, split)
        return m.T@h@m, m.T@g
    assert arm == 'J'
    hp, gp = prior_system(initial, poses, split)
    return h+hp, g+gp


def cost(stats, initial, poses, arm, split=8):
    value = plane.fit_planes(stats, poses)[0][:, 0]
    geometry = .5*float(np.mean(np.maximum(value, 0.)))/plane.GEOMETRY_SIGMA**2
    motion = prior_cost(initial, poses, split) if arm == 'J' else 0.
    return geometry+motion, dict(geometry=geometry, motion=motion)


def move(poses, step, arm, split=8):
    result = poses.copy()
    if arm == 'F':
        endpoint = poses[-1]
        next_endpoint = retract(endpoint, np.asarray(step).reshape(6))
        for i in range(split, len(poses)-1):
            result[i] = next_endpoint@(inverse(endpoint)@poses[i])
        # Do not recomposite E' @ rigid_inverse(E) @ E for the endpoint itself.
        # That numerically cubes existing rotation orthogonality error each update.
        result[-1] = next_endpoint
    else:
        assert arm == 'J'
        for i, delta in enumerate(np.asarray(step).reshape(-1, 6), start=1):
            result[i] = retract(poses[i], delta)
    return result


def correction(initial, poses, split=8):
    internal = []
    for ids, endpoint in ((range(split), 0), (range(split, len(poses)), len(poses)-1)):
        for i in ids:
            before = inverse(initial[endpoint])@initial[i]
            after = inverse(poses[endpoint])@poses[i]
            internal.append(inverse(before)@after)
    placement = inverse(initial[-1])@poses[-1]
    mt = float(np.max([np.linalg.norm(t[:3, 3]) for t in internal]))
    mr = float(np.max(Rotation.from_matrix(np.array(internal)[:, :3, :3]).magnitude())*180/np.pi)
    pt = float(np.linalg.norm(placement[:3, 3]))
    pr = float(Rotation.from_matrix(placement[:3, :3]).magnitude()*180/np.pi)
    return dict(internal_translation_m=mt, internal_rotation_deg=mr, endpoint_translation_m=pt,
                endpoint_rotation_deg=pr, valid=bool(mt <= 1. and mr <= 5. and pt <= 5. and pr <= 25.))


def optimize(stats, initial, arm, split=8):
    poses, trace = initial.copy(), []
    for iteration in range(24):
        before, _ = cost(stats, initial, poses, arm, split)
        h, g = system(stats, initial, poses, arm, split)
        diagonal = np.maximum(np.diag(h), 1e-8)
        step = -np.linalg.solve(h+np.diag(1e-3*diagonal), g).reshape(-1, 6)
        for delta in step:
            delta /= max(1., np.linalg.norm(delta[:3])/np.deg2rad(1.), np.linalg.norm(delta[3:])/.20)
        accepted, after = 0., before
        for scale in (1., .5, .25, .125, .0625):
            trial = move(poses, scale*step, arm, split)
            current, _ = cost(stats, initial, trial, arm, split)
            if current <= before+1e-13:
                poses, after, accepted = trial, current, scale
                break
        eigen = np.linalg.eigvalsh(h/np.sqrt(diagonal[:, None]*diagonal[None, :]))
        trace.append(dict(iteration=iteration, before=before, after=after, accepted_scale=accepted,
                          gradient_inf=float(np.max(abs(g))), normalized_hessian_min=float(eigen[0]),
                          normalized_hessian_max=float(eigen[-1])))
    return poses, trace, correction(initial, poses, split)
