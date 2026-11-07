"""Pure translation CV2 solver, with every initial scan rotation copied exactly."""
import numpy as np
from scipy.spatial.transform import Rotation
import shared_plane_map as plane
import cross_visit_map_attempt02 as rigid


def mapping(poses, initial, split=8):
    m = np.zeros((6*(len(poses)-1),3))
    endpoint_r = initial[-1,:3,:3]
    for i in range(split,len(poses)):
        m[6*(i-1)+3:6*i] = poses[i,:3,:3].T@endpoint_r
    return m


def system(stats, initial, poses, split=8):
    h,g = plane.plane_system(stats,poses)
    m = mapping(poses,initial,split)
    return m.T@h@m,m.T@g


def cost(stats,poses):
    values = np.maximum(plane.fit_planes(stats,poses)[0][:,0],0.)
    return .5*float(np.mean(values))/plane.GEOMETRY_SIGMA**2


def move(poses,initial,delta,split=8):
    result = poses.copy()
    displacement = initial[-1,:3,:3]@np.asarray(delta).reshape(3)
    result[split:,:3,3] += displacement
    result[:,:3,:3] = initial[:,:3,:3]
    return result


def correction(initial,poses,split=8):
    output = rigid.correction(initial,poses,split)
    relative = initial[:,:3,:3].transpose(0,2,1)@poses[:,:3,:3]
    change = float(np.max(Rotation.from_matrix(relative).magnitude())*180/np.pi)
    exact = bool(np.array_equal(initial[:,:3,:3],poses[:,:3,:3]))
    output.update(rotation_byte_identical=exact,max_scan_rotation_change_deg=change,
                  valid=bool(exact and change <= 1e-8 and output['endpoint_translation_m'] <= 5. and
                             output['internal_translation_m'] <= 1e-9 and output['internal_rotation_deg'] <= 1e-8))
    return output


def optimize(stats,initial,split=8):
    poses,trace = initial.copy(),[]
    for iteration in range(24):
        before = cost(stats,poses)
        h,g = system(stats,initial,poses,split)
        diagonal = np.maximum(np.diag(h),1e-8)
        step = -np.linalg.solve(h+np.diag(1e-3*diagonal),g)
        step /= max(1.,np.linalg.norm(step)/.20)
        accepted,after = 0.,before
        for scale in (1.,.5,.25,.125,.0625):
            trial = move(poses,initial,scale*step,split)
            current = cost(stats,trial)
            if current <= before+1e-13:
                poses,after,accepted = trial,current,scale
                break
        eigen = np.linalg.eigvalsh(h/np.sqrt(diagonal[:,None]*diagonal[None,:]))
        trace.append(dict(iteration=iteration,before=before,after=after,accepted_scale=accepted,
                          gradient_inf=float(np.max(abs(g))),normalized_hessian_min=float(eigen[0]),
                          normalized_hessian_max=float(eigen[-1])))
    return poses,trace,correction(initial,poses,split)
