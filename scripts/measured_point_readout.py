"""Isolated RLP1 frozen-context extraction and measured-point readout."""
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torch.nn import functional as F
from joint_token_extraction import extract

PAIRS = ((0, 1), (0, 2), (1, 2))


def contextual(model, batch, noise, independent=False, hook=False):
    controlled = dict(batch)
    if independent:
        controlled['points_per_part'] = torch.tensor([[1000], [1000], [1000]], device=noise.device)
        controlled['cu_seqlens'] = torch.tensor([0, 1000, 2000, 3000], device=noise.device)
        controlled['scales'] = batch['scales'].repeat(3)
    return extract(model, controlled, noise, export=True, hook=hook)


def proper(R):
    u, _, vt = np.linalg.svd(np.asarray(R, np.float64))
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def poses_from_camera(camera, calibration):
    C = np.asarray(calibration, np.float64)
    return np.linalg.inv(C)[None] @ camera @ C[None]


def relative_truth(poses):
    raw = np.stack([np.linalg.solve(poses[a], poses[b]) for a, b in PAIRS])
    angular = raw.copy()
    for t in angular:
        t[:3, :3] = proper(t[:3, :3])
    return raw, angular


def labels(points, poses):
    """Six directed set-positive masks, row classes and third-view support."""
    pp = np.split(np.asarray(points, np.float64), 3)
    out = {}
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            third = 3 - a - b
            T = np.linalg.solve(poses[a], poses[b])
            Tc = np.linalg.solve(poses[a], poses[third])
            moved = pp[b] @ T[:3, :3].T + T[:3, 3]
            support_points = pp[third] @ Tc[:3, :3].T + Tc[:3, 3]
            distance = cdist(pp[a], moved)
            distance = np.where(np.isfinite(distance), distance, np.inf)
            minimum = distance.min(1)
            support_distance = cdist(pp[a], support_points)
            supported = np.where(np.isfinite(support_distance), support_distance, np.inf).min(1) <= .60
            positive = distance <= .30
            cls = np.zeros(len(pp[a]), np.int8)
            cls[(minimum <= .30) & supported] = 1
            cls[(minimum >= .60) & supported] = 2
            out[f'positive_{a}{b}'] = np.packbits(positive, axis=1, bitorder='little')
            out[f'class_{a}{b}'] = cls
            out[f'support_{a}{b}'] = supported
            out[f'minimum_{a}{b}'] = minimum
    raw, angular = relative_truth(poses)
    out.update(raw_truth=raw, angular_truth=angular, lidar_poses=np.asarray(poses))
    return out


def class_counts(z):
    return {f'{a}{b}': dict(positive=int((z[f'class_{a}{b}'] == 1).sum()),
                            unmatched=int((z[f'class_{a}{b}'] == 2).sum()),
                            ignored=int((z[f'class_{a}{b}'] == 0).sum()))
            for a in range(3) for b in range(3) if a != b}


def objective_matrix(H):
    xx, xy, xz, yx, yy, yz, zx, zy, zz = H.reshape(-1).unbind()
    return torch.stack([torch.stack(row) for row in (
        (xx+yy+zz, yz-zy, zx-xz, xy-yx),
        (yz-zy, xx-yy-zz, xy+yx, zx+xz),
        (zx-xz, xy+yx, -xx+yy-zz, yz+zy),
        (xy-yx, zx+xz, yz+zy, -xx-yy+zz))])


def soft_pose(source, target, W):
    """W[target,source]; unique proper weighted objective, original coordinates."""
    source, target, W = source.double(), target.double(), W.double()
    mass = W.sum()
    if not torch.isfinite(mass):
        raise FloatingPointError('Nonfinite soft mass')
    if float(mass.detach()) <= 0:
        return None, 0.
    x = (W.sum(0) @ source) / mass
    y = (W.sum(1) @ target) / mass
    H = (source-x).T @ W.T @ (target-y)
    ev = torch.linalg.eigvalsh(objective_matrix(H).detach())
    gap = float(((ev[-1]-ev[-2]) / ev.abs().max().clamp_min(1e-12)).cpu())
    if gap <= 1e-6:
        return None, gap
    u, _, vh = torch.linalg.svd(H)
    sign = torch.linalg.det(vh.T @ u.T)
    correction = torch.diag(torch.stack([sign*0+1, sign*0+1, sign]))
    R = vh.T @ correction @ u.T
    translation = y-R@x
    return (R, translation), gap


def weighted_pose(source, target, weights):
    source, target, weights = map(lambda v: np.asarray(v, np.float64), (source, target, weights))
    mass = weights.sum()
    if not np.isfinite(mass) or mass <= 0 or len(weights) < 3:
        raise ValueError('Nonpositive or short pose support')
    x, y = weights @ source / mass, weights @ target / mass
    H = (source-x).T @ (weights[:, None] * (target-y))
    xx, xy, xz, yx, yy, yz, zx, zy, zz = H.ravel()
    K = np.array([[xx+yy+zz,yz-zy,zx-xz,xy-yx],
                  [yz-zy,xx-yy-zz,xy+yx,zx+xz],
                  [zx-xz,xy+yx,-xx+yy-zz,yz+zy],
                  [xy-yx,zx+xz,yz+zy,-xx-yy+zz]])
    eigen = np.linalg.eigvalsh(K)
    gap = (eigen[-1]-eigen[-2]) / max(np.abs(eigen).max(), 1e-12)
    if gap <= 1e-6:
        raise ValueError('Nonunique proper objective')
    u, _, vh = np.linalg.svd(H)
    R = vh.T @ np.diag([1., 1., np.linalg.det(vh.T @ u.T)]) @ u.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = y-R@x
    return T, float(gap)


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(521, 128)
        self.project = nn.Linear(128, 64)
        self.overlap = nn.Linear(128, 1)
        self.alpha = nn.Parameter(torch.zeros(()))
        assert sum(p.numel() for p in self.parameters()) == 75202

    def forward(self, feature, common):
        feature = F.normalize(feature, dim=-1, eps=1e-8)
        if feature.shape[-1] == 32:
            feature = F.pad(feature, (0, 480))
        hidden = F.silu(self.hidden(torch.cat([feature, common], -1)))
        return F.normalize(self.project(hidden), dim=-1, eps=1e-8), self.overlap(hidden).squeeze(-1)

    def probabilities(self, embedding, overlap, a, b):
        score = embedding[a] @ embedding[b].T / .10
        row = F.softmax(torch.cat([score, (self.alpha-overlap[a])[:, None]], 1), 1)
        reverse = F.softmax(torch.cat([score.T, (self.alpha-overlap[b])[:, None]], 1), 1)
        W = row[:, :-1] * reverse[:, :-1].T
        W = W * torch.sigmoid(overlap[a])[:, None] * torch.sigmoid(overlap[b])[None]
        return row, reverse, W


def common_channels(z):
    roles = np.zeros((3000, 1), np.float32); roles[:1000] = 1
    constants = np.tile(np.array([np.log(float(z['scale'])), .2], np.float32), (3000, 1))
    return np.concatenate([z['cond'], z['noise'], roles, constants], 1).reshape(3, 1000, 9)


def training_loss(head, feature, common, points, label, update):
    embedding, overlap = head(feature, common)
    corr_terms, motion_terms = [], []
    skipped = 0
    for pair_index, (a, b) in enumerate(PAIRS):
        row, reverse, W = head.probabilities(embedding, overlap, a, b)
        for probability, aa, bb in ((row, a, b), (reverse, b, a)):
            cls = label[f'class_{aa}{bb}']
            positive = label[f'positive_{aa}{bb}']
            terms = []
            mask = cls == 1
            if bool(mask.any()):
                p = (probability[mask, :-1] * positive[mask]).sum(1)
                terms.append(-p.clamp_min(1e-12).log().mean())
            mask = cls == 2
            if bool(mask.any()):
                terms.append(-probability[mask, -1].clamp_min(1e-12).log().mean())
            if terms:
                corr_terms.append(torch.stack(terms).mean())
        if update > 100:
            fitted, _ = soft_pose(points[b], points[a], W)
            if fitted is None:
                skipped += 1
            else:
                R, t = fitted
                truth = label['angular_truth'][pair_index]
                td = torch.linalg.vector_norm(t-truth[:3, 3]) / .20
                rd = torch.linalg.matrix_norm(R-truth[:3, :3]) / (np.sqrt(2.) * np.deg2rad(.2))
                motion_terms.append(F.huber_loss(td, torch.zeros_like(td), delta=1.) +
                                    F.huber_loss(rd, torch.zeros_like(rd), delta=1.))
    if not corr_terms:
        raise ValueError('No supervised rows in training window')
    corr = torch.stack(corr_terms).mean()
    motion = torch.stack(motion_terms).sum()/3 if motion_terms else corr*0
    return corr + .05*motion, dict(correspondence=float(corr.detach()),
                                 motion=float(motion.detach()), skipped=skipped)


def predict(head, feature, common, points):
    with torch.no_grad():
        embedding, overlap = head(feature, common)
        row, reverse, W = head.probabilities(embedding, overlap, 0, 1)
        row, reverse, W = [v.cpu().numpy() for v in (row, reverse, W)]
    target_ids = np.arange(1000)
    source_ids = row.argmax(1)
    valid = source_ids < 1000
    valid &= row[np.arange(1000), source_ids] >= .05
    clipped = np.minimum(source_ids, 999)
    valid &= reverse.argmax(1)[clipped] == target_ids
    valid &= reverse[clipped, target_ids] >= .05
    target_ids, source_ids = target_ids[valid], source_ids[valid]
    weights = W[target_ids, source_ids]
    arrays = dict(target_ids=target_ids, source_ids=source_ids, weights=weights)
    if len(weights) < 20:
        return None, dict(reason='fewer than 20 mutual matches', **arrays)
    target, source = points[0][target_ids], points[1][source_ids]
    try:
        pose, gap = weighted_pose(source, target, weights)
        history = [dict(count=len(weights), gap=gap)]
        keep = np.ones(len(weights), bool)
        for _ in range(2):
            distance = np.linalg.norm(source@pose[:3, :3].T+pose[:3, 3]-target, axis=1)
            keep &= distance <= .60
            pose, gap = weighted_pose(source[keep], target[keep], weights[keep])
            history.append(dict(count=int(keep.sum()), gap=gap))
        arrays.update(keep=keep, pose_history_counts=np.array([x['count'] for x in history]),
                      pose_history_gaps=np.array([x['gap'] for x in history]))
        return pose, dict(reason=None, **arrays)
    except ValueError as ex:
        return None, dict(reason=str(ex), **arrays)
