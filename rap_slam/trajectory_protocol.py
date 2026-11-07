"""Small, explicit trajectory contracts and KITTI-style segment statistics."""
import numpy as np


def validate_ids(ids, n=None):
    values = np.asarray(ids)
    if (values.ndim != 1 or not len(values) or
            not np.issubdtype(values.dtype, np.integer) or
            np.any(values < 0) or np.any(np.diff(values) <= 0) or
            (n is not None and len(values) != n)):
        raise ValueError('Source frame IDs must be nonempty, increasing, unique integers matching pose count')
    return values.astype(np.int64)


def validate_stamps(stamps, n=None):
    values = np.asarray(stamps, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0) or (n is not None and len(values) != n)):
        raise ValueError('Timestamps must be finite, strictly increasing and match pose count')
    return values


def validate_pair(gt, est):
    gt, est = np.asarray(gt, dtype=float), np.asarray(est, dtype=float)
    if gt.shape != est.shape or gt.ndim != 3 or gt.shape[1:] != (4, 4) or len(gt) < 2:
        raise ValueError('GT and estimate must have identical (N,4,4) shapes with N >= 2; no truncation')
    if not np.isfinite(gt).all() or not np.isfinite(est).all():
        raise ValueError('Nonfinite trajectory')
    return gt, est


def kitti_drift(gt, est, step=10, source_frame_ids=None):
    """All-segment mean, strict > endpoint, inv(delta_est) @ delta_gt.

    KITTI devkit uses nominal 10 Hz, including both endpoint frames for speed.
    Source IDs preserve that time span under stride; sparse trajectories remain
    an explicitly custom protocol (their path and start grid differ from KITTI).
    Coordinate reference is determined by the caller, never inferred here.
    """
    gt, est = validate_pair(gt, est)
    if not isinstance(step, int) or step <= 0:
        raise ValueError('step must be a positive integer')
    ids = validate_ids(np.arange(len(gt)) if source_frame_ids is None else source_frame_ids, len(gt))
    cum = np.r_[0., np.cumsum(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))]
    inv_gt, inv_est = np.linalg.inv(gt), np.linalg.inv(est)
    segments = []
    for first in range(0, len(gt), step):
        for length in range(100, 801, 100):
            last = int(np.searchsorted(cum, cum[first] + length, side='right'))
            if last >= len(gt):
                continue
            error = np.linalg.inv(inv_est[first] @ est[last]) @ (inv_gt[first] @ gt[last])
            trans = float(np.linalg.norm(error[:3, 3]) / length)
            rot = float(np.arccos(np.clip((np.trace(error[:3, :3]) - 1) / 2, -1, 1)) / length)
            segments.append(dict(first=first, last=last, source_first=int(ids[first]),
                                 source_last=int(ids[last]), length_m=length,
                                 translation_fraction=trans, rotation_rad_per_m=rot,
                                 speed_mps=float(length / (.1 * (ids[last] - ids[first] + 1)))))
    return dict(kitti_ter_pct=float(np.mean([s['translation_fraction'] for s in segments]) * 100) if segments else float('nan'),
                kitti_rer_deg100m=float(np.mean([s['rotation_rad_per_m'] for s in segments]) * 180 / np.pi * 100) if segments else float('nan'),
                kitti_segments=len(segments), kitti_segment_records=segments,
                kitti_protocol='all_segments_strict_endpoint_v2',
                kitti_sampling='dense' if np.all(np.diff(ids) == 1) else 'custom_sparse',
                kitti_speed_convention='nominal_10Hz_inclusive_source_frame_span')
