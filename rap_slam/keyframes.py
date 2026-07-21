"""Keyframe selection by cumulative displacement."""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def _yaw_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def select_keyframes(poses: List[np.ndarray], trans_thresh: float = 6.0,
                     rot_thresh_deg: float = 12.0) -> List[int]:
    """Select keyframe indices where cumulative displacement exceeds thresholds.

    The first and last scans are always keyframes.

    Args:
        poses: full pose sequence (sensor-to-world).
        trans_thresh: cumulative translation threshold (meters).
        rot_thresh_deg: cumulative yaw threshold (degrees).

    Returns:
        Sorted list of keyframe scan indices.
    """
    n = len(poses)
    if n == 0:
        return []
    keyframes = [0]
    acc_trans = 0.0
    acc_rot = 0.0
    prev_yaw = _yaw_of(poses[0])
    for i in range(1, n):
        step = np.linalg.norm(poses[i][:3, 3] - poses[i - 1][:3, 3])
        yaw = _yaw_of(poses[i])
        dyaw = (yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi
        prev_yaw = yaw
        acc_trans += step
        acc_rot += abs(dyaw)
        if acc_trans > trans_thresh or acc_rot > np.deg2rad(rot_thresh_deg):
            keyframes.append(i)
            acc_trans = 0.0
            acc_rot = 0.0
    if keyframes[-1] != n - 1:
        keyframes.append(n - 1)
    logger.info(f"Selected {len(keyframes)} keyframes from {n} scans "
                f"(trans>{trans_thresh} m or rot>{rot_thresh_deg} deg)")
    return keyframes
