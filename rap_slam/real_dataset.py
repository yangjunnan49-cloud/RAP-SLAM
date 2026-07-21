"""Real LiDAR sequence loader (KITTI-format datasets: KITTI odometry, POSS).

Loads a frame range of .bin velodyne scans plus ground-truth poses, converted
from the camera trajectory frame to the velodyne frame via the calibration
matrix (KITTI "Tr" / POSS "T").

Returns the same contract as sequence.generate_sequence():
    {"gt_poses": (N,4,4), "scans": [ (Ni,3) float32 ], "scan_files": [str]}
"""

import glob
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _read_calib_transform(calib_file: str) -> np.ndarray:
    """Read the lidar->camera 3x4 transform from a KITTI/POSS calib.txt."""
    with open(calib_file) as f:
        for line in f:
            if line.startswith(("Tr:", "T:")):
                vals = np.array([float(v) for v in line.split(":", 1)[1].split()])
                T = np.eye(4)
                T[:3, :] = vals.reshape(3, 4)
                return T
    raise ValueError(f"No Tr:/T: line found in {calib_file}")


def load_gt_poses(poses_file: str, calib_file: str,
                  start: int = 0, end: Optional[int] = None,
                  apply_calib: bool = False) -> np.ndarray:
    """Load GT poses [start..end].

    Args:
        apply_calib: when True (KITTI convention, poses are the camera
            trajectory), convert to the velodyne frame via
            T_velo_i = Tr @ T_cam_i @ Tr^-1. When False, poses are already in
            the LiDAR/world frame and used as-is (POSS convention — verified:
            the raw POSS pose trajectory is z-up horizontal; applying the
            calib similarity would stand it upright).
    """
    data = np.loadtxt(poses_file)
    if data.ndim == 1:
        data = data[None, :]
    poses = np.tile(np.eye(4), (len(data), 1, 1))
    poses[:, :3, :] = data.reshape(len(data), 3, 4)
    if apply_calib:
        Tr = _read_calib_transform(calib_file)
        Tr_inv = np.linalg.inv(Tr)
        poses = np.stack([Tr @ T @ Tr_inv for T in poses])
    end_idx = len(poses) - 1 if end is None else end
    return poses[start:end_idx + 1]


def load_scans(velodyne_dir: str, start: int = 0, end: Optional[int] = None,
               min_range: float = 0.0, max_range: float = 0.0,
               stride: int = 1) -> Tuple[List[np.ndarray], List[str]]:
    """Load .bin scans whose frame number (filename stem) is in [start..end].

    Selection is by absolute frame number, matching the GT pose indexing —
    the directory may hold only a segment of the full recording.
    """
    files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    end_idx = int(os.path.splitext(os.path.basename(files[-1]))[0]) if end is None else end
    selected = [p for p in files
                if start <= int(os.path.splitext(os.path.basename(p))[0]) <= end_idx]
    scans, used = [], []
    for path in selected[::stride]:
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
        if min_range > 0.0 or max_range > 0.0:
            r = np.linalg.norm(pts, axis=1)
            keep = np.ones(len(pts), dtype=bool)
            if min_range > 0.0:
                keep &= r >= min_range
            if max_range > 0.0:
                keep &= r <= max_range
            pts = pts[keep]
        scans.append(pts.astype(np.float32))
        used.append(path)
    logger.info(f"Loaded {len(scans)} scans from {velodyne_dir} "
                f"[{start}:{end_idx}:{stride}], "
                f"median pts/scan={int(np.median([len(s) for s in scans]))}")
    return scans, used


def load_real_sequence(root: str, sequence_id: str = "00",
                       start: int = 0, end: Optional[int] = None,
                       min_range: float = 0.0, max_range: float = 0.0,
                       stride: int = 1, apply_calib: bool = False) -> dict:
    """Load a KITTI-format real sequence segment.

    Args:
        root: dataset root containing dataset/poses/<seq>.txt and
            dataset/sequences/<seq>/{calib.txt,velodyne/}.
        sequence_id: e.g. "00".
        start, end: inclusive frame indices (end=None -> last).
        min_range, max_range: optional per-point range filter (0 disables).
        stride: take every k-th frame.
        apply_calib: True for KITTI odometry (cam-frame poses); False for
            POSS (poses already in the LiDAR/world frame).
    """
    poses_file = os.path.join(root, "dataset", "poses", f"{sequence_id}.txt")
    seq_dir = os.path.join(root, "dataset", "sequences", sequence_id)
    calib_file = os.path.join(seq_dir, "calib.txt")
    velodyne_dir = os.path.join(seq_dir, "velodyne")

    gt_poses = load_gt_poses(poses_file, calib_file, start=start, end=end,
                             apply_calib=apply_calib)
    scans, scan_files = load_scans(velodyne_dir, start=start, end=end,
                                   min_range=min_range, max_range=max_range,
                                   stride=stride)
    if stride > 1:
        gt_poses = gt_poses[::stride]
    assert len(gt_poses) == len(scans), \
        f"pose/scan count mismatch: {len(gt_poses)} vs {len(scans)}"
    logger.info(f"Real sequence: {len(scans)} frames "
                f"(GT from {poses_file}, converted to velodyne frame)")
    return {"gt_poses": gt_poses, "scans": scans, "scan_files": scan_files}
