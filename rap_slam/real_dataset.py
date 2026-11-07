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
from .trajectory_protocol import validate_ids, validate_stamps

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
            T_velo_i = Tr^-1 @ T_cam_i @ Tr. When False, poses are already in
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
        # Camera-trajectory -> velodyne-trajectory (world becomes velo_0).
        # NOTE: the inverse order (Tr @ T @ Tr^-1) is WRONG here — it leaves
        # the trajectory in a ~90-deg-rotated world frame (verified on KITTI
        # seq 04: APE mean 267 m vs 2.1 m after the fix).
        poses = np.stack([Tr_inv @ T @ Tr for T in poses])
    end_idx = len(poses) - 1 if end is None else end
    return poses[start:end_idx + 1]


def kitti_vertical_correction(points: np.ndarray,
                              angle_deg: float = 0.205) -> np.ndarray:
    """KITTI HDL-64 intrinsic vertical-angle correction (S0b).

    Exact CT-ICP convention (jedeschaud/ct_icp src/ct_icp/dataset.cpp,
    KITTI_GLOBAL_VERTICAL_ANGLE_OFFSET = 0.205, "Issue in the intrinsic
    calibration of the KITTI Velodyne HDL64"; same convention in MULLS /
    Traj-LO): every point p is rotated by +angle about the horizontal axis
    a = normalize(p x z_hat) — i.e. the axis perpendicular to the point's
    azimuth direction. Since a is perpendicular to p, Rodrigues reduces to
        p' = p cos(t) + (-x*z/rho, -y*z/rho, rho) sin(t),
    rho = sqrt(x^2+y^2). Points with rho ~ 0 (near-vertical) are untouched.
    """
    t = np.deg2rad(angle_deg)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rho = np.hypot(x, y)
    safe = rho > 1e-6
    c, s = np.cos(t), np.sin(t)
    out = points.copy()
    sxz = np.where(safe, x * z / np.maximum(rho, 1e-6), 0.0)
    syz = np.where(safe, y * z / np.maximum(rho, 1e-6), 0.0)
    sz = np.where(safe, rho, 0.0)
    out[:, 0] = np.where(safe, x * c - sxz * s, x)
    out[:, 1] = np.where(safe, y * c - syz * s, y)
    out[:, 2] = np.where(safe, z * c + sz * s, z)
    return out


def load_scans(velodyne_dir: str, start: int = 0, end: Optional[int] = None,
               min_range: float = 0.0, max_range: float = 0.0,
               stride: int = 1, kitti205: bool = False) -> Tuple[List[np.ndarray], List[str]]:
    """Load .bin scans whose frame number (filename stem) is in [start..end].

    Selection is by absolute frame number, matching the GT pose indexing —
    the directory may hold only a segment of the full recording.
    """
    if not isinstance(stride, int) or stride < 1 or start < 0:
        raise ValueError("start must be nonnegative and stride must be a positive integer")
    files = glob.glob(os.path.join(velodyne_dir, "*.bin"))
    by_id = {}
    for path in files:
        frame = int(os.path.splitext(os.path.basename(path))[0])
        if frame in by_id:
            raise ValueError(f"Duplicate source frame {frame}")
        by_id[frame] = path
    if not by_id:
        raise ValueError(f"No scans in {velodyne_dir}")
    end_idx = max(by_id) if end is None else end
    if end_idx < start:
        raise ValueError("Empty or reversed source frame range")
    selected_ids = range(start, end_idx + 1, stride)
    missing = [i for i in selected_ids if i not in by_id]
    if missing:
        raise ValueError(f"Missing requested source frames: {missing[:10]}")
    selected = [by_id[i] for i in selected_ids]
    scans, used = [], []
    for path in selected:
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
        if min_range > 0.0 or max_range > 0.0:
            r = np.linalg.norm(pts, axis=1)
            keep = np.ones(len(pts), dtype=bool)
            if min_range > 0.0:
                keep &= r >= min_range
            if max_range > 0.0:
                keep &= r <= max_range
            pts = pts[keep]
        if kitti205:
            pts = kitti_vertical_correction(pts)
        scans.append(pts.astype(np.float32))
        used.append(path)
    logger.info(f"Loaded {len(scans)} scans from {velodyne_dir} "
                f"[{start}:{end_idx}:{stride}], "
                f"median pts/scan={int(np.median([len(s) for s in scans]))}")
    return scans, used


def load_real_sequence(root: str, sequence_id: str = "00",
                       start: int = 0, end: Optional[int] = None,
                       min_range: float = 0.0, max_range: float = 0.0,
                       stride: int = 1, apply_calib: bool = False,
                       kitti205: bool = False) -> dict:
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
        kitti205: apply the 0.205-deg intrinsic vertical-angle correction
            (CT-ICP/MULLS/Traj-LO KITTI convention; KITTI HDL-64 ONLY —
            never for POSS or other sensors).
    """
    poses_file = os.path.join(root, "dataset", "poses", f"{sequence_id}.txt")
    seq_dir = os.path.join(root, "dataset", "sequences", sequence_id)
    calib_file = os.path.join(seq_dir, "calib.txt")
    velodyne_dir = os.path.join(seq_dir, "velodyne")

    gt_poses = load_gt_poses(poses_file, calib_file, start=0, end=None,
                             apply_calib=apply_calib)
    scans, scan_files = load_scans(velodyne_dir, start=start, end=end,
                                   min_range=min_range, max_range=max_range,
                                   stride=stride, kitti205=kitti205)
    source_frame_ids = validate_ids([int(os.path.splitext(os.path.basename(p))[0]) for p in scan_files])
    if source_frame_ids[-1] >= len(gt_poses):
        raise ValueError("Source frame outside GT pose table")
    gt_poses = gt_poses[source_frame_ids]
    timestamps = None
    times_file = os.path.join(seq_dir, "times.txt")
    if os.path.exists(times_file):
        all_times = validate_stamps(np.atleast_1d(np.loadtxt(times_file)))
        if source_frame_ids[-1] >= len(all_times):
            raise ValueError("Source frame outside timestamp table")
        timestamps = all_times[source_frame_ids].tolist()
    logger.info(f"Real sequence: {len(scans)} frames "
                f"(GT from {poses_file}, converted to velodyne frame)")
    return {"gt_poses": gt_poses, "scans": scans, "scan_files": scan_files,
            "source_frame_ids": source_frame_ids.tolist(), "timestamps": timestamps}
