"""Oxford Spires Dataset loader (P4 cross-sensor zero-shot probe).

Sensor: Hesai QT64 (64 ch, 104 deg VFoV, 60 m range) on a backpack rig,
10 Hz rolling-shutter sweeps. Raw clouds: raw/lidar-clouds/<t>.pcd
(PCD binary, x/y/z/intensity + possibly more fields — only x/y/z used).
GT: processed/trajectory/gt-tum.txt (TUM: ts tx ty tz qx qy qz qw,
~1-2 cm TLS-registered accuracy, base/rig frame in the TLS map frame).

Timestamp matching: PCD filename stem is the sweep-start epoch (seconds,
with nanosecond decimals); GT rows are matched by nearest timestamp
(max |dt| asserted < 0.05 s at 10 Hz).

Caveats (documented in the paper's zero-shot experiment):
- GT poses are the rig BASE frame; KISS-ICP estimates the LiDAR frame.
  After first-pose anchoring both conventions agree up to the constant
  base->lidar extrinsic (small lever arm); we do NOT apply the calib.
- Raw clouds are rolling-shutter (not deskewed) — consistent with the
  KITTI/POSS arms of the study (deskew was falsified separately).
"""

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# PCD header parsing (binary payload only)
_PCD_HEADER_RE = re.compile(rb"^(.*?)DATA binary\n", re.DOTALL)


def read_pcd_xyz(path: str) -> np.ndarray:
    """Read a binary PCD and return (N, 3) float32 xyz."""
    with open(path, "rb") as f:
        raw = f.read()
    m = _PCD_HEADER_RE.match(raw)
    if not m:
        raise ValueError(f"{path}: no 'DATA binary' header")
    header = m.group(1).decode("ascii", errors="strict")
    fields = sizes = types = counts = None
    n_points = None
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(x) for x in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(x) for x in parts[1:]]
        elif key == "POINTS":
            n_points = int(parts[1])
    if fields is None or n_points is None:
        raise ValueError(f"{path}: incomplete PCD header")
    if sizes is None:
        sizes = [4] * len(fields)
    if types is None:
        types = ["F"] * len(fields)
    if counts is None:
        counts = [1] * len(fields)

    # build a structured dtype for the payload
    np_types = []
    offset = 0
    for name, sz, tp, cnt in zip(fields, sizes, types, counts):
        if tp == "F":
            base = {4: np.float32, 8: np.float64}[sz]
        elif tp == "I":
            base = {1: np.int8, 2: np.int16, 4: np.int32}[sz]
        elif tp == "U":
            base = {1: np.uint8, 2: np.uint16, 4: np.uint32}[sz]
        else:
            raise ValueError(f"{path}: unknown PCD type {tp}")
        np_types.append((name, base, (cnt,) if cnt > 1 else ()))
    dt = np.dtype(np_types)
    payload = np.frombuffer(raw, dtype=dt, count=n_points,
                            offset=m.end())
    xyz = np.stack([payload["x"], payload["y"], payload["z"]],
                   axis=1).astype(np.float32)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return xyz


def load_gt_tum(path: str) -> Dict[float, np.ndarray]:
    """TUM -> {timestamp: 4x4 pose}."""
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None, :]
    out = {}
    for r in rows:
        ts = float(r[0])
        t = r[1:4]
        qx, qy, qz, qw = r[4:8]
        n = qx * qx + qy * qy + qz * qz + qw * qw
        s = 2.0 / n
        R = np.array([
            [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw),
             s * (qx * qz + qy * qw)],
            [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz),
             s * (qy * qz - qx * qw)],
            [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw),
             1 - s * (qx * qx + qy * qy)]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        out[ts] = T
    return out


def _stamp_from_name(path: str) -> float:
    return float(os.path.splitext(os.path.basename(path))[0])


def load_spires_sequence(root: str, seq: str,
                         start: int = 0,
                         end: Optional[int] = None,
                         min_range: float = 1.5,
                         max_range: float = 0.0,
                         ) -> Dict[str, object]:
    """Load one Oxford Spires sequence as pipeline-ready scans + GT.

    Args:
        root: dataset root containing sequences/<seq>/raw/lidar-clouds/*.pcd
              and gt/<seq>.txt (TUM, downloaded separately).
        start/end: scan-index slice AFTER sorting by timestamp.
        min_range/max_range: drop points outside [min, max(0=off)] m.

    Returns:
        {"scans": list[(N,3) float32], "gt_poses": list[4x4],
         "stamps": np.ndarray, "gt_stamps": np.ndarray}
    """
    cloud_dir = os.path.join(root, "sequences", seq, "raw", "lidar-clouds")
    gt_file = os.path.join(root, "gt", f"{seq}.txt")
    files = sorted(glob.glob(os.path.join(cloud_dir, "*.pcd")),
                   key=_stamp_from_name)
    if not files:
        raise FileNotFoundError(f"no PCDs under {cloud_dir}")
    gt = load_gt_tum(gt_file)
    gt_ts = np.array(sorted(gt.keys()))

    end_idx = len(files) - 1 if end is None else end
    files = files[start:end_idx + 1]
    scans: List[np.ndarray] = []
    gt_poses: List[np.ndarray] = []
    stamps, matched_gt_ts = [], []
    for path in files:
        ts = _stamp_from_name(path)
        k = int(np.argmin(np.abs(gt_ts - ts)))
        dt = abs(float(gt_ts[k]) - ts)
        if dt > 0.05:
            raise ValueError(f"{os.path.basename(path)}: nearest GT {dt:.3f} s "
                             f"away (> 0.05 s) — GT/scan rate mismatch")
        pts = read_pcd_xyz(path)
        if min_range > 0 or max_range > 0:
            r = np.linalg.norm(pts, axis=1)
            keep = np.ones(len(pts), dtype=bool)
            if min_range > 0:
                keep &= r >= min_range
            if max_range > 0:
                keep &= r <= max_range
            pts = pts[keep]
        scans.append(pts)
        gt_poses.append(gt[float(gt_ts[k])])
        stamps.append(ts)
        matched_gt_ts.append(float(gt_ts[k]))
    return {"scans": scans, "gt_poses": gt_poses,
            "stamps": np.asarray(stamps),
            "gt_stamps": np.asarray(matched_gt_ts)}
