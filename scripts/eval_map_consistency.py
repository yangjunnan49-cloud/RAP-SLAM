"""P3b — map-consistency axis: a second evaluation dimension next to
trajectory APE, for the P3 set-level matrix.

For each golden dataset, merge the KEYFRAME scans at three pose sets —
odometry, PGO (canonical/current backend), and GT — and report:

  chamfer_mean / chamfer_p95: symmetric nearest-neighbor distance (mean /
      95th percentile, m) between the estimated map and the GT map
      (map-space error; complements APE which is trajectory-space).
  crisp_mean / crisp_p95: GT-free map crispness — for every keyframe
      scan's points, distance to the nearest map point from a DIFFERENT
      scan (leave-one-scan-out), capped at 1 m. Measures double-vision /
      blur without needing GT.

Maps: per-scan voxel 0.4 m, merged, per-observation voxel 0.3 m, random-capped at
500k points (seed 0). cKDTree queries; CPU-only.

Run (WSL2 conda rap-inference):
  python scripts/eval_map_consistency.py --out output_map_consistency_v2
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from eval_paper_metrics import gt_for_live, gt_from_summary  # noqa: E402
from eval_calibrated_weighting import GOLDENS, load_poses_auto  # noqa: E402
from eval_verifier_swap import ScanProvider  # noqa: E402
from rap_slam.evaluate import align_to_first_pose  # noqa: E402

logger = logging.getLogger("eval_map_consistency")

# golden -> PGO trajectory file (current-backend canonical)
PGO_FILE = {
    "poss_off": "output_prods/poss/optimized_poses.txt",
    "kitti06_off": "output_prods/kitti06/optimized_poses.txt",
    "kitti05": "output_prods/kitti05/optimized_poses.txt",
    "kitti00_bb": "output_prods/kitti00_bb/optimized_poses.txt",
    "kitti02_live": "output_ros2/step3_kitti02/map_poses_current.txt",
    "kitti06_live": "output_ros2/step3_kitti06/map_poses_current.txt",
    "poss_live": "output_ros2/step3/map_poses_current.txt",
}


def voxel_down(points: np.ndarray, voxel: float) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def build_map(scans: ScanProvider, keyframes: List[int],
              poses: List[np.ndarray], cap: int = 500_000,
              stride_max: int = 4):
    """Merge keyframe scans at the given poses. Returns (points, labels)."""
    stride = max(1, int(np.ceil(len(keyframes) / 400)))  # <=400 scans
    stride = min(stride, stride_max)
    clouds, labels = [], []
    for li, idx in enumerate(keyframes[::stride]):
        if idx >= len(poses):
            continue
        pts = scans.raw(idx)  # raw; per-scan voxel below
        pts = voxel_down(pts.astype(np.float64), 0.4)
        T = poses[idx]
        pts = (T[:3, :3] @ pts.T).T + T[:3, 3]
        clouds.append(pts.astype(np.float32))
        labels.append(np.full(len(pts), li, dtype=np.int32))
    all_pts = np.concatenate(clouds)
    all_lab = np.concatenate(labels)
    keys = np.column_stack((all_lab, np.floor(all_pts / 0.3).astype(np.int64)))
    _, sel = np.unique(keys, axis=0, return_index=True)
    all_pts, all_lab = all_pts[sel], all_lab[sel]
    if len(all_pts) > cap:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(all_pts), cap, replace=False)
        all_pts, all_lab = all_pts[sel], all_lab[sel]
    return all_pts, all_lab


def chamfer(a: np.ndarray, b: np.ndarray):
    from scipy.spatial import cKDTree
    d1, _ = cKDTree(b).query(a, k=1)
    d2, _ = cKDTree(a).query(b, k=1)
    d = np.concatenate([d1, d2])
    return float(np.mean(d)), float(np.quantile(d, 0.95))


def crispness(pts: np.ndarray, labels: np.ndarray, cap_r: float = 1.0):
    """Exact capped nearest distance to a different scan in the retained map.

    Adaptive queries terminate once a foreign observation is found, the cap
    radius is exhausted, or all retained points have been searched. The input
    map may be sampled; this is not an independent survey-reference metric.
    """
    from scipy.spatial import cKDTree
    pts, labels = np.asarray(pts), np.asarray(labels)
    if pts.shape != (len(labels), 3) or not len(pts) or not np.isfinite(pts).all():
        raise ValueError("Expected nonempty finite points and one label per point")
    if not np.isfinite(cap_r) or cap_r <= 0:
        raise ValueError("cap_r must be positive and finite")
    distances = np.full(len(pts), cap_r, dtype=float)
    if len(np.unique(labels)) > 1:
        tree = cKDTree(pts)
        pending = np.arange(len(pts))
        k = min(8, len(pts))
        while len(pending):
            # Bound memory even for dense single-scan neighborhoods.
            next_pending = []
            chunk_size = max(1, 1_000_000 // k)
            for begin in range(0, len(pending), chunk_size):
                query = pending[begin:begin + chunk_size]
                d, idx = tree.query(pts[query], k=k, distance_upper_bound=cap_r)
                d, idx = np.asarray(d).reshape(len(query), k), np.asarray(idx).reshape(len(query), k)
                valid = idx < len(pts)
                foreign = valid & (labels[np.minimum(idx, len(pts) - 1)] != labels[query, None])
                best = np.min(np.where(foreign, d, cap_r), axis=1)
                distances[query] = best
                unresolved = (~foreign.any(axis=1)) & valid.all(axis=1)
                if k < len(pts):
                    next_pending.extend(query[unresolved])
            pending = np.asarray(next_pending, dtype=int)
            k = min(k * 2, len(pts))
    return float(np.mean(distances)), float(np.quantile(distances, 0.95))



def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: "
                               "%(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New output directory; existing paths are rejected")
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=False)

    lines = ["# P3b — map-consistency axis (odometry vs PGO; GT map as "
             "reference)", ""]
    lines.append("| golden | traj | chamfer mean / p95 [m] | crispness "
                 "mean / p95 [m] |")
    lines.append("|---|---|---|---|")
    report: Dict[str, dict] = {}
    for name, (odom_f, kf_f, _, source, seq, _nms) in GOLDENS.items():
        if source:
            # live golden: stream frame 0 = dataset frame `offset`
            # (POSS live plays segA starting at POSS frame 3850;
            #  KITTI live plays the full sequence from frame 0)
            cfg = {"source": source, "seq": seq,
                   "offset": 3850 if source == "poss" else 0}
        else:
            # offline golden: dataset identity + offset from summary config
            d = os.path.dirname(os.path.join(REPO_ROOT, odom_f))
            c = json.load(open(os.path.join(d, "summary.json")))["config"]
            if c.get("real_seq") == "00" and not c.get("real_apply_calib"):
                # POSS segA (KITTI-00-format poses, Velodyne-32e scans)
                cfg = {"source": "poss", "seq": "A",
                       "offset": int(c.get("real_start", 3850))}
            else:
                cfg = {"source": "kitti", "seq": c["real_seq"],
                       "offset": int(c.get("real_start", 0))}
        scans = ScanProvider(cfg)
        keyframes = json.load(open(os.path.join(REPO_ROOT, kf_f)))
        odom = load_poses_auto(os.path.join(REPO_ROOT, odom_f))
        pgo = load_poses_auto(os.path.join(REPO_ROOT, PGO_FILE[name]))
        if source:
            gt = gt_for_live(source, seq, len(odom))
        else:
            gt = gt_from_summary(os.path.dirname(os.path.join(
                REPO_ROOT, odom_f)))
        # Same gauge convention as the unaligned-APE metric: anchor all
        # pose sets at their own first pose (POSS GT lives in a global
        # frame ~454 m away from the odometry frame; KITTI GT is already
        # first-frame-anchored, odom/pgo start at identity).
        odom = align_to_first_pose(odom)
        pgo = align_to_first_pose(pgo)
        gt = align_to_first_pose(gt)
        report[name] = {}
        maps = {}
        for label, poses in (("odometry", odom), ("pgo", pgo), ("gt", gt)):
            logger.info(f"[{name}] building {label} map "
                        f"({len(keyframes)} keyframes)")
            maps[label] = build_map(scans, keyframes, poses)
        for label in ("odometry", "pgo"):
            cm, c95 = chamfer(maps[label][0], maps["gt"][0])
            sm, s95 = crispness(*maps[label])
            report[name][label] = {"chamfer_mean": cm, "chamfer_p95": c95,
                                   "crisp_mean": sm, "crisp_p95": s95}
            logger.info(f"[{name}][{label}] chamfer {cm:.3f}/{c95:.3f} m, "
                        f"crisp {sm:.3f}/{s95:.3f} m")
            lines.append(f"| {name} | {label} | {cm:.3f} / {c95:.3f} | "
                         f"{sm:.3f} / {s95:.3f} |")
    with open(os.path.join(out_dir, "map_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "map_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote {out_dir}/map_report.{{md,json}}")


if __name__ == "__main__":
    main()
