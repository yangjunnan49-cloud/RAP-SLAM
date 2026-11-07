"""Paper-metrics evaluation for all saved RAP-SLAM trajectory products.

For every saved trajectory (offline pipeline products and ROS 2 live runs),
compute three metric families against ground truth:

1. UNALIGNED APE (our running convention): first-pose anchoring on both
   trajectories, then evo APE (translation part). Matches summary.json.
2. ALIGNED APE: evo SE(3) Umeyama alignment first (--align convention used by
   most of the SLAM literature's "ATE RMSE" tables). NOT comparable to (1) —
   this is the number to use for horizontal comparison.
3. KITTI-style relative drift: mean relative translational error (%) and
   rotational error (deg/100 m) over subsegments of length 100..800 m
   (all-segment mean; strict distance endpoint; start-step 10 frames,
   matching the official devkit convention). Gauge-invariant, so frame
   anchoring does not matter.

Ground-truth conventions (shared with the pipeline and verify scripts):
- KITTI: load_gt_poses(..., apply_calib=True) on
  <kitti_root>/poses/<seq>.txt with sequences/<seq>/calib.txt.
- POSS: load_gt_poses(..., apply_calib=False); live products use
  segment-relative indices (frame 0 = POSS frame 3850 for segment A).

Outputs: <out>/paper_metrics.json + <out>/paper_metrics.md.

CPU-only. Run inside the conda env (evo required):
    python scripts/eval_paper_metrics.py [--out output_paper_metrics]
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rap_slam import evaluate, real_dataset  # noqa: E402
from rap_slam.trajectory_protocol import kitti_drift, validate_pair, validate_ids, validate_stamps

logger = logging.getLogger("eval_paper_metrics")

KITTI_ROOT = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset"
POSS_ROOT = os.path.join(REPO_ROOT, "data", "poss", "dataset")
POSS_SEGMENTS = {"A": (3850, 5268), "B": (6750, 9500)}

# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_kitti_format(path: str) -> List[np.ndarray]:
    """Load 12-float-per-row pose files (odom_poses.txt / optimized_poses.txt)."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    poses = np.tile(np.eye(4), (len(data), 1, 1))
    poses[:, :3, :] = data.reshape(len(data), 3, 4)
    return list(poses)


def load_tum(path: str, return_timestamps=False) -> List[np.ndarray]:
    """Load TUM rows (stamp tx ty tz qx qy qz qw) — same math as verify scripts."""
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise ValueError("TUM requires finite timestamp + seven pose values")
    stamps = validate_stamps(rows[:, 0])
    poses = []
    for r in rows:
        t = r[1:4]
        qx, qy, qz, qw = r[4:8]
        n_ = qx * qx + qy * qy + qz * qz + qw * qw
        if n_ < 1e-12:
            raise ValueError("Zero TUM quaternion")
        s = 2.0 / n_
        xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
        xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
        wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
        T = np.eye(4)
        T[:3, :3] = [[1 - (yy + zz), xy - wz, xz + wy],
                     [xy + wz, 1 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1 - (xx + yy)]]
        T[:3, 3] = t
        poses.append(T)
    return (stamps, poses) if return_timestamps else poses


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def aligned_ape(gt: List[np.ndarray], est: List[np.ndarray]) -> Dict[str, float]:
    """evo APE (translation) after SE(3) Umeyama alignment."""
    return evaluate.compute_ape_rpe(gt, est, align=True)


def path_length(poses: List[np.ndarray]) -> float:
    t = np.array([p[:3, 3] for p in poses])
    return float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))


def evaluate_pair(gt: List[np.ndarray], est: List[np.ndarray], source_frame_ids=None) -> Dict[str, float]:
    """All metric families for one estimate trajectory."""
    gt, est = validate_pair(gt, est)
    n = len(gt)
    gt_rel = evaluate.align_to_first_pose(gt)
    est_rel = evaluate.align_to_first_pose(est)
    m_unaligned = evaluate.compute_ape_rpe(gt_rel, est_rel)
    m_aligned = aligned_ape(gt_rel, est_rel)
    m_kitti = kitti_drift(gt_rel, est_rel, source_frame_ids=source_frame_ids)
    return {
        "n_frames": n,
        "path_length_m": path_length(gt_rel),
        "ape_unaligned_rmse": m_unaligned["ape_rmse"],
        "ape_unaligned_max": m_unaligned["ape_max"],
        "ape_aligned_rmse": m_aligned["ape_rmse"],
        "ape_aligned_max": m_aligned["ape_max"],
        **m_kitti,
    }


# ---------------------------------------------------------------------------
# product registry
# ---------------------------------------------------------------------------


def gt_from_summary(out_dir: str) -> Optional[List[np.ndarray]]:
    """Derive the GT trajectory from an offline run's summary.json config."""
    summary_file = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_file):
        return None
    cfg = json.load(open(summary_file)).get("config", {})
    root, seq = cfg.get("real_root"), cfg.get("real_seq")
    if not root or not seq:
        return None
    gt = real_dataset.load_gt_poses(
        os.path.join(root, "dataset", "poses", f"{seq}.txt"),
        os.path.join(root, "dataset", "sequences", seq, "calib.txt"),
        start=0, end=None,
        apply_calib=cfg.get("real_apply_calib", False))
    ids, _ = offline_frame_ids(out_dir, cfg, len(gt))
    return list(np.asarray(gt)[ids])


def offline_frame_ids(out_dir, cfg, n_gt):
    path = os.path.join(out_dir, "source_frames.json")
    if os.path.exists(path):
        manifest = json.load(open(path))
        ids = validate_ids(manifest["source_frame_ids"])
        if manifest.get("timestamps") is not None:
            validate_stamps(manifest["timestamps"], len(ids))
        binding = "persisted_source_frame_ids"
    else:
        start, end, stride = cfg.get("real_start", 0), cfg.get("real_end"), cfg.get("real_stride", 1)
        if not isinstance(stride, int) or stride < 1 or start < 0:
            raise ValueError("Invalid legacy frame selection")
        ids = validate_ids(np.arange(start, n_gt if end is None else end + 1, stride))
        binding = "legacy_config_assumed_complete"
    if ids[-1] >= n_gt:
        raise ValueError("Frame ID outside GT table")
    return ids, binding


def gt_for_live(source: str, seq: str, n: int) -> List[np.ndarray]:
    """GT for an explicitly assumed contiguous live prefix (legacy helper)."""
    if n <= 0:
        raise ValueError("Live frame count must be positive")
    if source == "kitti":
        gt = real_dataset.load_gt_poses(
            os.path.join(KITTI_ROOT, "poses", f"{seq}.txt"),
            os.path.join(KITTI_ROOT, "sequences", seq, "calib.txt"),
            start=0, end=n - 1, apply_calib=True)
    elif source == "poss":
        start, end = POSS_SEGMENTS[seq]
        if n > end - start + 1:
            raise ValueError("Requested live prefix exceeds POSS segment")
        gt = real_dataset.load_gt_poses(
            os.path.join(POSS_ROOT, "poses", "00.txt"),
            os.path.join(POSS_ROOT, "sequences", "00", "calib.txt"),
            start=start, end=start + n - 1, apply_calib=False)
    else:
        raise ValueError(f"Unknown live source: {source}")
    if len(gt) != n:
        raise ValueError("Requested live frame range exceeds GT")
    return list(gt)


OFFLINE_DIRS = [
    "output_prods/poss", "output_prods/kitti06", "output_prods/kitti05",
    "output_prods/kitti00_bb",
    "output_deskew/poss", "output_deskew/kitti06", "output_deskew/kitti05",
    "output_deskew/kitti00_bb", "output_deskew/kitti02_off",
    "output_deskew/kitti02",
    "output_possb_smoke", "output_kitti04_smoke",
]

LIVE_RUNS = [  # (name, live_poses, map_poses, source, seq/segment)
    ("step3_poss", "output_ros2/live_poses_step3.txt",
     "output_ros2/step3/map_poses.txt", "poss", "A"),
    # *_current: PGO re-run on the saved live inputs with the CURRENT
    # rap_slam.pose_graph defaults (relfloor+Huber, loop-edge NMS w=10) via
    # scripts/rerun_pgo_saved.py — the saved map_poses.txt freeze older
    # backend code (e.g. step3_kitti02 predates RC-1 and NMS).
    ("step3_poss_current", "output_ros2/live_poses_step3.txt",
     "output_ros2/step3/map_poses_current.txt", "poss", "A"),
    ("step3_kitti02_current", "output_ros2/live_poses_kitti02.txt",
     "output_ros2/step3_kitti02/map_poses_current.txt", "kitti", "02"),
    ("step3_kitti06_current", "output_ros2/live_poses_kitti06.txt",
     "output_ros2/step3_kitti06/map_poses_current.txt", "kitti", "06"),
    ("step3_poss_deskew", "output_ros2/live_poses_step3_deskew.txt",
     "output_ros2/step3_deskew/map_poses.txt", "poss", "A"),
    ("step3_kitti01", "output_ros2/live_poses_kitti01.txt",
     "output_ros2/step3_kitti01/map_poses.txt", "kitti", "01"),
    ("step3_kitti02", "output_ros2/live_poses_kitti02.txt",
     "output_ros2/step3_kitti02/map_poses.txt", "kitti", "02"),
    ("step3_kitti03", "output_ros2/live_poses_kitti03.txt",
     "output_ros2/step3_kitti03/map_poses.txt", "kitti", "03"),
    ("step3_kitti04", "output_ros2/live_poses_kitti04.txt",
     "output_ros2/step3_kitti04/map_poses.txt", "kitti", "04"),
    ("step3_kitti06", "output_ros2/live_poses_kitti06.txt",
     "output_ros2/step3_kitti06/map_poses.txt", "kitti", "06"),
]


def eval_offline(out_dir: str) -> Optional[Dict]:
    odom_file = os.path.join(out_dir, "odom_poses.txt")
    opt_file = os.path.join(out_dir, "optimized_poses.txt")
    if not os.path.exists(odom_file):
        logger.warning(f"skip {out_dir}: no odom_poses.txt")
        return None
    gt = gt_from_summary(out_dir)
    if gt is None:
        logger.warning(f"skip {out_dir}: cannot derive GT from summary.json")
        return None
    cfg = json.load(open(os.path.join(out_dir, "summary.json")))["config"]
    # GT selection has already checked the full table bounds.
    frame_path = os.path.join(out_dir, "source_frames.json")
    if os.path.exists(frame_path):
        ids = validate_ids(json.load(open(frame_path))["source_frame_ids"], len(gt))
        binding = "persisted_source_frame_ids"
    else:
        ids = cfg.get("real_start", 0) + np.arange(len(gt)) * cfg.get("real_stride", 1)
        binding = "legacy_config_assumed_complete"
    result = {"kind": "offline", "dir": out_dir, "frame_binding": binding,
              "source_frame_ids": ids.tolist(), "ape_reference": "lidar"}
    result["odometry"] = evaluate_pair(gt, load_kitti_format(odom_file), ids)
    if os.path.exists(opt_file):
        result["pgo"] = evaluate_pair(gt, load_kitti_format(opt_file), ids)
    if cfg.get("real_apply_calib", False):
        calib = os.path.join(cfg["real_root"], "dataset", "sequences", cfg["real_seq"], "calib.txt")
        tr = real_dataset._read_calib_transform(calib)
        tr_inv = np.linalg.inv(tr)
        cam_gt = [tr @ p @ tr_inv for p in gt]
        for name, path in [("odometry", odom_file), ("pgo", opt_file)]:
            if name in result:
                cam_est = [tr @ p @ tr_inv for p in load_kitti_format(path)]
                result[name]["camera_kitti"] = kitti_drift(cam_gt, cam_est, source_frame_ids=ids)
    return result


def eval_live(name: str, odom_tum: str, map_tum: str,
              source: str, seq: str, allow_legacy_index=False) -> Optional[Dict]:
    if not os.path.exists(odom_tum):
        logger.warning(f"skip {name}: no {odom_tum}")
        return None
    stamps, odom = load_tum(odom_tum, return_timestamps=True)
    manifest_path = odom_tum + ".frames.json"
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        ids = validate_ids(manifest["source_frame_ids"], len(odom))
        expected = validate_stamps(manifest["timestamps"], len(odom))
        if not np.allclose(stamps, expected, rtol=0, atol=1e-6):
            raise ValueError("Live timestamps do not match frame manifest")
        offset = POSS_SEGMENTS[seq][0] if source == "poss" else 0
        if ids[0] < offset:
            raise ValueError("Live source frame precedes selected segment")
        gt = np.asarray(gt_for_live(source, seq, int(ids[-1] - offset + 1)))[ids - offset]
        binding = "persisted_source_frame_ids"
    elif allow_legacy_index:
        gt = gt_for_live(source, seq, len(odom))
        ids = np.arange(len(odom)) + (POSS_SEGMENTS[seq][0] if source == "poss" else 0)
        binding = "legacy_index_assumption_not_verified"
    else:
        raise ValueError("Live source-frame manifest missing; explicit legacy-index opt-in required")
    result = {"kind": "live", "odom_file": odom_tum, "frame_binding": binding,
              "source_frame_ids": ids.tolist()}
    result["odometry"] = evaluate_pair(gt, odom, ids)
    if os.path.exists(map_tum):
        map_stamps, mapped = load_tum(map_tum, return_timestamps=True)
        if len(map_stamps) != len(stamps) or not np.allclose(map_stamps, stamps, rtol=0, atol=1e-6):
            raise ValueError("Odometry and map trajectory timestamps differ")
        result["pgo"] = evaluate_pair(gt, mapped, ids)
    if source == "kitti":
        tr = real_dataset._read_calib_transform(os.path.join(KITTI_ROOT, "sequences", seq, "calib.txt"))
        tr_inv = np.linalg.inv(tr)
        cam_gt = [tr @ p @ tr_inv for p in gt]
        for key, poses in [("odometry", odom), ("pgo", mapped if "pgo" in result else None)]:
            if poses is not None:
                result[key]["camera_kitti"] = kitti_drift(cam_gt, [tr @ p @ tr_inv for p in poses], source_frame_ids=ids)
    return result


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

FIELDS = [("n_frames", "{:d}"), ("path_length_m", "{:.0f}"),
          ("ape_unaligned_rmse", "{:.3f}"), ("ape_aligned_rmse", "{:.3f}"),
          ("kitti_ter_pct", "{:.3f}"), ("kitti_rer_deg100m", "{:.4f}")]


def md_table(results: Dict[str, Dict], title: str) -> str:
    lines = [f"### {title}", "",
             "| dataset | traj | n | path [m] | APE unaligned [m] "
             "| **APE aligned [m]** | LiDAR ter [%] | LiDAR rer [deg/100m] |",
             "|---|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        for traj in ("odometry", "pgo"):
            if traj not in res:
                continue
            m = res[traj]
            row = [fmt.format(m[k]) for k, fmt in FIELDS]
            lines.append(f"| {name} | {traj} | " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output_paper_metrics")
    parser.add_argument("--allow-legacy-live-index", action="store_true")
    parser.add_argument("--offline-only", action="store_true", help="Evaluate the offline core without legacy live products")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    offline, live = {}, {}
    for d in OFFLINE_DIRS:
        name = d.replace("output_", "").replace("/", "_")
        logger.info(f"== offline {name}")
        r = eval_offline(d)
        if r:
            offline[name] = r
            logger.info(f"   odom aligned APE "
                        f"{r['odometry']['ape_aligned_rmse']:.3f} m, "
                        f"ter {r['odometry']['kitti_ter_pct']:.3f} %")
            if "pgo" in r:
                logger.info(f"   PGO  aligned APE "
                            f"{r['pgo']['ape_aligned_rmse']:.3f} m, "
                            f"ter {r['pgo']['kitti_ter_pct']:.3f} %")
    for name, odom_tum, map_tum, source, seq in ([] if args.offline_only else LIVE_RUNS):
        logger.info(f"== live {name}")
        r = eval_live(name, odom_tum, map_tum, source, seq, args.allow_legacy_live_index)
        if r:
            live[name] = r

    os.makedirs(args.out, exist_ok=True)
    all_results = {"offline": offline, "live": live,
                   "notes": {
                       "aligned_ape": "evo APE translation after SE(3) "
                                      "Umeyama alignment (literature "
                                      "'ATE RMSE' convention)",
                       "unaligned_ape": "first-pose anchoring only (our "
                                        "running convention, matches "
                                        "summary.json)",
                       "kitti": "KITTI-style all-segment relative drift, 100-800 m "
                                "subsegments, start step 10 frames",
                       "gt": "KITTI apply_calib=True; POSS "
                             "apply_calib=False, segA offset 3850"}}
    json_path = os.path.join(args.out, "paper_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    md = ["# RAP-SLAM paper metrics draft", "",
          "Aligned APE = evo `--align` convention (comparable to literature "
          "ATE RMSE). Unaligned APE = our running convention (first-pose "
          "anchoring). KITTI ter/rer = KITTI-style all-segment relative drift "
          "(100–800 m subsegments).", ""]
    md.append(md_table(offline, "Offline pipeline products"))
    md.append(md_table(live, "ROS 2 live products"))
    md_path = os.path.join(args.out, "paper_metrics.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    logger.info(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
