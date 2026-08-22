"""Paper-metrics evaluation for all saved RAP-SLAM trajectory products.

For every saved trajectory (offline pipeline products and ROS 2 live runs),
compute three metric families against ground truth:

1. UNALIGNED APE (our running convention): first-pose anchoring on both
   trajectories, then evo APE (translation part). Matches summary.json.
2. ALIGNED APE: evo SE(3) Umeyama alignment first (--align convention used by
   most of the SLAM literature's "ATE RMSE" tables). NOT comparable to (1) —
   this is the number to use for horizontal comparison.
3. KITTI official relative drift: mean relative translational error (%) and
   rotational error (deg/100 m) over subsegments of length 100..800 m
   (per-length mean, then averaged over the 8 lengths; start-step 10 frames,
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


def load_tum(path: str) -> List[np.ndarray]:
    """Load TUM rows (stamp tx ty tz qx qy qz qw) — same math as verify scripts."""
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None, :]
    poses = []
    for r in rows:
        t = r[1:4]
        qx, qy, qz, qw = r[4:8]
        n_ = qx * qx + qy * qy + qz * qz + qw * qw
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
    return poses


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def aligned_ape(gt: List[np.ndarray], est: List[np.ndarray]) -> Dict[str, float]:
    """evo APE (translation) after SE(3) Umeyama alignment."""
    return evaluate.compute_ape_rpe(gt, est, align=True)


def path_length(poses: List[np.ndarray]) -> float:
    t = np.array([p[:3, 3] for p in poses])
    return float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))


def kitti_drift(gt: List[np.ndarray], est: List[np.ndarray],
                step: int = 10) -> Dict[str, float]:
    """KITTI official relative drift over 100..800 m subsegments.

    Per subsegment: t_err = |t_err| / L (fraction), r_err = angle / L
    converted to deg/100 m. Per-length mean, then averaged over lengths.
    Start frames advance by `step` (devkit convention).
    """
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    gt_t = [p[:3, 3] for p in gt]
    # cumulative GT path distance
    cum = [0.0]
    for i in range(1, len(gt_t)):
        cum.append(cum[-1] + float(np.linalg.norm(gt_t[i] - gt_t[i - 1])))
    cum = np.asarray(cum)

    t_means, r_means = [], []
    n_frames = len(gt)
    for L in lengths:
        t_errs, r_errs = [], []
        for first in range(0, n_frames, step):
            # first frame where the GT path covers L metres
            last = int(np.searchsorted(cum, cum[first] + L))
            if last >= n_frames:
                continue
            d_gt = np.linalg.inv(gt[first]) @ gt[last]
            d_est = np.linalg.inv(est[first]) @ est[last]
            err = np.linalg.inv(d_gt) @ d_est
            t_errs.append(float(np.linalg.norm(err[:3, 3])) / L)
            cos = np.clip((np.trace(err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            r_errs.append(float(np.arccos(cos)) / L * 180.0 / np.pi * 100.0)
        if t_errs:
            t_means.append(float(np.mean(t_errs)))
            r_means.append(float(np.mean(r_errs)))
    if not t_means:
        return {"kitti_ter_pct": float("nan"), "kitti_rer_deg100m": float("nan"),
                "kitti_segments": 0}
    return {"kitti_ter_pct": 100.0 * float(np.mean(t_means)),
            "kitti_rer_deg100m": float(np.mean(r_means)),
            "kitti_segments": len(t_means)}


def evaluate_pair(gt: List[np.ndarray], est: List[np.ndarray]) -> Dict[str, float]:
    """All metric families for one estimate trajectory."""
    n = min(len(gt), len(est))
    gt, est = gt[:n], est[:n]
    gt_rel = evaluate.align_to_first_pose(gt)
    est_rel = evaluate.align_to_first_pose(est)
    m_unaligned = evaluate.compute_ape_rpe(gt_rel, est_rel)
    m_aligned = aligned_ape(gt_rel, est_rel)
    m_kitti = kitti_drift(gt_rel, est_rel)
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
        start=cfg.get("real_start", 0), end=cfg.get("real_end"),
        apply_calib=cfg.get("real_apply_calib", False))
    return list(gt)


def gt_for_live(source: str, seq: str, n: int) -> List[np.ndarray]:
    """GT for a live run: KITTI seq or a POSS segment, truncated to n frames."""
    if source == "kitti":
        gt = real_dataset.load_gt_poses(
            os.path.join(KITTI_ROOT, "poses", f"{seq}.txt"),
            os.path.join(KITTI_ROOT, "sequences", seq, "calib.txt"),
            start=0, end=n - 1, apply_calib=True)
    else:  # poss
        start, _ = POSS_SEGMENTS[seq]
        gt = real_dataset.load_gt_poses(
            os.path.join(POSS_ROOT, "poses", "00.txt"),
            os.path.join(POSS_ROOT, "sequences", "00", "calib.txt"),
            start=start, end=start + n - 1, apply_calib=False)
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
    result = {"kind": "offline", "dir": out_dir}
    result["odometry"] = evaluate_pair(gt, load_kitti_format(odom_file))
    if os.path.exists(opt_file):
        result["pgo"] = evaluate_pair(gt, load_kitti_format(opt_file))
    return result


def eval_live(name: str, odom_tum: str, map_tum: str,
              source: str, seq: str) -> Optional[Dict]:
    if not os.path.exists(odom_tum):
        logger.warning(f"skip {name}: no {odom_tum}")
        return None
    odom = load_tum(odom_tum)
    gt = gt_for_live(source, seq, len(odom))
    result = {"kind": "live", "odom_file": odom_tum}
    result["odometry"] = evaluate_pair(gt, odom)
    if os.path.exists(map_tum):
        result["pgo"] = evaluate_pair(gt, load_tum(map_tum))
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
             "| **APE aligned [m]** | KITTI ter [%] | rer [deg/100m] |",
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
    args = parser.parse_args()
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
    for name, odom_tum, map_tum, source, seq in LIVE_RUNS:
        logger.info(f"== live {name}")
        r = eval_live(name, odom_tum, map_tum, source, seq)
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
                       "kitti": "official relative drift, 100-800 m "
                                "subsegments, start step 10 frames",
                       "gt": "KITTI apply_calib=True; POSS "
                             "apply_calib=False, segA offset 3850"}}
    json_path = os.path.join(args.out, "paper_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    md = ["# RAP-SLAM paper metrics draft", "",
          "Aligned APE = evo `--align` convention (comparable to literature "
          "ATE RMSE). Unaligned APE = our running convention (first-pose "
          "anchoring). KITTI ter/rer = official relative drift "
          "(100–800 m subsegments).", ""]
    md.append(md_table(offline, "Offline pipeline products"))
    md.append(md_table(live, "ROS 2 live products"))
    md_path = os.path.join(args.out, "paper_metrics.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    logger.info(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
