"""[meeting-20260921] Evaluate meeting-arm run directories with the frozen
paper-metric code path (aligned ATE RMSE via evo SE(3) Umeyama + KITTI
official 100-800 m ter/rer), reusing scripts/eval_paper_metrics.py.

Usage (WSL, rap-inference env, from RAP-SLAM root):
    python scripts/eval_meeting_arms.py --out /mnt/e/githouse/RAP-main/docs/meeting_20260921/meeting_metrics.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_paper_metrics as epm  # noqa: E402


def load_poses(path):
    return epm.load_kitti_format(path)


def eval_dir(out_dir):
    gt = epm.gt_from_summary(out_dir)
    if gt is None:
        raise ValueError(f"no GT derivable for {out_dir}")
    result = {"dir": out_dir}
    summary = json.load(open(os.path.join(out_dir, "summary.json")))
    cfg = summary.get("config", {})
    result["config_interest"] = {
        "apply_kitti_205": cfg.get("apply_kitti_205"),
        "odom_max_range": cfg.get("odom_max_range"),
        "odom_min_range": cfg.get("odom_min_range"),
        "odom_voxel": cfg.get("odom_voxel"),
        "max_candidates": cfg.get("max_candidates"),
        "loop_sep": cfg.get("loop_sep"),
    }
    result["n_loops_accepted"] = summary.get("n_loops_accepted")
    result["n_loop_candidates"] = summary.get("n_loop_candidates")
    for tag, fname in [("odometry", "odom_poses.txt"),
                       ("full", "optimized_poses.txt")]:
        est = load_poses(os.path.join(out_dir, fname))
        result[tag] = epm.evaluate_pair(gt, est)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="*")
    args = ap.parse_args()
    dirs = args.dirs or [
        "output_meeting/kitti06_armB",
        "output_meeting/kitti05_armB",
        "output_meeting/kitti00_armB",
        "output_prods/kitti06_205",
        "output_prods/kitti05_205",
        "output_prods/kitti00_205",
        "output_prods/kitti02_205",
    ]
    results = {}
    for d in dirs:
        if not os.path.exists(os.path.join(d, "optimized_poses.txt")):
            print(f"skip {d} (no optimized_poses.txt)")
            continue
        print(f"evaluating {d} ...", flush=True)
        results[d] = eval_dir(d)
        r = results[d]
        print(f"  full: aligned ATE {r['full']['ape_aligned_rmse']:.4f} m, "
              f"ter {r['full']['kitti_ter_pct']:.3f} %, "
              f"rer {r['full']['kitti_rer_deg100m']:.4f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
