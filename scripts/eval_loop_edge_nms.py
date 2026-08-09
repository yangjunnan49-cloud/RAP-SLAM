#!/usr/bin/env python
"""Loop-edge NMS before PGO (KITTI-06 dense-edge finding, 2026-08-08).

KITTI-06 live: cluster-persistence retrieval accepted 114 near-duplicate
loop edges at the single revisit site; PGO treats them as independent and
over-weights the site's noise -> live PGO APE 3.221 m, WORSE than odometry
2.777 m. This script checks offline (GPU-free, saved live products) whether
greedy (i,j)-space NMS on accepted loop edges fixes KITTI-06 without
regressing KITTI-02 (26.423 m) or POSS (1.757 m).

NMS: sort edges by overlap desc; greedily select; suppress a candidate when
an already-selected edge lies within Chebyshev distance w in BOTH indices
(|i-i'| <= w AND |j-j'| <= w). Arms: w in {0, 5, 10, 25, 50} + the
all-edges baseline (w=0 removes only exact (i,j) duplicates). Loop noise
model: live defaults relfloor+huber (RC-1), reusing eval_pgo_robust's PGO.

Run (WSL2, conda env rap-inference):
  python scripts/eval_loop_edge_nms.py
"""

import json
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # eval_pgo_robust

import eval_pgo_robust as epr  # noqa: E402
from rap_slam import evaluate, real_dataset  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "output_ros2")
KITTI_ROOT = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset"

DATASETS = {
    "kitti06": dict(
        odom=os.path.join(OUT_DIR, "live_poses_kitti06.txt"),
        save_dir=os.path.join(OUT_DIR, "step3_kitti06"),
        kind="kitti", seq="06"),
    "kitti02": dict(
        odom=os.path.join(OUT_DIR, "live_poses_kitti02.txt"),
        save_dir=os.path.join(OUT_DIR, "step3_kitti02"),
        kind="kitti", seq="02"),
    "poss": dict(
        odom=os.path.join(OUT_DIR, "live_poses_step3.txt"),
        save_dir=os.path.join(OUT_DIR, "step3"),
        kind="poss", seq="00"),
}

NMS_WIDTHS = (0, 5, 10, 25, 50)


def nms_edges(loops, w):
    """Greedy (i,j)-space NMS, highest overlap first: suppress a candidate
    when a selected edge lies within w in BOTH scan indices."""
    kept = []
    for lc in sorted(loops, key=lambda e: -e["overlap"]):
        if all(abs(lc["i"] - k["i"]) > w or abs(lc["j"] - k["j"]) > w
               for k in kept):
            kept.append(lc)
    return kept


def load_gt(cfg, n):
    if cfg["kind"] == "poss":
        poss_root = os.path.join(REPO_ROOT, "data", "poss", "dataset")
        gt = real_dataset.load_gt_poses(
            os.path.join(poss_root, "poses", "00.txt"),
            None, start=3850, end=3850 + n - 1, apply_calib=False)
    else:
        gt = real_dataset.load_gt_poses(
            os.path.join(KITTI_ROOT, f"poses/{cfg['seq']}.txt"),
            os.path.join(KITTI_ROOT, f"sequences/{cfg['seq']}/calib.txt"),
            start=0, end=n - 1, apply_calib=True)
    return evaluate.align_to_first_pose(list(gt))


def main():
    # Live default loop-edge noise model (RC-1): relfloor ref=1.0 cap=3.0
    # + Huber(k=1).
    loop_noise = epr.make_noise_models()["relfloor_r1.0c3.0+huber"]
    summary = {}
    for name, cfg in DATASETS.items():
        odom_poses = epr.load_tum(cfg["odom"])
        keyframes = json.load(
            open(os.path.join(cfg["save_dir"], "keyframes.json")))
        loops = json.load(
            open(os.path.join(cfg["save_dir"], "accepted_loops.json")))
        n = len(odom_poses)
        gt_rel = load_gt(cfg, n)
        m_odom = evaluate.compute_ape_rpe(
            gt_rel[:n], evaluate.align_to_first_pose(odom_poses))
        print(f"\n=== {name}: {n} frames, {len(keyframes)} keyframes, "
              f"{len(loops)} loop edges | odom APE "
              f"{m_odom['ape_rmse']:.3f} m ===", flush=True)

        # Redundancy diagnostic: highest endpoint degrees (i or j shared).
        cnt = Counter(e["i"] for e in loops) + Counter(e["j"] for e in loops)
        print(f"  endpoint degree top-3: {cnt.most_common(3)}")

        res = {"odom_ape": m_odom["ape_rmse"], "arms": {}}
        for w in (None,) + NMS_WIDTHS:
            edges = loops if w is None else nms_edges(loops, w)
            optimized, n_edges = epr.run_pgo(odom_poses, keyframes, edges,
                                             loop_noise)
            m = epr.full_traj_ape(odom_poses, keyframes, optimized, gt_rel)
            tag = "all" if w is None else f"w={w}"
            res["arms"][tag] = {"edges": n_edges,
                                "ape_rmse": m["ape_rmse"]}
            print(f"  {tag:>6s}: edges {n_edges:4d} -> PGO APE "
                  f"{m['ape_rmse']:7.3f} m", flush=True)
        summary[name] = res

    out = os.path.join(OUT_DIR, "eval_loop_edge_nms.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nsummary saved to {out}")


if __name__ == "__main__":
    main()
