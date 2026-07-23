#!/usr/bin/env python
"""Measure acceptance-gate discriminators on KITTI 02 true vs false loop pairs.

Registers, with the RAPBackend:
  - ~8 TRUE loop pairs sampled from the GT revisit clusters of KITTI 02
    (scans ~930-1030 <-> ~4200-4270 and ~1800-1820 <-> ~4400-4410,
    separation >= 200, GT distance < 5 m), and
  - the 11 FALSE pairs accepted by the big-budget live run
    (output_ros2/step3_kitti02_bigbudget/accepted_loops.json),
and reports, per pair: overlap ratio, mean symmetric NN distance after
alignment (mean_dist), and the T_rel error vs GT relative pose.

Run in the conda env:
  /root/miniconda3/envs/rap-inference/bin/python scripts/eval_acceptance_gate.py
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rap_slam import real_dataset  # noqa: E402

KITTI_SEQ_DIR = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset/sequences/02"
POSES_FILE = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset/poses/02.txt"
FALSE_LOOPS_JSON = os.path.join(
    REPO_ROOT, "output_ros2/step3_kitti02_bigbudget/accepted_loops.json")

# True pairs from the GT revisit clusters (stride-10 GT analysis).
TRUE_PAIRS = [
    (930, 4200), (950, 4220), (980, 4240), (990, 4250),
    (1010, 4260), (1800, 4400), (1810, 4400), (1820, 4410),
]


def load_scan(seq_dir: str, idx: int) -> np.ndarray:
    path = os.path.join(seq_dir, "velodyne", f"{idx:06d}.bin")
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def rotation_error_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def mean_nn_dist(ds_a: np.ndarray, ds_b: np.ndarray, T_rel: np.ndarray) -> float:
    """Mean symmetric nearest-neighbor distance after alignment (meters).

    Same distance field as RAPBackend._overlap_ratio, reduced by mean
    instead of the <1 m fraction.
    """
    from scipy.spatial import cKDTree
    b_in_a = (T_rel[:3, :3] @ ds_b.T).T + T_rel[:3, 3]
    tree_a = cKDTree(ds_a)
    tree_b = cKDTree(b_in_a)
    d_ba, _ = tree_a.query(b_in_a, k=1)
    d_ab, _ = tree_b.query(ds_a, k=1)
    return 0.5 * (float(np.mean(d_ba)) + float(np.mean(d_ab)))


def main():
    from rap_slam.rap_backend import RAPBackend

    with open(FALSE_LOOPS_JSON) as f:
        false_pairs = [(lc["i"], lc["j"]) for lc in json.load(f)]
    print(f"false pairs from {FALSE_LOOPS_JSON}: {len(false_pairs)}")

    gt = list(real_dataset.load_gt_poses(
        POSES_FILE, os.path.join(KITTI_SEQ_DIR, "calib.txt"),
        start=0, end=4660, apply_calib=True))

    rap = RAPBackend()  # defaults: rap_12 ckpt, voxel 0.4, 4000 pts, spinnet

    needed = sorted({s for pair in TRUE_PAIRS + false_pairs for s in pair})
    scans = {s: load_scan(KITTI_SEQ_DIR, s) for s in needed}
    print(f"loaded {len(scans)} scans; registering "
          f"{len(TRUE_PAIRS)} true + {len(false_pairs)} false pairs\n")

    print(f"{'pair':>13} {'kind':>5} {'overlap':>7} {'mean_dist':>9} "
          f"{'GTpos':>7} {'dT_err':>7} {'dR_err':>7}")
    rows = []
    for kind, pairs in (("TRUE", TRUE_PAIRS), ("FALSE", false_pairs)):
        for i, j in pairs:
            T_rel, diag = rap.register(scans[i], scans[j],
                                       return_registered=True)
            # Metrics on the same downsampled clouds the backend aligned.
            md = mean_nn_dist(diag["ds_a"], diag["ds_b"], T_rel)
            overlap = diag["overlap_ratio"]
            T_gt = np.linalg.inv(gt[i]) @ gt[j]
            dG = np.linalg.inv(T_gt) @ T_rel
            te = float(np.linalg.norm(dG[:3, 3]))
            re = rotation_error_deg(dG[:3, :3])
            gt_pos = float(np.linalg.norm(gt[i][:3, 3] - gt[j][:3, 3]))
            rows.append((i, j, kind, overlap, md, gt_pos, te, re))
            print(f"({i:5d},{j:5d}) {kind:>5} {overlap:7.3f} {md:9.3f} "
                  f"{gt_pos:7.1f} {te:7.2f} {re:7.2f}")

    def col(k, idx):
        return [r[idx] for r in rows if r[2] == k]

    print("\n--- summary ---")
    for k in ("TRUE", "FALSE"):
        ov, md, te = col(k, 3), col(k, 4), col(k, 6)
        print(f"{k:>5}: overlap {min(ov):.3f}..{max(ov):.3f} | "
              f"mean_dist {min(md):.3f}..{max(md):.3f} | "
              f"T_rel trans err {min(te):.2f}..{max(te):.2f} m")
    tov, fov = col("TRUE", 3), col("FALSE", 3)
    tmd, fmd = col("TRUE", 4), col("FALSE", 4)
    print(f"\noverlap gap: min(TRUE) {min(tov):.3f} vs max(FALSE) {max(fov):.3f}")
    print(f"mean_dist gap: max(TRUE) {max(tmd):.3f} vs min(FALSE) {min(fmd):.3f}")


if __name__ == "__main__":
    main()
