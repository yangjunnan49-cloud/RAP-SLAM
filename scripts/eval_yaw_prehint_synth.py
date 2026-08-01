#!/usr/bin/env python
"""RC-2 addendum: synthetic large-yaw test for SC yaw pre-rotation.

KITTI 02's true revisit pairs all have small GT yaw (<= 22 deg), so the
large-yaw true-loop regime cannot be tested on real pairs. This script
takes real true pairs and synthetically rotates scan j by phi in {90, 180}
deg, making the pair a large-yaw registration problem with known GT:
  T_gt' = T_gt @ Rz(-phi)   (since p_j' = Rz(phi) p_j)
and scores vanilla vs SC-prehint registration exactly like
eval_yaw_prehint.py. Decisive either way: if vanilla RAP still nails the
180 deg pairs, its global registration needs no yaw hint; if it fails and
prehint succeeds, the hint is worth wiring into the live pipeline.

Run in the conda env:
  /root/miniconda3/envs/rap-inference/bin/python scripts/eval_yaw_prehint_synth.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rap_slam import real_dataset  # noqa: E402
from rap_slam.scan_context import make_scan_context, sc_distance  # noqa: E402

KITTI_SEQ_DIR = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset/sequences/02"
POSES_FILE = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset/poses/02.txt"

PAIRS = [(930, 4200), (1010, 4260), (1800, 4400), (1820, 4410)]
PHIS = [90.0, 180.0]
OVERLAP_MIN, MEAN_DIST_MAX = 0.7, 2.5


def load_scan(idx: int) -> np.ndarray:
    path = os.path.join(KITTI_SEQ_DIR, "velodyne", f"{idx:06d}.bin")
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def rotz3(psi_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(psi_deg)), np.sin(np.radians(psi_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_error_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    from rap_slam.rap_backend import RAPBackend
    import torch

    gt = list(real_dataset.load_gt_poses(
        POSES_FILE, os.path.join(KITTI_SEQ_DIR, "calib.txt"),
        start=0, end=4660, apply_calib=True))
    rap = RAPBackend()

    needed = sorted({s for p in PAIRS for s in p})
    scans = {s: load_scan(s) for s in needed}

    def reseed():
        torch.manual_seed(42)
        np.random.seed(42)

    print(f"{'pair':>13} {'phi':>4} {'hint':>6} {'herr':>6} | "
          f"{'ov0':>5} {'dT0':>5} {'dR0':>5} {'g0':>2} | "
          f"{'ov1':>5} {'dT1':>5} {'dR1':>5} {'g1':>2}")
    for i, j in PAIRS:
        T_gt = np.linalg.inv(gt[i]) @ gt[j]
        sc_a = make_scan_context(scans[i], max_radius=80.0)
        for phi in PHIS:
            b = (rotz3(phi) @ scans[j].T).T
            T_gtp = T_gt @ np.vstack([np.hstack([rotz3(-phi), np.zeros((3, 1))]),
                                      [0, 0, 0, 1]])
            y_gtp = float(np.degrees(np.arctan2(T_gtp[1, 0], T_gtp[0, 0])))
            d_sc, shift = sc_distance(sc_a, make_scan_context(b, max_radius=80.0))
            psi = shift * (360.0 / sc_a.shape[1])
            herr = (psi - y_gtp + 180.0) % 360.0 - 180.0

            reseed()
            T0, dg0 = rap.register(scans[i], b)
            dG0 = np.linalg.inv(T_gtp) @ T0
            te0, re0 = float(np.linalg.norm(dG0[:3, 3])), rotation_error_deg(dG0[:3, :3])
            g0 = dg0["overlap_ratio"] >= OVERLAP_MIN and dg0["mean_dist"] <= MEAN_DIST_MAX

            Rz = np.eye(4)
            Rz[:3, :3] = rotz3(psi)
            reseed()
            T1p, dg1 = rap.register(scans[i], (Rz[:3, :3] @ b.T).T)
            dG1 = np.linalg.inv(T_gtp) @ (T1p @ Rz)
            te1, re1 = float(np.linalg.norm(dG1[:3, 3])), rotation_error_deg(dG1[:3, :3])
            g1 = dg1["overlap_ratio"] >= OVERLAP_MIN and dg1["mean_dist"] <= MEAN_DIST_MAX

            print(f"({i:5d},{j:5d}) {phi:4.0f} {psi:6.1f} {herr:6.1f} | "
                  f"{dg0['overlap_ratio']:5.2f} {te0:5.2f} {re0:5.1f} "
                  f"{'Y' if g0 else 'n':>2} | {dg1['overlap_ratio']:5.2f} "
                  f"{te1:5.2f} {re1:5.1f} {'Y' if g1 else 'n':>2}")


if __name__ == "__main__":
    main()
