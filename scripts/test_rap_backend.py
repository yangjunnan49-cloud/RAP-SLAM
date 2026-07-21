#!/usr/bin/env python3
"""Standalone validation of the in-process RAP backend against known GT."""
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rap_slam import sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")

MAP = "/mnt/e/githouse/RAP-main/demo_example_data/map_kitti/kitti00.ply"


def rot_err_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    t0 = time.time()
    map_points, tree = sequence.load_map(MAP)
    gt_poses = sequence.generate_loop_poses(map_points, tree, center=(250.0, -30.0),
                                            radius=80.0, n_poses=100)
    rng = np.random.default_rng(0)
    # Simulate 6 scans (poses 0..5) ~5 m apart.
    scans = [sequence.simulate_scan(map_points, tree, gt_poses[k], n_target=5000,
                                    rng=rng) for k in range(6)]
    print(f"scans generated in {time.time()-t0:.1f}s; sizes: {[len(s) for s in scans]}")

    from rap_slam.rap_backend import RAPBackend
    rap = RAPBackend(sampling_steps=5, max_points_per_scan=4000, voxel_size=0.4)

    pairs = [(0, 1), (0, 2), (1, 3), (2, 5)]
    for i, j in pairs:
        T_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
        t0 = time.time()
        T_rel, diag = rap.register(scans[i], scans[j])
        dt = time.time() - t0
        dT = np.linalg.inv(T_gt) @ T_rel
        te = float(np.linalg.norm(dT[:3, 3]))
        re = rot_err_deg(dT[:3, :3])
        print(f"pair ({i},{j}): GT dist={np.linalg.norm(T_gt[:3,3]):.1f} m | "
              f"RAP error dT={te:.2f} m dR={re:.2f} deg | overlap={diag['overlap_ratio']:.2f} | "
              f"n=({diag['n_a']},{diag['n_b']}) scale={diag['scale']:.1f} anchor={diag['anchor_idx']} | "
              f"prep={diag['t_preprocess']:.1f}s infer={diag['t_inference']:.1f}s tot={dt:.1f}s")


if __name__ == "__main__":
    main()
