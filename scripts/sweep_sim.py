#!/usr/bin/env python3
"""Sweep simulation parameters to find a config with realistic odometry drift."""
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rap_slam import evaluate, odometry, sequence

logging.basicConfig(level=logging.WARNING)

MAP = "/mnt/e/githouse/RAP-main/demo_example_data/map_kitti/kitti00.ply"


def run_config(n_poses, points_range, noise, sensor_range, seed=42):
    map_points, tree = sequence.load_map(MAP)
    gt_poses = sequence.generate_loop_poses(map_points, tree, center=(250.0, -30.0),
                                            radius=80.0, n_poses=n_poses)
    rng = np.random.default_rng(seed)
    scans = [sequence.simulate_scan(map_points, tree, p, sensor_range=sensor_range,
                                    n_target=int(rng.integers(points_range[0], points_range[1] + 1)),
                                    noise_sigma=noise, rng=rng)
             for p in gt_poses]
    t0 = time.time()
    poses, backend = odometry.run_odometry(scans)
    dt = time.time() - t0
    gt_rel = evaluate.align_to_first_pose(list(gt_poses))
    od_rel = evaluate.align_to_first_pose(poses)
    m = evaluate.compute_ape_rpe(gt_rel, od_rel)
    # Loop closure error: pose[N-1] should equal pose[0] == identity in aligned frame.
    closure = np.linalg.norm(od_rel[-1][:3, 3])
    print(f"n={n_poses:3d} pts={points_range} sigma={noise:.2f} range={sensor_range:.0f} "
          f"| {backend} | APE rmse={m['ape_rmse']:.3f} max={m['ape_max']:.3f} "
          f"RPE rmse={m['rpe_rmse']:.4f} | closure err={closure:.3f} m | {dt:.1f}s",
          flush=True)


if __name__ == "__main__":
    configs = [
        (100, (3000, 8000), 0.05, 50.0),
        (60, (2000, 4000), 0.08, 50.0),
        (60, (1500, 3000), 0.10, 45.0),
        (40, (1500, 3000), 0.10, 40.0),
        (60, (1500, 3000), 0.15, 40.0),
        (80, (1000, 2500), 0.10, 40.0),
    ]
    for c in configs:
        run_config(*c)
