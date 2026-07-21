import logging, sys, time
import numpy as np
sys.path.insert(0, "/mnt/e/githouse/RAP-SLAM")
logging.basicConfig(level=logging.WARNING)
from rap_slam import evaluate, odometry, sequence
MAP = "/mnt/e/githouse/RAP-main/demo_example_data/map_kitti/kitti00.ply"
map_points, tree = sequence.load_map(MAP)

def run(n_poses, pts_lo, pts_hi, noise, rng_seed=42):
    gt_poses = sequence.generate_loop_poses(map_points, tree, center=(250.0, -30.0), radius=80.0, n_poses=n_poses)
    rng = np.random.default_rng(rng_seed)
    scans = [sequence.simulate_scan(map_points, tree, p, sensor_range=50.0,
                                    n_target=int(rng.integers(pts_lo, pts_hi + 1)),
                                    noise_sigma=noise, rng=rng) for p in gt_poses]
    t0 = time.time()
    poses, backend = odometry.run_odometry(scans)
    dt = time.time() - t0
    gt_rel = evaluate.align_to_first_pose(list(gt_poses)); od_rel = evaluate.align_to_first_pose(poses)
    m = evaluate.compute_ape_rpe(gt_rel, od_rel)
    e01 = np.linalg.inv(np.linalg.inv(gt_rel[0]) @ gt_rel[1]) @ (np.linalg.inv(od_rel[0]) @ od_rel[1])
    closure = np.linalg.norm(od_rel[-1][:3, 3])
    print(f"n={n_poses} pts=({pts_lo},{pts_hi}) sigma={noise}: APE rmse={m['ape_rmse']:.3f} max={m['ape_max']:.3f} "
          f"RPE rmse={m['rpe_rmse']:.4f} f01={np.linalg.norm(e01[:3,3]):.3f} closure={closure:.3f} | {dt:.1f}s", flush=True)

for n in [120, 150, 170]:
    run(n, 2000, 4000, 0.05)
run(150, 1500, 3000, 0.05)
run(150, 2000, 4000, 0.08)
