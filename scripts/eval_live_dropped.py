#!/usr/bin/env python3
"""Timestamp-aware evaluation of a live KITTI run with dropped frames.

The odometry node runs with queue_depth=1 (always newest scan), so under
CPU contention it drops frames. The naive eval then compares live pose k
against GT pose k, which shifts correspondence after every drop and makes
APE/RPE meaningless (26 drops on KITTI 02 inflated "APE" to 60 m).

The player stamps each cloud at publish time (12 Hz grid), so the true
frame index of every saved pose is recoverable from its timestamp:
frame = round((t - t0) * rate). This script re-evaluates map_poses.txt
(PGO-corrected) and the odometry TUM file against GT with that mapping.

  python scripts/eval_live_dropped.py --seq 02 \
      --map-poses output_ros2/step3_kitti02/map_poses.txt \
      --odom-tum output_ros2/live_poses_kitti02.txt
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rap_slam import evaluate, real_dataset  # noqa: E402


def load_tum(path):
    rows = np.loadtxt(path)
    stamps, poses = [], []
    for r in np.atleast_2d(rows):
        t, x, y, z, qx, qy, qz, qw = r
        n_ = qx*qx + qy*qy + qz*qz + qw*qw
        s = 2.0 / n_
        xx, yy, zz = qx*qx*s, qy*qy*s, qz*qz*s
        xy, xz, yz = qx*qy*s, qx*qz*s, qy*qz*s
        wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s
        T = np.eye(4)
        T[:3, :3] = [[1-(yy+zz), xy-wz, xz+wy],
                     [xy+wz, 1-(xx+zz), yz-wx],
                     [xz-wy, yz+wx, 1-(xx+yy)]]
        T[:3, 3] = [x, y, z]
        stamps.append(t)
        poses.append(T)
    return np.array(stamps), poses


def frame_indices(stamps, rate, n_frames):
    """Recover player frame indices from publish timestamps.

    The player's effective rate can be below the requested one (dense
    clouds), so estimate the grid period end-to-end: the player publishes
    frames 0..n_frames-1 on a roughly uniform grid and the last saved
    pose is (almost always) the last frame. Per-interval rounding then
    keeps dropped-frame gaps exact."""
    period = (stamps[-1] - stamps[0]) / (n_frames - 1)
    steps = np.rint(np.diff(stamps) / period).astype(int)  # >=1 per interval
    idx = np.concatenate([[0], np.cumsum(steps)])
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", required=True)
    p.add_argument("--map-poses", required=True)
    p.add_argument("--odom-tum", required=True)
    p.add_argument("--rate", type=float, default=12.0)
    p.add_argument("--n-frames", type=int, default=None,
                   help="frames played (default: GT pose count)")
    p.add_argument("--kitti-root",
                   default="/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset")
    p.add_argument("--loops-json", default=None)
    p.add_argument("--keyframes-json", default=None)
    args = p.parse_args()

    gt = list(real_dataset.load_gt_poses(
        os.path.join(args.kitti_root, f"poses/{args.seq}.txt"),
        os.path.join(args.kitti_root, f"sequences/{args.seq}/calib.txt"),
        start=0, end=None, apply_calib=True))

    om_stamps, om_poses = load_tum(args.odom_tum)
    n_frames = args.n_frames or len(gt)
    om_idx = frame_indices(om_stamps, args.rate, n_frames)
    print(f"odom poses: {len(om_poses)}, recovered frame range "
          f"{om_idx[0]}..{om_idx[-1]}, gaps={int(np.sum(np.diff(om_idx) > 1))}")

    gt_rel = evaluate.align_to_first_pose(gt)
    om_rel = evaluate.align_to_first_pose(om_poses)
    m = evaluate.compute_ape_rpe([gt_rel[i] for i in om_idx], om_rel)
    print(f"[drop-aware] odometry APE rmse = {m['ape_rmse']:.3f} m "
          f"(RPE {m.get('rpe_rmse', float('nan')):.4f} m)")

    mp_stamps, mp_poses = load_tum(args.map_poses)
    mp_idx = frame_indices(mp_stamps, args.rate, n_frames)
    mp_rel = evaluate.align_to_first_pose(mp_poses)
    m2 = evaluate.compute_ape_rpe([gt_rel[i] for i in mp_idx], mp_rel)
    print(f"[drop-aware] PGO APE rmse      = {m2['ape_rmse']:.3f} m "
          f"(RPE {m2.get('rpe_rmse', float('nan')):.4f} m)")

    if args.loops_json and args.keyframes_json:
        import json
        loops = json.load(open(args.loops_json))
        kfs = json.load(open(args.keyframes_json))
        # Loop (i, j) indices are loop-node scan indices; map them to true
        # frames via the timestamp grid, then report GT distances.
        kf_frames = {k: int(mp_idx[k]) for k in kfs if k < len(mp_idx)}
        dists = []
        for lc in loops:
            if lc["i"] not in kf_frames or lc["j"] not in kf_frames:
                continue
            fi, fj = kf_frames[lc["i"]], kf_frames[lc["j"]]
            d = float(np.linalg.norm(gt[fi][:3, 3] - gt[fj][:3, 3]))
            dists.append((lc["i"], lc["j"], fi, fj, d, lc["overlap"]))
        far = [x for x in dists if x[4] > 10.0]
        print(f"accepted loops with GT distance > 10 m (potential false): "
              f"{len(far)} / {len(dists)}")
        for x in sorted(far, key=lambda x: -x[4])[:10]:
            print(f"  node({x[0]},{x[1]}) -> frames({x[2]},{x[3]}) "
                  f"GT dist {x[4]:.1f} m overlap {x[5]:.2f}")
        near = [x[4] for x in dists if x[4] <= 10.0]
        if near:
            print(f"true-loop GT distance: median {np.median(near):.2f} m, "
                  f"max {max(near):.2f} m (n={len(near)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
