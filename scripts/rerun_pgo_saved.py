"""Re-run PGO from saved live products with the CURRENT rap_slam.pose_graph
defaults, producing "current-configuration" trajectories for the paper
comparison table.

Motivation: saved live `map_poses.txt` files freeze the backend code at their
run date. Later backend changes (RC-1 relfloor+Huber default 2026-08-01;
loop-edge NMS live default w=10, 2026-08-08) improved PGO, so e.g. the saved
KITTI-02 live map (27.387 m) no longer reflects the current system
(expected ~26.0 m). This script re-runs ONLY the backend on the saved inputs
(odometry TUM + keyframes.json + accepted_loops.json) — no GPU, no ROS.

The live defaults are used (loop_nms_width=10) because the saved edges come
from streaming retrieval, where dense same-site clusters are the norm.

Run (WSL2, conda env rap-inference):
  python scripts/rerun_pgo_saved.py \
      --odom output_ros2/live_poses_kitti02.txt \
      --keyframes output_ros2/step3_kitti02/keyframes.json \
      --loops output_ros2/step3_kitti02/accepted_loops.json \
      --out output_ros2/step3_kitti02/map_poses_current.txt
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rap_slam import pose_graph  # noqa: E402

logger = logging.getLogger("rerun_pgo_saved")


def load_tum(path):
    """TUM rows (stamp tx ty tz qx qy qz qw) -> (stamps, list of 4x4)."""
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
    return rows[:, 0], poses


def mat2quat(R):
    """Rotation matrix -> quaternion (qx, qy, qz, qw), branch-free Shepperd."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw, qx, qy, qz = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, \
            (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw, qx, qy, qz = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, \
            (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw, qx, qy, qz = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, \
            (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw, qx, qy, qz = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, \
            0.25 * s
    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--odom", required=True, help="odometry TUM poses")
    p.add_argument("--keyframes", required=True, help="keyframes.json list")
    p.add_argument("--loops", required=True, help="accepted_loops.json list")
    p.add_argument("--out", required=True, help="output TUM poses")
    p.add_argument("--loop-nms-width", type=float, default=10.0,
                   help="live default is 10; 0 disables NMS")
    p.add_argument("--loop-weighting", default="relfloor")
    p.add_argument("--robust-kernel", default="huber")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stamps, odom_poses = load_tum(args.odom)
    keyframes = json.load(open(args.keyframes))
    loops = json.load(open(args.loops))
    for lc in loops:
        lc["T_rel"] = np.asarray(lc["T_rel"], dtype=float)
    logger.info(f"Loaded {len(odom_poses)} odom poses, {len(keyframes)} "
                f"keyframes, {len(loops)} loop edges")

    optimized = pose_graph.optimize_pose_graph(
        odom_poses, keyframes, loops,
        loop_weighting=args.loop_weighting,
        robust_kernel=args.robust_kernel,
        loop_nms_width=args.loop_nms_width)
    full = pose_graph.propagate_to_full_trajectory(
        odom_poses, keyframes, optimized)

    with open(args.out, "w") as f:
        for stamp, T in zip(stamps, full):
            qx, qy, qz, qw = mat2quat(T[:3, :3])
            f.write(f"{stamp:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} "
                    f"{qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")
    logger.info(f"Wrote {args.out} ({len(full)} poses)")


if __name__ == "__main__":
    main()
