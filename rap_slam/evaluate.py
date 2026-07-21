"""Trajectory evaluation with evo (APE/RPE), plotting, and map merging."""

import logging
import os
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def align_to_first_pose(poses: List[np.ndarray]) -> List[np.ndarray]:
    """Express poses in the first pose's frame (removes the gauge freedom)."""
    T0_inv = np.linalg.inv(poses[0])
    return [T0_inv @ T for T in poses]


def compute_ape_rpe(gt_poses: List[np.ndarray], est_poses: List[np.ndarray],
                    align: bool = False) -> Dict[str, float]:
    """Compute evo APE and per-frame RPE (translation part).

    Args:
        align: if True, apply evo's SE3 Umeyama alignment first. When GT and
            estimate are already expressed in a common frame (e.g. via
            align_to_first_pose), leave False.

    Returns:
        Dict with APE (rmse/mean/max) and RPE (rmse/mean) statistics.
    """
    from evo.core import metrics, trajectory
    from evo.core.trajectory import PoseTrajectory3D

    stamps = np.arange(len(gt_poses), dtype=float)
    traj_ref = PoseTrajectory3D(poses_se3=[np.asarray(T) for T in gt_poses],
                                timestamps=stamps)
    traj_est = PoseTrajectory3D(poses_se3=[np.asarray(T) for T in est_poses],
                                timestamps=stamps)
    if align:
        traj_est.align(traj_ref, correct_scale=False)

    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    ape_stats = ape_metric.get_all_statistics()

    rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part,
                             delta=1, delta_unit=metrics.Unit.frames,
                             all_pairs=False)
    rpe_metric.process_data((traj_ref, traj_est))
    rpe_stats = rpe_metric.get_all_statistics()

    return {
        "ape_rmse": float(ape_stats["rmse"]),
        "ape_mean": float(ape_stats["mean"]),
        "ape_max": float(ape_stats["max"]),
        "ape_std": float(ape_stats["std"]),
        "rpe_rmse": float(rpe_stats["rmse"]),
        "rpe_mean": float(rpe_stats["mean"]),
        "rpe_max": float(rpe_stats["max"]),
        "rpe_std": float(rpe_stats["std"]),
    }


def plot_trajectories(gt_poses, odom_poses, opt_poses, loop_constraints,
                      keyframes, out_path: str, title: str = ""):
    """Top-down XY plot of GT vs odometry vs optimized trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_xy = np.array([T[:3, 3] for T in gt_poses])[:, :2]
    odo_xy = np.array([T[:3, 3] for T in odom_poses])[:, :2]
    opt_xy = np.array([T[:3, 3] for T in opt_poses])[:, :2]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", lw=2.0, label="Ground truth")
    ax.plot(odo_xy[:, 0], odo_xy[:, 1], "r--", lw=1.4, label="Odometry (KISS-ICP)")
    ax.plot(opt_xy[:, 0], opt_xy[:, 1], "g-.", lw=1.6, label="PGO optimized")

    kf_xy = opt_xy[np.asarray(keyframes)]
    ax.scatter(kf_xy[:, 0], kf_xy[:, 1], c="g", s=12, alpha=0.5, label="Keyframes")

    for lc in loop_constraints:
        p1, p2 = opt_xy[lc["i"]], opt_xy[lc["j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", lw=2.2, alpha=0.9)
    if loop_constraints:
        ax.plot([], [], "b-", lw=2.2, label=f"RAP loop closures ({len(loop_constraints)})")

    ax.scatter([gt_xy[0, 0]], [gt_xy[0, 1]], marker="o", s=90, c="lime",
               edgecolors="k", zorder=5, label="Start/End (loop)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Trajectory plot saved to {out_path}")


def merge_map(scans: List[np.ndarray], poses: List[np.ndarray],
              out_path: str, voxel_size: float = 0.3,
              stride: int = 1) -> str:
    """Merge scans into a map using the given poses; voxel-downsample; save PLY."""
    import open3d as o3d

    merged = o3d.geometry.PointCloud()
    for scan, pose in zip(scans[::stride], poses[::stride]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scan.astype(np.float64))
        pcd.transform(pose)
        merged += pcd
    if voxel_size > 0:
        merged = merged.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(out_path, merged, write_ascii=False)
    logger.info(f"Merged map saved to {out_path} ({len(merged.points)} points)")
    return out_path
