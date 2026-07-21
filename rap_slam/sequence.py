"""Simulated LiDAR scan sequence from a real aggregated point-cloud map.

A virtual sensor drives a closed loop through the map. Ground-truth poses are
known exactly. Each scan is produced by collecting map points within a sensor
range of the pose, transforming them into the sensor frame, subsampling, and
adding Gaussian noise.
"""

import logging
import os
from typing import List, Tuple

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def load_map(map_path: str) -> Tuple[np.ndarray, o3d.geometry.KDTreeFlann]:
    """Load the aggregated map and build a KD-tree for range queries."""
    pcd = o3d.io.read_point_cloud(map_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError(f"Empty map: {map_path}")
    tree = o3d.geometry.KDTreeFlann(pcd)
    logger.info(f"Loaded map {map_path} with {len(points)} points")
    return points, tree


def _estimate_ground_z(map_points: np.ndarray, tree: o3d.geometry.KDTreeFlann,
                       x: float, y: float, query_radius: float = 20.0,
                       fallback_z: float = 5.0) -> float:
    """Estimate local ground height as a low percentile of nearby map z values."""
    # Use a mid-range z guess for the 3D ball query; refine below.
    idx = tree.search_radius_vector_3d(np.array([x, y, fallback_z]), query_radius)[1]
    if len(idx) < 50:
        return fallback_z
    z_vals = map_points[np.asarray(idx), 2]
    return float(np.percentile(z_vals, 10.0))


def generate_loop_poses(
    map_points: np.ndarray,
    tree: o3d.geometry.KDTreeFlann,
    center: Tuple[float, float] = (250.0, -30.0),
    radius: float = 80.0,
    n_poses: int = 100,
    sensor_height: float = 1.7,
    close_loop: bool = True,
) -> np.ndarray:
    """Generate GT sensor-to-world poses along a circle (mostly yaw rotation).

    Args:
        center: (x, y) circle center in the map frame.
        radius: circle radius in meters.
        n_poses: number of poses on the circle.
        sensor_height: sensor height above estimated local ground.
        close_loop: if True, append a final pose identical to pose 0.

    Returns:
        poses: (N, 4, 4) sensor-to-world transforms. Sensor convention: x forward,
            y left, z up (KITTI velodyne style), tangent to the circle.
    """
    cx, cy = center
    poses = []
    for k in range(n_poses):
        yaw_path = 2.0 * np.pi * k / n_poses
        x = cx + radius * np.cos(yaw_path)
        y = cy + radius * np.sin(yaw_path)
        ground_z = _estimate_ground_z(map_points, tree, x, y)
        z = ground_z + sensor_height
        # Heading tangent to the circle (counter-clockwise motion).
        yaw = yaw_path + np.pi / 2.0
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        poses.append(T)
    if close_loop:
        poses.append(poses[0].copy())
    return np.stack(poses)


def simulate_scan(
    map_points: np.ndarray,
    tree: o3d.geometry.KDTreeFlann,
    pose: np.ndarray,
    sensor_range: float = 50.0,
    min_range: float = 1.5,
    n_target: int = 5000,
    max_candidates: int = 60000,
    noise_sigma: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Simulate one scan: map points near the pose, in the sensor frame.

    Args:
        pose: (4, 4) sensor-to-world transform.
        sensor_range: max 3D range from the sensor.
        min_range: drop points closer than this.
        n_target: upper bound on kept points (random subset).
        max_candidates: pre-subsample cap before choosing the final subset.
        noise_sigma: Gaussian noise std (meters) added per axis.
        rng: numpy Generator.

    Returns:
        (N, 3) float32 point cloud in the sensor frame.
    """
    if rng is None:
        rng = np.random.default_rng()
    t = pose[:3, 3]
    idx = np.asarray(tree.search_radius_vector_3d(t, sensor_range)[1], dtype=np.int64)
    if len(idx) == 0:
        raise ValueError(f"No map points within {sensor_range} m of pose at {t}")
    if len(idx) > max_candidates:
        idx = rng.choice(idx, size=max_candidates, replace=False)
    pts_world = map_points[idx]

    # World -> sensor frame: p_s = R^T (p_w - t)
    R = pose[:3, :3]
    pts_sensor = (pts_world - t) @ R

    # Range filtering in the sensor frame.
    ranges = np.linalg.norm(pts_sensor, axis=1)
    keep = (ranges >= min_range) & (ranges <= sensor_range)
    pts_sensor = pts_sensor[keep]
    if len(pts_sensor) == 0:
        raise ValueError("All points filtered out by range limits")

    # Random subset down to n_target.
    if len(pts_sensor) > n_target:
        sel = rng.choice(len(pts_sensor), size=n_target, replace=False)
        pts_sensor = pts_sensor[sel]

    # Gaussian sensor noise.
    pts_sensor = pts_sensor + rng.normal(0.0, noise_sigma, pts_sensor.shape)
    return pts_sensor.astype(np.float32)


def generate_sequence(
    map_path: str,
    output_dir: str,
    center: Tuple[float, float] = (250.0, -30.0),
    radius: float = 80.0,
    n_poses: int = 100,
    sensor_range: float = 50.0,
    points_range: Tuple[int, int] = (3000, 8000),
    noise_sigma: float = 0.05,
    seed: int = 42,
    save_scans: bool = True,
) -> dict:
    """Generate the full simulated sequence and save GT + scans to output_dir.

    Returns dict with keys: gt_poses (N,4,4), scans (list of (Ni,3) float32),
    scan_files (list of str).
    """
    os.makedirs(output_dir, exist_ok=True)
    scan_dir = os.path.join(output_dir, "scans")
    if save_scans:
        os.makedirs(scan_dir, exist_ok=True)

    map_points, tree = load_map(map_path)
    gt_poses = generate_loop_poses(map_points, tree, center=center, radius=radius,
                                   n_poses=n_poses)
    rng = np.random.default_rng(seed)

    scans, scan_files = [], []
    for i, pose in enumerate(gt_poses):
        n_target = int(rng.integers(points_range[0], points_range[1] + 1))
        scan = simulate_scan(map_points, tree, pose, sensor_range=sensor_range,
                             n_target=n_target, noise_sigma=noise_sigma, rng=rng)
        scans.append(scan)
        if save_scans:
            path = os.path.join(scan_dir, f"scan_{i:04d}.ply")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scan.astype(np.float64))
            o3d.io.write_point_cloud(path, pcd, write_ascii=False)
            scan_files.append(path)
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"  generated scan {i + 1}/{len(gt_poses)} ({len(scan)} pts)")

    # Save GT trajectory in KITTI format (12 floats per line, row-major 3x4).
    gt_file = os.path.join(output_dir, "gt_poses.txt")
    save_kitti_poses(gt_poses, gt_file)
    logger.info(f"Saved {len(gt_poses)} GT poses to {gt_file}; {len(scans)} scans")

    return {"gt_poses": gt_poses, "scans": scans, "scan_files": scan_files,
            "gt_file": gt_file}


def save_kitti_poses(poses: np.ndarray, path: str) -> None:
    """Save (N, 4, 4) poses in KITTI format: one line per pose, 12 floats (3x4)."""
    with open(path, "w") as f:
        for T in poses:
            vals = T[:3, :].reshape(-1)
            f.write(" ".join(f"{v:.10f}" for v in vals) + "\n")


def load_kitti_poses(path: str) -> np.ndarray:
    """Load a KITTI-format pose file into (N, 4, 4)."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    poses = np.tile(np.eye(4), (len(data), 1, 1))
    poses[:, :3, :] = data.reshape(len(data), 3, 4)
    return poses
