"""Frame-to-frame LiDAR odometry.

Primary: KISS-ICP. Fallback: Open3D point-to-plane ICP chained frame-to-frame.
Both produce sensor-to-world poses for every scan; relative transforms between
consecutive scans feed the pose graph.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class KissOdometry:
    """KISS-ICP odometry wrapper (kiss-icp >= 1.3)."""

    def __init__(self, max_range: float = 60.0, min_range: float = 1.5,
                 voxel_size: float = 0.8):
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP

        cfg = KISSConfig()
        cfg.data.max_range = float(max_range)
        cfg.data.min_range = float(min_range)
        cfg.data.deskew = False  # simulated scans carry no motion distortion
        cfg.mapping.voxel_size = float(voxel_size)  # must be set explicitly
        self._odom = KissICP(cfg)
        self._poses: List[np.ndarray] = []
        self.backend_name = "kiss-icp"

    def register(self, scan: np.ndarray) -> np.ndarray:
        """Register one scan (N, 3) and return the current sensor-to-world pose."""
        timestamps = np.zeros(len(scan), dtype=np.float64)
        self._odom.register_frame(scan.astype(np.float64), timestamps)
        pose = np.asarray(self._odom.last_pose, dtype=np.float64)
        self._poses.append(pose)
        return pose

    @property
    def poses(self) -> List[np.ndarray]:
        return self._poses


class Open3DICPOdometry:
    """Fallback odometry: chained Open3D point-to-plane ICP."""

    def __init__(self, voxel_size: float = 0.8, max_range: float = 60.0):
        self.voxel_size = float(voxel_size)
        self.max_range = float(max_range)
        self._poses: List[np.ndarray] = []
        self._prev = None
        self.backend_name = "open3d-icp"

    def _prep(self, scan: np.ndarray):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scan.astype(np.float64))
        pcd = pcd.voxel_down_sample(self.voxel_size)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 3, max_nn=30))
        return pcd

    def register(self, scan: np.ndarray) -> np.ndarray:
        import open3d as o3d
        curr = self._prep(scan)
        if self._prev is None:
            pose = np.eye(4)
        else:
            init_rel = (np.linalg.inv(self._poses[-2]) @ self._poses[-1]
                        if len(self._poses) > 1 else np.eye(4))
            result = o3d.pipelines.registration.registration_icp(
                curr, self._prev, self.voxel_size * 2.0, init_rel,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            T_curr_prev = result.transformation  # maps curr -> prev
            pose = self._poses[-1] @ T_curr_prev
        self._prev = curr
        self._poses.append(pose)
        return pose

    @property
    def poses(self) -> List[np.ndarray]:
        return self._poses


def run_odometry(scans: List[np.ndarray], prefer_kiss: bool = True,
                 max_range: float = 60.0) -> tuple:
    """Run odometry over all scans.

    Returns:
        poses: list of (4, 4) sensor-to-world poses (len == len(scans)).
        backend_name: which backend actually ran.
    """
    backend = None
    if prefer_kiss:
        try:
            candidate = KissOdometry(max_range=max_range)
            candidate.register(scans[0])  # smoke-test on the first scan
            backend = candidate
        except Exception as e:
            logger.warning(f"KISS-ICP failed to initialize/run ({e}); "
                           f"falling back to Open3D ICP")
            backend = None
    if backend is None:
        backend = Open3DICPOdometry(max_range=max_range)
        backend.register(scans[0])

    logger.info(f"Odometry backend: {backend.backend_name}")
    for i in range(1, len(scans)):
        backend.register(scans[i])
        if (i + 1) % 20 == 0:
            logger.info(f"  odometry {i + 1}/{len(scans)}")
    return backend.poses, backend.backend_name


def relative_transforms(poses: List[np.ndarray]) -> List[np.ndarray]:
    """Consecutive relative transforms T_i^{-1} @ T_{i+1} (pose of i+1 w.r.t. i)."""
    rels = []
    for i in range(len(poses) - 1):
        rels.append(np.linalg.inv(poses[i]) @ poses[i + 1])
    return rels
