#!/usr/bin/env python3
"""Step-2 KISS-ICP odometry node: PointCloud2 in, Odometry + TF + Path out.

Standalone rclpy node (no colcon build needed). Reuses the validated pipeline
wrapper `rap_slam.odometry.KissOdometry` (same parameters as the offline runs
that produced APE 8.26 m on POSS), so the live trajectory matches the offline
odometry one-to-one.

    source /opt/ros/jazzy/setup.bash
    python3 odometry_node.py
    python3 odometry_node.py --ros-args -p save_poses:=/tmp/live_poses.txt

Topics / frames:
    sub  /points_raw  (sensor_msgs/PointCloud2, Reliable - matches the player)
    pub  /odom        (nav_msgs/Odometry, odom -> base_link)
    pub  /odom_path   (nav_msgs/Path, odom frame)
    TF   odom -> base_link (dynamic), base_link -> velodyne (static identity)
"""

import os
import sys
import time

import numpy as np
import rclpy
import rclpy.executors  # for ExternalShutdownException
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

# rap_slam lives in the repo root (one level up from ros2/). Pure numpy +
# kiss_icp, so it imports fine into the system python3 that has rclpy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rap_slam.odometry import KissOdometry  # noqa: E402


def rot_to_quat(R: np.ndarray):
    """Rotation matrix -> quaternion (x, y, z, w), Shepperd's method."""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return x, y, z, w


def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """PointCloud2 -> (N, 3) float64. Fast path for dense float32 xyzi clouds
    (what pointcloud_player.py publishes); generic fallback otherwise."""
    f = msg.fields
    if (msg.point_step == 16 and len(f) >= 3
            and f[0].name == "x" and f[0].offset == 0
            and f[1].name == "y" and f[1].offset == 4
            and f[2].name == "z" and f[2].offset == 8):
        raw = np.frombuffer(msg.data, dtype=np.float32)
        return raw.reshape(-1, 4)[:, :3].astype(np.float64)
    from sensor_msgs_py import point_cloud2
    pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float64)


class KissOdometryNode(Node):
    def __init__(self):
        super().__init__("kiss_icp_odometry")

        self.declare_parameter("topic", "/points_raw")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("lidar_frame", "velodyne")
        # Same values as the offline POSS pipeline (run_pipeline.py passes
        # max_range=80.0; rap_slam defaults cover min_range/voxel_size).
        self.declare_parameter("max_range", 80.0)
        self.declare_parameter("min_range", 1.5)
        self.declare_parameter("voxel_size", 0.8)
        # Motion compensation: synthesize per-point azimuth timestamps and
        # enable KISS-ICP deskew (real spinning-LiDAR scans only). Default off
        # to preserve the verified golden trajectories bit-exact.
        self.declare_parameter("deskew", False)
        # Subscriber queue depth. 1 = always take the newest scan (drop frames
        # under CPU load instead of building up lag); raise it if every scan
        # must be processed and the CPU keeps up.
        self.declare_parameter("queue_depth", 1)
        # Optional TUM-format pose log (stamp tx ty tz qx qy qz qw per line),
        # appended incrementally so it survives SIGKILL.
        self.declare_parameter("save_poses", "")

        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        lidar_frame = self.get_parameter("lidar_frame").value

        self.kiss = KissOdometry(
            max_range=float(self.get_parameter("max_range").value),
            min_range=float(self.get_parameter("min_range").value),
            voxel_size=float(self.get_parameter("voxel_size").value),
            deskew=bool(self.get_parameter("deskew").value),
        )

        depth = int(self.get_parameter("queue_depth").value)
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=depth,
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.create_subscription(PointCloud2, self.get_parameter("topic").value,
                                 self._on_scan, sub_qos)
        self.odom_pub = self.create_publisher(Odometry, "/odom", pub_qos)
        self.path_pub = self.create_publisher(Path, "/odom_path", pub_qos)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Static identity base_link -> velodyne so RViz can resolve the cloud
        # frame through odom -> base_link -> velodyne.
        self.static_broadcaster = StaticTransformBroadcaster(self)
        st = TransformStamped()
        st.header.stamp = self.get_clock().now().to_msg()
        st.header.frame_id = self.base_frame
        st.child_frame_id = lidar_frame
        st.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(st)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame

        save_path = str(self.get_parameter("save_poses").value)
        self._pose_file = open(save_path, "w", buffering=1) if save_path else None

        self.count = 0
        self.get_logger().info(
            f"KISS-ICP odometry up: {self.get_parameter('topic').value} -> /odom + /odom_path, "
            f"TF {self.odom_frame}->{self.base_frame}->{lidar_frame}, queue_depth={depth}"
            + (f", logging poses to {save_path}" if save_path else "")
        )

    def _on_scan(self, msg: PointCloud2):
        t0 = time.perf_counter()
        xyz = cloud_to_xyz(msg)
        if len(xyz) == 0:
            return
        pose = self.kiss.register(xyz)
        dt_ms = (time.perf_counter() - t0) * 1e3

        R = pose[:3, :3]
        t = pose[:3, 3]
        qx, qy, qz, qw = rot_to_quat(R)
        stamp = msg.header.stamp  # match the cloud stamp so TF interpolation works

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = t
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = t
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.odom_frame
        ps.pose = odom.pose.pose
        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(ps)
        self.path_pub.publish(self.path_msg)

        if self._pose_file is not None:
            sec = stamp.sec + stamp.nanosec * 1e-9
            self._pose_file.write(
                f"{sec:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")

        self.count += 1
        if self.count % 50 == 0 or self.count == 1:
            self.get_logger().info(
                f"[{self.count}] pos=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) "
                f"pts={len(xyz)} register={dt_ms:.0f} ms"
            )

    def destroy_node(self):
        if self._pose_file is not None:
            self._pose_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = KissOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass  # SIGTERM (e.g. run_step2.sh teardown) — not an error
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
