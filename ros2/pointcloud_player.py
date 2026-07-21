#!/usr/bin/env python3
"""Step-1 visualization player: stream KITTI/POSS .bin scans as PointCloud2.

Standalone rclpy node (no colcon build needed):

    source /opt/ros/jazzy/setup.bash
    python3 pointcloud_player.py                      # uses POSS defaults
    python3 pointcloud_player.py --ros-args -p data_dir:=/path/to/velodyne -p rate:=5.0

Reads *.bin (float32 x,y,z,intensity) in filename order and publishes one
scan per timer tick. QoS is Reliable/Volatile on purpose: the default RViz
PointCloud2 display subscribes with Reliable, and on localhost there is no
packet loss to protect against, so this avoids the classic silent mismatch.
(Real sensor drivers and rosbag use BestEffort; RViz must then be switched.)
"""

import glob
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

DEFAULT_DATA_DIR = "/mnt/e/githouse/RAP-SLAM/data/poss/dataset/sequences/00/velodyne"


class BinPlayer(Node):
    def __init__(self):
        super().__init__("bin_pointcloud_player")

        self.declare_parameter("data_dir", DEFAULT_DATA_DIR)
        self.declare_parameter("topic", "/points_raw")
        self.declare_parameter("frame_id", "velodyne")
        self.declare_parameter("rate", 10.0)   # scans per second
        self.declare_parameter("start", -1)    # first frame number, -1 = first file
        self.declare_parameter("end", -1)      # last frame number, -1 = last file
        self.declare_parameter("loop", False)

        data_dir = self.get_parameter("data_dir").value
        self.topic = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value
        rate = float(self.get_parameter("rate").value)
        start = int(self.get_parameter("start").value)
        end = int(self.get_parameter("end").value)
        self.loop = bool(self.get_parameter("loop").value)

        files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if start >= 0 or end >= 0:
            lo = start if start >= 0 else -10**9
            hi = end if end >= 0 else 10**9
            files = [p for p in files
                     if lo <= int(os.path.splitext(os.path.basename(p))[0]) <= hi]
        if not files:
            raise RuntimeError(f"No .bin files found in {data_dir} (start={start}, end={end})")
        self.files = files
        self.idx = 0
        self._finished_logged = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.pub = self.create_publisher(PointCloud2, self.topic, qos)
        self.fields = [
            PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate(("x", "y", "z", "intensity"))
        ]
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"Playing {len(self.files)} scans from {data_dir} "
            f"at {rate:.1f} Hz on {self.topic} (frame_id={self.frame_id}, loop={self.loop})"
        )

    def _tick(self):
        if self.idx >= len(self.files):
            if self.loop:
                self.idx = 0
            else:
                if not self._finished_logged:
                    self.get_logger().info("Finished all scans. Node stays alive; Ctrl-C to exit.")
                    self._finished_logged = True
                return
        path = self.files[self.idx]
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)
        # tolist(): version-proof input for create_cloud (Nx4 numpy support varies)
        msg = point_cloud2.create_cloud(header, self.fields, pts.tolist())
        self.pub.publish(msg)
        if self.idx % 50 == 0:
            self.get_logger().info(
                f"[{self.idx + 1}/{len(self.files)}] {os.path.basename(path)}: {len(pts)} points"
            )
        self.idx += 1


def main():
    rclpy.init()
    node = BinPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
