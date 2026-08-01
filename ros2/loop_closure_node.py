#!/usr/bin/env python3
"""Step-3 loop-closure + PGO node: Scan Context / proximity candidates ->
RAP verification (via rap_server.py) -> GTSAM PGO -> map->odom correction TF.

Runs in the SYSTEM python3 (rclpy). The heavy lifting (RAP on GPU, GTSAM)
lives in rap_server.py, which runs in the conda env; this node talks to it
over a localhost TCP socket (length-prefixed pickle, see rap_server.py).
Scan Context and keyframe selection are pure NumPy and run in-process,
reusing the offline pipeline's modules so live behavior matches batch runs.

    source /opt/ros/jazzy/setup.bash
    python3 loop_closure_node.py --ros-args -p save_dir:=/path/to/out

Topics / frames:
    sub  /points_raw   (sensor_msgs/PointCloud2, Reliable)
    sub  /odom         (nav_msgs/Odometry from odometry_node.py)
    pub  /map_path     (nav_msgs/Path, map frame — PGO-corrected trajectory)
    pub  /loop_edges   (visualization_msgs/Marker LINE_LIST, map frame)
    pub  /map_cloud    (sensor_msgs/PointCloud2, map frame — accumulated map,
                        rebuilt from stored scans at their latest corrected
                        poses after every PGO and every map_period scans)
    TF   map -> odom (dynamic correction; identity until the first accepted loop)

Frames chain: map -> odom -> base_link -> velodyne.
"""

import json
import os
import pickle
import queue
import socket
import struct
import sys
import threading
import time

import numpy as np
import rclpy
import rclpy.executors
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rap_slam import scan_context as sc_mod  # noqa: E402
from rap_slam.cluster_gate import ClusterGate  # noqa: E402
from rap_slam.pose_graph import propagate_to_full_trajectory  # noqa: E402


# ---------------------------------------------------------------------------
# Small math / io helpers (duplicated from odometry_node.py to keep each
# node standalone-runnable)
# ---------------------------------------------------------------------------
def rot_to_quat(R: np.ndarray):
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


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 2.0 / n if n > 0 else 0.0
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def odom_msg_to_pose(msg: Odometry) -> np.ndarray:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """PointCloud2 -> (N, 3) float32 (fast path for dense float32 xyzi)."""
    f = msg.fields
    if (msg.point_step == 16 and len(f) >= 3
            and f[0].name == "x" and f[0].offset == 0
            and f[1].name == "y" and f[1].offset == 4
            and f[2].name == "z" and f[2].offset == 8):
        raw = np.frombuffer(msg.data, dtype=np.float32)
        return raw.reshape(-1, 4)[:, :3].copy()
    from sensor_msgs_py import point_cloud2
    pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)


def _yaw_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def _rotation_error_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """NumPy voxel downsample: keep the first point landing in each voxel."""
    if voxel <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def xyz_to_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Build a dense xyz float32 PointCloud2 without the slow .tolist() path
    (matters at ~1e5-1e6 points; create_cloud(tolist()) capped the player)."""
    from sensor_msgs.msg import PointField
    pts = np.ascontiguousarray(points, dtype=np.float32)
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(pts)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32,
                             count=1) for i, n in enumerate("xyz")]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(pts)
    msg.data = pts.tobytes()
    msg.is_dense = True
    return msg


def write_ply_xyz(path: str, points: np.ndarray) -> None:
    """Minimal binary-little-endian PLY (x y z float), no open3d needed."""
    pts = np.ascontiguousarray(points, dtype=np.float32)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())


def _recv_msg(conn: socket.socket):
    header = b""
    while len(header) < 8:
        chunk = conn.recv(8 - len(header))
        if not chunk:
            raise ConnectionError("server closed connection")
        header += chunk
    (n,) = struct.unpack(">Q", header)
    payload = b""
    while len(payload) < n:
        chunk = conn.recv(min(1 << 20, n - len(payload)))
        if not chunk:
            raise ConnectionError("server dropped mid-message")
        payload += chunk
    return pickle.loads(payload)


class RapClient:
    """Persistent TCP client for rap_server.py (used only by the worker thread)."""

    def __init__(self, host: str, port: int, logger, stop: threading.Event):
        self.host, self.port = host, port
        self.logger = logger
        self.sock = None
        while not stop.is_set():
            try:
                self.sock = socket.create_connection((host, port), timeout=5.0)
                self.sock.settimeout(600.0)  # RAP inference + PGO can take seconds
                self.request({"cmd": "ping"})
                return
            except OSError:
                if self.sock is not None:
                    self.sock.close()
                    self.sock = None
                time.sleep(1.0)
        raise RuntimeError("stopped before the RAP server became reachable")

    def request(self, d: dict) -> dict:
        payload = pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self.sock.sendall(struct.pack(">Q", len(payload)) + payload)
            resp = _recv_msg(self.sock)
        except OSError as e:
            try:
                self.sock.close()
            finally:
                self.sock = None
            raise ConnectionError(f"RAP server request failed: {e}")
        return resp


class LoopClosureNode(Node):
    def __init__(self):
        super().__init__("loop_closure")

        self.declare_parameter("topic", "/points_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 5557)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        # Offline pipeline parameters (run_pipeline.py POSS run):
        self.declare_parameter("kf_trans", 6.0)
        self.declare_parameter("kf_rot", 12.0)
        self.declare_parameter("loop_sep", 200)      # offline POSS used --loop-sep 200
        self.declare_parameter("loop_dist", 15.0)
        self.declare_parameter("sc_thresh", 0.25)
        self.declare_parameter("sc_shortlist", 10)   # deprecated (ring-key prefilter removed 2026-07-31)
        self.declare_parameter("sc_max_radius", 80.0)
        self.declare_parameter("min_overlap", 0.7)   # acceptance gate (see below)
        self.declare_parameter("max_mean_dist", 2.5)  # m; 0 disables the 2nd gate
        self.declare_parameter("max_per_kf", 3)      # candidate budget per keyframe
        self.declare_parameter("max_total", 0)       # total verification budget (0 = unlimited)
        # Retrieval rework 2026-07-31: full-scan SC + cluster-persistence gate.
        self.declare_parameter("sc_strong", 0.12)    # SC score skipping the gate
        self.declare_parameter("min_cluster", 2)     # cluster size to promote
        self.declare_parameter("max_per_cluster", 2) # persistence promotions / cluster
        self.declare_parameter("cluster_w", 50)      # (i,j) Chebyshev window (scans)
        self.declare_parameter("cluster_spacing", 25)  # in-cluster NMS (scans)
        self.declare_parameter("pgo_max_wait", 30.0)  # s; run pending PGO even if the queue is busy
        self.declare_parameter("save_dir", "")
        # Accumulated-map publication (/map_cloud):
        self.declare_parameter("map_voxel", 0.3)     # per-scan pre-downsample (m)
        self.declare_parameter("map_stride", 5)      # keep every k-th scan
        self.declare_parameter("map_max_points", 500000)
        self.declare_parameter("map_period", 200)    # scans between map republishes

        gp = self.get_parameter
        self.map_frame = gp("map_frame").value
        self.odom_frame = gp("odom_frame").value
        self.kf_trans = float(gp("kf_trans").value)
        self.kf_rot = float(gp("kf_rot").value)
        self.loop_sep = int(gp("loop_sep").value)
        self.loop_dist = float(gp("loop_dist").value)
        self.sc_thresh = float(gp("sc_thresh").value)
        self.sc_shortlist = int(gp("sc_shortlist").value)
        self.sc_max_radius = float(gp("sc_max_radius").value)
        self.min_overlap = float(gp("min_overlap").value)
        self.max_mean_dist = float(gp("max_mean_dist").value)
        self.max_per_kf = int(gp("max_per_kf").value)
        self.max_total = int(gp("max_total").value)
        self.pgo_max_wait = float(gp("pgo_max_wait").value)
        self.gate = ClusterGate(
            window=int(gp("cluster_w").value),
            min_cluster=int(gp("min_cluster").value),
            max_per_cluster=int(gp("max_per_cluster").value),
            spacing=int(gp("cluster_spacing").value),
            strong_thresh=float(gp("sc_strong").value))
        self.save_dir = str(gp("save_dir").value)
        self.map_voxel = float(gp("map_voxel").value)
        self.map_stride = max(1, int(gp("map_stride").value))
        self.map_max_points = int(gp("map_max_points").value)
        self.map_period = int(gp("map_period").value)

        # --- trajectory state (spin thread appends; worker reads snapshots) ---
        self.poses = []        # odom 4x4 per scan, arrival order == offline index
        self.stamps = []       # float seconds per scan
        self.keyframes = []    # scan indices
        self.kf_scans = {}     # keyframe idx -> (N,3) float32 raw scan
        self._sc_db_indices = []  # keyframe scan indices in arrival order
        self._sc_db = []       # Scan Context per keyframe (aligned with _sc_db_indices)
        self.loops = []        # accepted loop dicts {i, j, T_rel, overlap, ...}
        self.optimized = None  # {keyframe idx: 4x4} from the latest PGO
        self.kfs_at_pgo = []   # keyframe list snapshot used by the latest PGO
        self.count = 0
        self._acc_trans = 0.0
        self._acc_rot = 0.0
        self._prev_yaw = 0.0

        # --- stamp-pairing buffers (cloud stamp == odom stamp by construction) ---
        self._cloud_buf = {}
        self._odom_buf = {}
        self._last_key = None  # newest processed (sec, nanosec)
        self._dropped = 0

        # --- candidate pipeline to the worker thread (best-first) ---
        self._queue = queue.PriorityQueue()
        self._qc = 0           # tie-break counter for queue items
        self._proposed = set()
        self._queued_pairs = set()
        self._verified = 0
        self._stop = threading.Event()

        # --- accumulated map state (sensor-frame downsampled clouds) ---
        self._map_scans = []      # list of (scan_idx, (N,3) float32)
        self._map_request = False  # worker picks this up and republishes

        sub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE, depth=50)
        odom_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE, depth=200)
        self.create_subscription(PointCloud2, gp("topic").value, self._on_cloud, sub_qos)
        self.create_subscription(Odometry, gp("odom_topic").value, self._on_odom, odom_qos)

        # Latched-style outputs so RViz shows the latest state whenever it starts.
        out_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.path_pub = self.create_publisher(Path, "/map_path", out_qos)
        self.marker_pub = self.create_publisher(Marker, "/loop_edges", out_qos)
        self.map_pub = self.create_publisher(PointCloud2, "/map_cloud", out_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.1, self._publish_correction_tf)   # 10 Hz map->odom
        self.create_timer(1.0, self._republish_path)          # keep path fresh

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"Loop-closure node up: {gp('topic').value} + {gp('odom_topic').value} -> "
            f"/map_path + /loop_edges + TF map->odom; server "
            f"{gp('server_host').value}:{gp('server_port').value}, "
            f"sep={self.loop_sep} sc<{self.sc_thresh} "
            f"strong<{self.gate.strong_thresh} cluster>={self.gate.min_cluster} "
            f"overlap>={self.min_overlap} mean_dist<={self.max_mean_dist}"
            + (f", saving to {self.save_dir}" if self.save_dir else ""))

    # ------------------------------------------------------------------
    # Spin thread: pairing + keyframes + candidate proposal
    # ------------------------------------------------------------------
    def _on_cloud(self, msg: PointCloud2):
        k = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._cloud_buf[k] = msg
        self._drain_pairs()

    def _on_odom(self, msg: Odometry):
        k = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._odom_buf[k] = odom_msg_to_pose(msg)
        self._drain_pairs()

    def _drain_pairs(self):
        for k in sorted(set(self._cloud_buf) & set(self._odom_buf)):
            cloud = self._cloud_buf.pop(k)
            pose = self._odom_buf.pop(k)
            self._process_scan(k, cloud, pose)
            self._last_key = k
        # Expire messages the other side will never match (dropped frames).
        if self._last_key is not None:
            for buf in (self._cloud_buf, self._odom_buf):
                stale = [k for k in buf if k < self._last_key]
                for k in stale:
                    buf.pop(k)
                    self._dropped += 1

    def _process_scan(self, key, cloud_msg: PointCloud2, pose: np.ndarray):
        idx = self.count
        self.poses.append(pose)
        self.stamps.append(key[0] + key[1] * 1e-9)

        is_kf = False
        if idx == 0:
            is_kf = True
            self._prev_yaw = _yaw_of(pose)
        else:
            step = float(np.linalg.norm(pose[:3, 3] - self.poses[idx - 1][:3, 3]))
            yaw = _yaw_of(pose)
            dyaw = (yaw - self._prev_yaw + np.pi) % (2 * np.pi) - np.pi
            self._prev_yaw = yaw
            self._acc_trans += step
            self._acc_rot += abs(dyaw)
            if self._acc_trans > self.kf_trans or self._acc_rot > np.deg2rad(self.kf_rot):
                is_kf = True
                self._acc_trans = 0.0
                self._acc_rot = 0.0

        need_cloud = is_kf or (idx % self.map_stride == 0)
        xyz = cloud_to_xyz(cloud_msg) if need_cloud else None

        if idx % self.map_stride == 0:
            self._map_scans.append((idx, voxel_downsample(xyz, self.map_voxel)))
            if self.map_period > 0 and idx > 0 and idx % self.map_period < self.map_stride:
                self._map_request = True  # worker republishes /map_cloud

        if is_kf:
            self.keyframes.append(idx)
            self.kf_scans[idx] = xyz
            sc = sc_mod.make_scan_context(xyz, max_radius=self.sc_max_radius)
            self.get_logger().info(
                f"keyframe #{len(self.keyframes)} at scan {idx} "
                f"pos=({pose[0, 3]:.1f}, {pose[1, 3]:.1f}, {pose[2, 3]:.1f})")
            if len(self.keyframes) > 1:
                self._propose_candidates(idx, sc)
            # Register in the SC database AFTER proposing (earlier keyframes
            # only are matched against the new one).
            self._sc_db_indices.append(idx)
            self._sc_db.append(sc)

        self.count += 1
        if self.count % 200 == 0:
            self.get_logger().info(
                f"[{self.count} scans] keyframes={len(self.keyframes)} "
                f"loops={len(self.loops)} queue={self._queue.qsize()} "
                f"dropped={self._dropped}")

    def _propose_candidates(self, j: int, sc_j: np.ndarray):
        """Proximity + full-scan Scan Context candidates for keyframe j,
        filtered through the cluster-persistence gate (2026-07-31 rework).

        Replaces the old ring-key shortlist: SC distances are computed
        against ALL earlier keyframes (vectorized, no prefilter), then the
        ClusterGate decides which pairs reach the verification queue —
        strong SC hits and proximity hits queue immediately, weaker SC
        hits must persist (their (i, j) cluster must reach min_cluster
        members). The queue is best-first: SC hits (by SC distance)
        before proximity hits."""
        pos_j = self.poses[j][:3, 3]
        earlier = [i for i in self._sc_db_indices if j - i >= self.loop_sep]
        cand = []  # (tier, score, i, yaw_hint): tier 0 = SC, 1 = proximity
        if earlier:
            db = np.stack([self._sc_db[self._sc_db_indices.index(i)]
                           for i in earlier])
            dists, shifts = sc_mod.sc_distances_batch(db, sc_j)
            for i, d, shift in zip(earlier, dists, shifts):
                if d < self.sc_thresh:
                    cand.append((0, float(d), i, float(shift) * 6.0))
        for i in earlier:
            d = float(np.linalg.norm(pos_j - self.poses[i][:3, 3]))
            if d < self.loop_dist:
                cand.append((1, d / self.loop_dist, i, None))
        cand.sort(key=lambda c: (c[0], c[1]))

        added = 0
        for tier, score, i, yaw_hint in cand:
            if added >= self.max_per_kf:
                break
            key = (min(i, j), max(i, j))
            if key in self._proposed:
                continue
            self._proposed.add(key)
            added += 1
            src = ("proximity" if tier == 1
                   else f"sc(yaw_hint={yaw_hint:.0f}deg)")
            for qi, qj, qtier, qscore in self.gate.add(key[0], key[1],
                                                       tier, score):
                qkey = (qi, qj)
                if qkey in self._queued_pairs:
                    continue
                if (self.max_total > 0 and
                        self._verified + self._queue.qsize() >= self.max_total):
                    break
                self._queued_pairs.add(qkey)
                self._queue.put(((qtier, qscore, self._qc), qkey))
                self._qc += 1
                self.get_logger().info(
                    f"candidate ({qkey[0]}, {qkey[1]}) queued by {src} "
                    f"(score={qscore:.3f})")

    # ------------------------------------------------------------------
    # Worker thread: RAP verification + PGO (via rap_server)
    # ------------------------------------------------------------------
    def _worker_loop(self):
        try:
            client = RapClient(self.get_parameter("server_host").value,
                               int(self.get_parameter("server_port").value),
                               self.get_logger(), self._stop)
        except RuntimeError:
            return
        self.get_logger().info("RAP server connected")

        pgo_pending = False
        pgo_since = 0.0
        was_busy = False
        while not self._stop.is_set():
            # A busy queue must not starve PGO forever: force the pending
            # optimization once it has waited pgo_max_wait seconds.
            if pgo_pending and time.time() - pgo_since > self.pgo_max_wait:
                self._run_pgo(client)  # also republishes the map
                pgo_pending = False
            try:
                _, (i, j) = self._queue.get(timeout=0.5)
            except queue.Empty:
                if pgo_pending:
                    self._run_pgo(client)  # also republishes the map
                    pgo_pending = False
                elif self._map_request:
                    self._map_request = False
                    self._publish_map()
                elif was_busy:
                    was_busy = False
                    self.get_logger().info(
                        f"Loop worker idle: verified={self._verified} "
                        f"accepted={len(self.loops)}")
                continue
            was_busy = True
            if self._verify_pair(client, i, j):
                if not pgo_pending:
                    pgo_since = time.time()
                pgo_pending = True
        self.get_logger().info("Loop worker stopped")

    def _verify_pair(self, client, i: int, j: int) -> bool:
        t0 = time.time()
        try:
            resp = client.request({
                "cmd": "register",
                "points_a": self.kf_scans[i],
                "points_b": self.kf_scans[j],
            })
        except (ConnectionError, RuntimeError) as e:
            self.get_logger().error(f"verification of ({i}, {j}) failed: {e}")
            return False
        self._verified += 1
        if not resp.get("ok"):
            self.get_logger().warn(f"RAP rejected ({i}, {j}): {resp.get('error')}")
            return False

        T_rel = resp["T_rel"]
        overlap = resp["overlap"]
        mean_dist = resp.get("mean_dist", float("nan"))
        T_odom = np.linalg.inv(self.poses[i]) @ self.poses[j]
        dT = np.linalg.inv(T_odom) @ T_rel
        trans_err = float(np.linalg.norm(dT[:3, 3]))
        rot_err = _rotation_error_deg(dT[:3, :3])
        # Data-driven acceptance gate (eval_acceptance_gate.py, KITTI 02):
        # true loops overlap 0.76-0.96 / mean_dist 0.39-1.45 m, false loops
        # overlap <= 0.64 / mean_dist >= 3.7 m for gross failures (two clever
        # urban false hits reached overlap 0.64 / mean_dist 1.26 m, so the
        # overlap threshold carries the separation: gap is 0.64 < t < 0.76).
        # Odometry consistency stays diagnostic-only (offline lesson).
        ok = (overlap >= self.min_overlap and
              (self.max_mean_dist <= 0.0 or mean_dist <= self.max_mean_dist))
        self.get_logger().info(
            f"pair ({i:3d}, {j:3d}): overlap={overlap:.2f} "
            f"mean_dist={mean_dist:.2f} m "
            f"odom-consistency dT={trans_err:.2f} m dR={rot_err:.1f} deg "
            f"[{resp['t_infer']:.1f} s infer, {time.time() - t0:.1f} s total] "
            f"=> {'ACCEPT' if ok else 'reject'}")
        if not ok:
            return False
        self.loops.append({
            "i": i, "j": j, "T_rel": T_rel, "overlap": overlap,
            "mean_dist": mean_dist,
            "odom_trans_err": trans_err, "odom_rot_err_deg": rot_err,
            "t_infer": resp["t_infer"],
        })
        return True

    def _run_pgo(self, client):
        kfs = list(self.keyframes)
        odom_kf = {k: self.poses[k] for k in kfs}
        loops = [{"i": lc["i"], "j": lc["j"], "T_rel": lc["T_rel"]}
                 for lc in self.loops]
        t0 = time.time()
        try:
            resp = client.request({"cmd": "pgo", "odom_kf": odom_kf,
                                   "keyframes": kfs, "loops": loops})
        except (ConnectionError, RuntimeError) as e:
            self.get_logger().error(f"PGO request failed: {e}")
            return
        if not resp.get("ok"):
            self.get_logger().error(f"PGO rejected: {resp.get('error')}")
            return
        self.optimized = resp["optimized"]
        self.kfs_at_pgo = kfs
        self.get_logger().info(
            f"PGO done: {len(kfs)} keyframes, {len(loops)} loop edges, "
            f"{time.time() - t0:.2f} s — trajectory corrected")
        self._publish_map_path()
        self._publish_loop_marker()
        self._publish_map()

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------
    def _current_correction(self) -> np.ndarray:
        """T_map_odom from the latest PGO (identity before the first loop)."""
        if self.optimized is None or not self.kfs_at_pgo:
            return np.eye(4)
        k = self.kfs_at_pgo[-1]
        return self.optimized[k] @ np.linalg.inv(self.poses[k])

    def _publish_correction_tf(self):
        C = self._current_correction()
        qx, qy, qz, qw = rot_to_quat(C[:3, :3])
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.odom_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = C[:3, 3]
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    def _corrected_trajectory(self):
        if self.optimized is None:
            return list(self.poses)
        return propagate_to_full_trajectory(self.poses, self.kfs_at_pgo,
                                            self.optimized)

    def _publish_map_path(self):
        corrected = self._corrected_trajectory()
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for T, s in zip(corrected, self.stamps):
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            sec = int(s)
            ps.header.stamp.sec = sec
            ps.header.stamp.nanosec = int((s - sec) * 1e9)
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = T[:3, 3]
            qx, qy, qz, qw = rot_to_quat(T[:3, :3])
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)
        self.path_pub.publish(path)

    def _republish_path(self):
        if self.optimized is not None:
            self._publish_map_path()

    def _build_map_cloud(self) -> np.ndarray:
        """Merge the stored scans at their latest corrected poses (map frame)."""
        corrected = self._corrected_trajectory()
        chunks = []
        for idx, pts in self._map_scans:
            if idx >= len(corrected) or len(pts) == 0:
                continue
            T = corrected[idx]
            chunks.append((T[:3, :3] @ pts.T).T + T[:3, 3])
        if not chunks:
            return np.zeros((0, 3), dtype=np.float32)
        merged = np.concatenate(chunks).astype(np.float32)
        if len(merged) > self.map_max_points:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(merged), self.map_max_points, replace=False)
            merged = merged[sel]
        return merged

    def _publish_map(self):
        t0 = time.time()
        merged = self._build_map_cloud()
        if len(merged) == 0:
            return
        self.map_pub.publish(xyz_to_pointcloud2(
            merged, self.map_frame, self.get_clock().now().to_msg()))
        self.get_logger().info(
            f"map published: {len(merged)} points from {len(self._map_scans)} "
            f"scans ({time.time() - t0:.2f} s)")

    def _publish_loop_marker(self):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "loops"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.35
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 1.0, 1.0
        m.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point
        for lc in self.loops:
            for k in (lc["i"], lc["j"]):
                p = Point()
                p.x, p.y, p.z = (float(v) for v in self.optimized[k][:3, 3])
                m.points.append(p)
        self.marker_pub.publish(m)

    # ------------------------------------------------------------------
    # Shutdown: persist products for the headless regression
    # ------------------------------------------------------------------
    def destroy_node(self):
        self._stop.set()
        self._worker.join(timeout=5.0)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "accepted_loops.json"), "w") as f:
                json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in lc.items()} for lc in self.loops], f, indent=2)
            with open(os.path.join(self.save_dir, "keyframes.json"), "w") as f:
                json.dump(self.keyframes, f)
            corrected = self._corrected_trajectory()
            with open(os.path.join(self.save_dir, "map_poses.txt"), "w") as f:
                for T, s in zip(corrected, self.stamps):
                    qx, qy, qz, qw = rot_to_quat(T[:3, :3])
                    t = T[:3, 3]
                    f.write(f"{s:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                            f"{qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")
            merged = self._build_map_cloud()
            if len(merged) > 0:
                write_ply_xyz(os.path.join(self.save_dir, "merged_map_live.ply"),
                              merged)
            self.get_logger().info(
                f"Saved {len(self.loops)} loops + {len(corrected)} corrected "
                f"poses + {len(merged)} map points to {self.save_dir}")
        super().destroy_node()


def main():
    rclpy.init()
    node = LoopClosureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass  # SIGTERM (run script teardown) — products still saved below
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
