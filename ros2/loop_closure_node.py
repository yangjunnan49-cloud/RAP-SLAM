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
        self.declare_parameter("sc_shortlist", 10)
        self.declare_parameter("sc_max_radius", 80.0)
        self.declare_parameter("min_overlap", 0.25)  # acceptance: RAP overlap only
        self.declare_parameter("max_per_kf", 3)      # candidate budget per keyframe
        self.declare_parameter("max_total", 16)      # total verification budget
        self.declare_parameter("save_dir", "")

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
        self.max_per_kf = int(gp("max_per_kf").value)
        self.max_total = int(gp("max_total").value)
        self.save_dir = str(gp("save_dir").value)

        # --- trajectory state (spin thread appends; worker reads snapshots) ---
        self.poses = []        # odom 4x4 per scan, arrival order == offline index
        self.stamps = []       # float seconds per scan
        self.keyframes = []    # scan indices
        self.kf_scans = {}     # keyframe idx -> (N,3) float32 raw scan
        self.kf_desc = {}      # keyframe idx -> {"sc", "ring_key"}
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

        # --- candidate pipeline to the worker thread ---
        self._queue = queue.Queue()
        self._proposed = set()
        self._verified = 0
        self._stop = threading.Event()

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
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.1, self._publish_correction_tf)   # 10 Hz map->odom
        self.create_timer(1.0, self._republish_path)          # keep path fresh

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"Loop-closure node up: {gp('topic').value} + {gp('odom_topic').value} -> "
            f"/map_path + /loop_edges + TF map->odom; server "
            f"{gp('server_host').value}:{gp('server_port').value}, "
            f"sep={self.loop_sep} sc<{self.sc_thresh} overlap>={self.min_overlap}"
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

        if is_kf:
            xyz = cloud_to_xyz(cloud_msg)
            self.keyframes.append(idx)
            self.kf_scans[idx] = xyz
            sc = sc_mod.make_scan_context(xyz, max_radius=self.sc_max_radius)
            self.kf_desc[idx] = {"sc": sc, "ring_key": sc_mod.make_ring_key(sc)}
            self.get_logger().info(
                f"keyframe #{len(self.keyframes)} at scan {idx} "
                f"pos=({pose[0, 3]:.1f}, {pose[1, 3]:.1f}, {pose[2, 3]:.1f})")
            if len(self.keyframes) > 1:
                self._propose_candidates(idx)

        self.count += 1
        if self.count % 200 == 0:
            self.get_logger().info(
                f"[{self.count} scans] keyframes={len(self.keyframes)} "
                f"loops={len(self.loops)} queue={self._queue.qsize()} "
                f"dropped={self._dropped}")

    def _propose_candidates(self, j: int):
        """Proximity + Scan Context candidates for the new keyframe j."""
        pos_j = self.poses[j][:3, 3]
        prox = []
        earlier = [i for i in self.keyframes[:-1] if j - i >= self.loop_sep]
        for i in earlier:
            d = float(np.linalg.norm(pos_j - self.poses[i][:3, 3]))
            if d < self.loop_dist:
                prox.append((i, j, d, "proximity"))

        sc_hits = []
        rk_j = self.kf_desc[j]["ring_key"]
        shortlist = sorted(
            ((i, float(np.linalg.norm(rk_j - self.kf_desc[i]["ring_key"])))
             for i in earlier),
            key=lambda x: x[1])[:self.sc_shortlist]
        for i, _ in shortlist:
            d, shift = sc_mod.sc_distance(self.kf_desc[i]["sc"], self.kf_desc[j]["sc"])
            if d < self.sc_thresh:
                yaw_hint = shift * 6.0  # 360 deg / 60 sectors
                sc_hits.append((i, j, d, f"sc(yaw_hint={yaw_hint:.0f}deg)"))

        added = 0
        for i, j_, d, src in sorted(prox, key=lambda x: x[2]) + \
                sorted(sc_hits, key=lambda x: x[2]):
            if added >= self.max_per_kf:
                break
            if self._verified + self._queue.qsize() >= self.max_total:
                break
            key = (min(i, j_), max(i, j_))
            if key in self._proposed:
                continue
            self._proposed.add(key)
            self._queue.put(key)
            added += 1
            self.get_logger().info(
                f"candidate ({key[0]}, {key[1]}) proposed by {src} (d={d:.3f})")

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
        was_busy = False
        while not self._stop.is_set():
            try:
                i, j = self._queue.get(timeout=0.5)
            except queue.Empty:
                if pgo_pending:
                    self._run_pgo(client)
                    pgo_pending = False
                elif was_busy:
                    was_busy = False
                    self.get_logger().info(
                        f"Loop worker idle: verified={self._verified} "
                        f"accepted={len(self.loops)}")
                continue
            was_busy = True
            if self._verify_pair(client, i, j):
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
        T_odom = np.linalg.inv(self.poses[i]) @ self.poses[j]
        dT = np.linalg.inv(T_odom) @ T_rel
        trans_err = float(np.linalg.norm(dT[:3, 3]))
        rot_err = _rotation_error_deg(dT[:3, :3])
        # Acceptance on RAP overlap ONLY — gating on odometry consistency is
        # self-defeating under drift (offline lesson, see loop_closure.py).
        ok = overlap >= self.min_overlap
        self.get_logger().info(
            f"pair ({i:3d}, {j:3d}): overlap={overlap:.2f} "
            f"odom-consistency dT={trans_err:.2f} m dR={rot_err:.1f} deg "
            f"[{resp['t_infer']:.1f} s infer, {time.time() - t0:.1f} s total] "
            f"=> {'ACCEPT' if ok else 'reject'}")
        if not ok:
            return False
        self.loops.append({
            "i": i, "j": j, "T_rel": T_rel, "overlap": overlap,
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
            self.get_logger().info(
                f"Saved {len(self.loops)} loops + {len(corrected)} corrected "
                f"poses to {self.save_dir}")
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
