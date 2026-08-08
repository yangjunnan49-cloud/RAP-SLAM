#!/usr/bin/env python3
"""Step-3 GPU worker: RAP verification + GTSAM PGO behind a localhost socket.

Runs in the conda env `rap-inference` (torch/CUDA/gtsam), NOT under ROS — the
ROS 2 nodes run in the system python3 where rclpy lives, and importing rclpy
into the conda env is not worth the pain. The loop-closure node connects over
TCP and exchanges length-prefixed pickled dicts (8-byte big-endian length):

    {"cmd": "ping"}                          -> {"ok": true}
    {"cmd": "register", "points_a": (N,3)f4,  -> {"ok": true, "T_rel": 4x4,
     "points_b": (M,3)f4}                        "overlap": float, ...}
    {"cmd": "pgo", "odom_kf": {k: 4x4},       -> {"ok": true,
     "keyframes": [k...], "loops": [...]}        "optimized": {k: 4x4}}

One persistent connection, requests handled sequentially (the RAP model is
not thread-safe anyway). Prints a readiness line that run scripts grep for:

    RAP server listening on 127.0.0.1:5557

Usage (WSL):
    /root/miniconda3/envs/rap-inference/bin/python rap_server.py [--port 5557]
"""

import argparse
import logging
import os
import pickle
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("rap_server")


def recv_msg(conn: socket.socket):
    """Read one length-prefixed pickled object; None on clean disconnect."""
    header = b""
    while len(header) < 8:
        chunk = conn.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    (n,) = struct.unpack(">Q", header)
    payload = b""
    while len(payload) < n:
        chunk = conn.recv(min(1 << 20, n - len(payload)))
        if not chunk:
            raise ConnectionError("connection dropped mid-message")
        payload += chunk
    return pickle.loads(payload)


def send_msg(conn: socket.socket, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">Q", len(payload)) + payload)


class Worker:
    """Loads RAP once, serves register + pgo requests."""

    def __init__(self, args):
        from rap_slam.rap_backend import RAPBackend

        t0 = time.time()
        self.rap = RAPBackend(
            rap_root=args.rap_root,
            sampling_steps=args.sampling_steps,
            voxel_size=args.voxel,
            max_points_per_scan=args.max_points,
            feature_extraction_on=True,
            seed=args.seed,
        )
        logger.info(f"RAP backend ready in {time.time() - t0:.1f} s")

    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True}
        if cmd == "register":
            return self._register(req)
        if cmd == "pgo":
            return self._pgo(req)
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    def _register(self, req: dict) -> dict:
        points_a = np.asarray(req["points_a"], dtype=np.float32)
        points_b = np.asarray(req["points_b"], dtype=np.float32)
        k = int(req.get("k", 1))  # RC-3: optional multi-generation verify
        if k > 1:
            T_rel, diag = self.rap.register_consistent(points_a, points_b,
                                                       k=k)
        else:
            T_rel, diag = self.rap.register(points_a, points_b)
        resp = {
            "ok": True,
            "T_rel": T_rel,
            "overlap": float(diag["overlap_ratio"]),
            "mean_dist": float(diag["mean_dist"]),
            "t_preprocess": float(diag["t_preprocess"]),
            "t_infer": float(diag["t_inference"]),
            "t_total": float(diag["t_total"]),
        }
        if k > 1:
            resp.update({
                "overlap_min": float(diag["overlap_min"]),
                "disp_t": float(diag["disp_t"]),
                "disp_r": float(diag["disp_r"]),
                "n_generations": int(diag["n_generations"]),
            })
        return resp

    def _pgo(self, req: dict) -> dict:
        from rap_slam.pose_graph import optimize_pose_graph

        # odom_kf arrives as {scan_idx: 4x4}; optimize_pose_graph only ever
        # indexes odom_poses[k] for k in keyframes, so the dict drops in
        # directly without materializing the full trajectory.
        t0 = time.time()
        optimized = optimize_pose_graph(
            odom_poses=req["odom_kf"],
            keyframes=list(req["keyframes"]),
            loop_constraints=req["loops"],
            loop_weighting=req.get("loop_weighting", "relfloor"),
            robust_kernel=req.get("robust_kernel", "huber"),
            huber_k=float(req.get("huber_k", 1.0)),
            dcs_phi=float(req.get("dcs_phi", 10.0)),
            relfloor_ref=float(req.get("relfloor_ref", 1.0)),
            relfloor_cap=float(req.get("relfloor_cap", 3.0)),
        )
        logger.info(f"PGO: {len(req['keyframes'])} nodes, "
                    f"{len(req['loops'])} loop edges "
                    f"(weighting={req.get('loop_weighting', 'fixed')}, "
                    f"kernel={req.get('robust_kernel')}), "
                    f"{time.time() - t0:.2f} s")
        return {"ok": True, "optimized": optimized}


def main():
    p = argparse.ArgumentParser(description="RAP-SLAM step-3 GPU worker")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5557)
    p.add_argument("--rap-root", default="/mnt/e/githouse/RAP-main")
    p.add_argument("--sampling-steps", type=int, default=5)
    p.add_argument("--voxel", type=float, default=0.4)
    p.add_argument("--max-points", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    worker = Worker(args)  # slow: torch import + checkpoint load (~30 s)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    # Readiness marker — run_step3.sh / verify_step3_headless.sh grep for it.
    print(f"RAP server listening on {args.host}:{args.port}", flush=True)

    while True:
        conn, addr = srv.accept()
        logger.info(f"client connected: {addr}")
        try:
            while True:
                req = recv_msg(conn)
                if req is None:
                    break
                try:
                    resp = worker.handle(req)
                except Exception as e:  # keep serving after a bad request
                    logger.exception(f"request {req.get('cmd')!r} failed")
                    resp = {"ok": False, "error": str(e)}
                send_msg(conn, resp)
        except (ConnectionError, OSError) as e:
            logger.warning(f"client connection error: {e}")
        finally:
            conn.close()
            logger.info("client disconnected; waiting for reconnect")


if __name__ == "__main__":
    main()
