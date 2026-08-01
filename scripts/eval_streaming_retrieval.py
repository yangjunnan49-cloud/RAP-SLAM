#!/usr/bin/env python3
"""Offline evaluation of STREAMING loop-candidate retrieval policies.

Replays the live loop_closure_node proposal logic keyframe-by-keyframe
(same keyframe scans, same odometry poses, same arrival order) and
compares retrieval policies without needing a GPU or ROS:

  * "old" — ring-key shortlist + FIFO queue + hard total verification
    budget (the live node up to 2026-07-23, which missed ALL true loops
    on KITTI 02: ring-key prefilter dropped them and the budget was
    exhausted by false SC hits)
  * "new" — full-scan vectorized SC matching (sc_distances_batch) +
    best-first priority queue, no permanent budget (2026-07-31 rework)

A simple event-driven worker model stands in for RAP verification: the
worker processes one candidate at a time (--t-verify seconds each),
always taking the best-scoring pending candidate. Scans arrive at a
fixed rate (--hz). Reports per policy: proposed / verified counts,
precision / recall against GT loop pairs, WHEN each true pair would be
verified (as % of sequence time), verifications spent on false pairs,
and peak queue depth.

No GPU / ROS needed. Run inside WSL:

  python scripts/eval_streaming_retrieval.py --preset kitti02
  python scripts/eval_streaming_retrieval.py --preset poss
"""

import argparse
import heapq
import json
import logging
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rap_slam import scan_context as sc_mod  # noqa: E402
from rap_slam.cluster_gate import ClusterGate  # noqa: E402
from rap_slam.real_dataset import load_gt_poses  # noqa: E402

logger = logging.getLogger("eval_streaming_retrieval")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = {
    "kitti02": dict(
        real_root="/mnt/e/githouse/Datasets/data_odometry_velodyne",
        seq="02", start=0, end=None, apply_calib=True,
        odom_poses=os.path.join(REPO, "output_ros2/live_poses_kitti02.txt"),
        odom_format="tum",
        keyframes_json=os.path.join(REPO, "output_ros2/step3_kitti02/keyframes.json"),
    ),
    "poss": dict(
        real_root="/mnt/e/githouse/RAP-SLAM/data/poss",
        seq="00", start=3850, end=5268, apply_calib=False,
        odom_poses=os.path.join(REPO, "output_poss/odom_poses.txt"),
        odom_format="kitti",
        keyframes_json=os.path.join(REPO, "output_ros2/step3/keyframes.json"),
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Streaming loop-retrieval policy evaluation")
    p.add_argument("--preset", choices=sorted(PRESETS), default="kitti02")
    p.add_argument("--real-root", default=None)
    p.add_argument("--seq", default=None)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--apply-calib", action="store_true", default=None)
    p.add_argument("--odom-poses", default=None)
    p.add_argument("--odom-format", choices=["tum", "kitti"], default=None)
    p.add_argument("--keyframes-json", default=None,
                   help='path to a saved keyframes.json, or "auto" to '
                        'recompute streaming keyframes from the odom poses '
                        '(same rule as loop_closure_node)')
    # Policy parameters (live node defaults)
    p.add_argument("--loop-sep", type=int, default=200)
    p.add_argument("--loop-dist", type=float, default=15.0)
    p.add_argument("--sc-thresh", type=float, default=0.25)
    p.add_argument("--sc-shortlist", type=int, default=10)
    p.add_argument("--sc-max-radius", type=float, default=80.0)
    p.add_argument("--max-per-kf", type=int, default=3)
    p.add_argument("--max-total", type=int, default=16, help="old policy budget")
    # New policy: cluster-persistence gate
    p.add_argument("--sc-strong", type=float, default=0.12,
                   help="SC score below which candidates skip the cluster gate")
    p.add_argument("--min-cluster", type=int, default=2)
    p.add_argument("--max-per-cluster", type=int, default=2)
    p.add_argument("--cluster-w", type=int, default=50)
    p.add_argument("--cluster-spacing", type=int, default=25)
    p.add_argument("--kf-trans", type=float, default=6.0)
    p.add_argument("--kf-rot", type=float, default=12.0)
    # GT + simulation
    p.add_argument("--gt-loop-dist", type=float, default=3.0)
    p.add_argument("--hz", type=float, default=12.0, help="scan arrival rate")
    p.add_argument("--t-verify", type=float, default=3.0, help="seconds per RAP verification")
    p.add_argument("--sweep", action="store_true",
                   help="also sweep sc_thresh for the new policy")
    p.add_argument("--dump-proposals", default=None,
                   help="write new-policy proposals as JSON lines to this path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
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


def load_tum_poses(path: str) -> List[np.ndarray]:
    poses = []
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) != 8:
                continue
            _, x, y, z, qx, qy, qz, qw = map(float, v)
            T = np.eye(4)
            T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
            T[:3, 3] = [x, y, z]
            poses.append(T)
    return poses


def load_kitti_poses(path: str) -> List[np.ndarray]:
    poses = []
    with open(path) as f:
        for line in f:
            v = [float(x) for x in line.split()]
            if len(v) != 12:
                continue
            T = np.eye(4)
            T[:3, :] = np.array(v).reshape(3, 4)
            poses.append(T)
    return poses


def load_keyframe_scans(velodyne_dir: str, frame_numbers: List[int]
                        ) -> Dict[int, np.ndarray]:
    """Load only the given frames (raw xyz, no range filter — the live
    player publishes the raw .bin, so the node's SC sees raw points)."""
    scans = {}
    t0 = time.time()
    for f in frame_numbers:
        path = os.path.join(velodyne_dir, f"{f:06d}.bin")
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
        scans[f] = pts.astype(np.float32)
    logger.info(f"Loaded {len(scans)} keyframe scans in {time.time() - t0:.1f} s")
    return scans


def streaming_keyframes(poses: List[np.ndarray], trans_thresh: float,
                        rot_thresh_deg: float) -> List[int]:
    """Recompute the live node's keyframe list from poses (identical
    streaming rule: accumulate per-step translation and absolute yaw
    change, emit a keyframe when either exceeds its threshold)."""
    kfs = [0]
    acc_t, acc_r = 0.0, 0.0
    prev_yaw = float(np.arctan2(poses[0][1, 0], poses[0][0, 0]))
    for idx in range(1, len(poses)):
        acc_t += float(np.linalg.norm(poses[idx][:3, 3] - poses[idx - 1][:3, 3]))
        yaw = float(np.arctan2(poses[idx][1, 0], poses[idx][0, 0]))
        dyaw = (yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi
        prev_yaw = yaw
        acc_r += abs(dyaw)
        if acc_t > trans_thresh or acc_r > np.deg2rad(rot_thresh_deg):
            kfs.append(idx)
            acc_t, acc_r = 0.0, 0.0
    return kfs


def gt_loop_pairs(gt_poses: np.ndarray, keyframes: List[int],
                  min_sep: int, dist: float) -> Set[Tuple[int, int]]:
    pairs = set()
    pos = {kf: gt_poses[kf][:3, 3] for kf in keyframes}
    for ii in range(len(keyframes)):
        for jj in range(ii + 1, len(keyframes)):
            i, j = keyframes[ii], keyframes[jj]
            if j - i < min_sep:
                continue
            if np.linalg.norm(pos[i] - pos[j]) < dist:
                pairs.add((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Streaming policy simulation
# ---------------------------------------------------------------------------
class OldPolicy:
    """Exact replica of the pre-2026-07-31 live node proposal logic:
    proximity + ring-key-shortlist SC, per-kf cap, hard total budget,
    FIFO verification order."""

    def __init__(self, args):
        self.args = args
        self.proposed: Set[Tuple[int, int]] = set()
        self.pending: List[Tuple[float, Tuple[int, int]]] = []  # FIFO (push_t, pair)
        self.n_started = 0

    def propose(self, j: int, t: float, keyframes, poses, desc):
        a = self.args
        earlier = [i for i in keyframes if i < j and j - i >= a.loop_sep]
        prox = []
        for i in earlier:
            d = float(np.linalg.norm(poses[j][:3, 3] - poses[i][:3, 3]))
            if d < a.loop_dist:
                prox.append((i, d))
        sc_hits = []
        rk_j = desc[j]["ring_key"]
        shortlist = sorted(
            ((i, float(np.linalg.norm(rk_j - desc[i]["ring_key"]))) for i in earlier),
            key=lambda x: x[1])[:a.sc_shortlist]
        for i, _ in shortlist:
            d, _ = sc_mod.sc_distance(desc[i]["sc"], desc[j]["sc"])
            if d < a.sc_thresh:
                sc_hits.append((i, d))
        added = 0
        for i, d in sorted(prox, key=lambda x: x[1]) + sorted(sc_hits, key=lambda x: x[1]):
            if added >= a.max_per_kf:
                break
            if self.n_started + len(self.pending) >= a.max_total:
                break
            key = (min(i, j), max(i, j))
            if key in self.proposed:
                continue
            self.proposed.add(key)
            self.pending.append((t, key))
            added += 1

    def peek_next(self):
        return self.pending[0] if self.pending else None

    def pop_next(self):
        self.n_started += 1
        return self.pending.pop(0)[1]


class NewPolicy:
    """2026-07-31 rework: full-scan vectorized SC (no ring-key prefilter),
    cluster-persistence gate (ClusterGate), best-first priority queue,
    no permanent verification budget."""

    def __init__(self, args):
        self.args = args
        self.proposed: Set[Tuple[int, int]] = set()  # entered the gate
        self.queued: Set[Tuple[int, int]] = set()    # sent to the worker
        self.pending: List[Tuple[Tuple[float, float, int], Tuple[float, Tuple[int, int]]]] = []
        self._counter = 0
        self._db_indices: List[int] = []
        self._db_scs: List[np.ndarray] = []
        self.proposal_log: List[dict] = []  # every QUEUED pair w/ score
        self.gate = ClusterGate(window=args.cluster_w,
                                min_cluster=args.min_cluster,
                                max_per_cluster=args.max_per_cluster,
                                spacing=args.cluster_spacing,
                                strong_thresh=args.sc_strong)

    def propose(self, j: int, t: float, keyframes, poses, desc):
        a = self.args
        earlier = [i for i in self._db_indices if j - i >= a.loop_sep]
        cand = []  # (tier, score, i): tier 0 = SC, tier 1 = proximity
        if earlier:
            k = [self._db_indices.index(i) for i in earlier]
            db = np.stack([self._db_scs[kk] for kk in k])
            dists, _ = sc_mod.sc_distances_batch(db, desc[j]["sc"])
            for i, d in zip(earlier, dists):
                if d < a.sc_thresh:
                    cand.append((0, float(d), i))
        for i in earlier:
            d = float(np.linalg.norm(poses[j][:3, 3] - poses[i][:3, 3]))
            if d < a.loop_dist:
                cand.append((1, d / a.loop_dist, i))
        cand.sort()
        added = 0
        for tier, score, i in cand:
            if added >= a.max_per_kf:
                break
            key = (min(i, j), max(i, j))
            if key in self.proposed:
                continue
            self.proposed.add(key)
            added += 1
            for qi, qj, qtier, qscore in self.gate.add(key[0], key[1], tier, score):
                qkey = (qi, qj)
                if qkey in self.queued:
                    continue
                self.queued.add(qkey)
                self.proposal_log.append({"i": qi, "j": qj,
                                          "tier": qtier, "score": round(qscore, 4)})
                heapq.heappush(self.pending,
                               ((qtier, qscore, self._counter), (t, qkey)))
                self._counter += 1
        # Register j in the SC database AFTER proposing (matches live order).
        self._db_indices.append(j)
        self._db_scs.append(desc[j]["sc"])

    def peek_next(self):
        return self.pending[0][1] if self.pending else None

    def pop_next(self):
        return heapq.heappop(self.pending)[1][1]


def simulate(policy, keyframes, poses, desc, hz, t_verify):
    """Event-driven worker simulation. Returns (verified_at, stats):
    verified_at maps pair -> simulation time (s) of verification.

    The worker verifies one candidate at a time (t_verify each), always
    taking the policy's next pending candidate; a candidate can only start
    after it was pushed."""
    verified_at: Dict[Tuple[int, int], float] = {}
    worker_free_at = 0.0
    max_depth = 0

    def advance(until: float):
        nonlocal worker_free_at
        while True:
            item = policy.peek_next()
            if item is None:
                break
            push_t, pair = item
            start = max(worker_free_at, push_t)
            finish = start + t_verify
            if finish > until:
                break
            done = policy.pop_next()
            assert done == pair
            verified_at[pair] = finish
            worker_free_at = finish

    for j in keyframes:
        advance(j / hz)
        policy.propose(j, j / hz, keyframes, poses, desc)
        max_depth = max(max_depth, len(policy.pending))
    advance(float("inf"))  # drain after the sequence ends
    stats = {"max_depth": max_depth,
             "n_proposed": len(policy.proposed),
             "n_queued": len(getattr(policy, "queued", policy.proposed)),
             "n_verified": len(verified_at)}
    return verified_at, stats


def prf(proposed: Set[Tuple[int, int]], truth: Set[Tuple[int, int]]):
    tp = len(proposed & truth)
    prec = tp / len(proposed) if proposed else 0.0
    rec = tp / len(truth) if truth else 0.0
    return tp, prec, rec


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    pre = PRESETS[args.preset]
    real_root = args.real_root or pre["real_root"]
    seq = args.seq or pre["seq"]
    start = pre["start"] if args.start is None else args.start
    end = pre["end"] if args.end is None else args.end
    apply_calib = pre["apply_calib"] if args.apply_calib is None else args.apply_calib
    odom_path = args.odom_poses or pre["odom_poses"]
    odom_format = args.odom_format or pre["odom_format"]
    kf_json = args.keyframes_json or pre["keyframes_json"]

    poses_file = os.path.join(real_root, "dataset", "poses", f"{seq}.txt")
    calib_file = os.path.join(real_root, "dataset", "sequences", seq, "calib.txt")
    velodyne_dir = os.path.join(real_root, "dataset", "sequences", seq, "velodyne")
    gt_poses = load_gt_poses(poses_file, calib_file, start=start, end=end,
                             apply_calib=apply_calib)
    n = len(gt_poses)

    odom_poses = (load_tum_poses(odom_path) if odom_format == "tum"
                  else load_kitti_poses(odom_path))
    assert len(odom_poses) == n, f"odom pose count {len(odom_poses)} != {n}"
    logger.info(f"{n} poses (GT + cached odometry from {odom_path})")

    if kf_json == "auto":
        keyframes = streaming_keyframes(odom_poses, args.kf_trans, args.kf_rot)
        logger.info(f"{len(keyframes)} keyframes recomputed from odom poses "
                    f"(trans>{args.kf_trans} m, rot>{args.kf_rot} deg)")
    else:
        with open(kf_json) as f:
            keyframes = json.load(f)
        logger.info(f"{len(keyframes)} keyframes from {kf_json}")
    assert max(keyframes) < n, "keyframe index beyond poses"

    scans = load_keyframe_scans(velodyne_dir, [start + k for k in keyframes])
    t0 = time.time()
    desc = {}
    for k in keyframes:
        sc = sc_mod.make_scan_context(scans[start + k], max_radius=args.sc_max_radius)
        desc[k] = {"sc": sc, "ring_key": sc_mod.make_ring_key(sc)}
    logger.info(f"Descriptors built in {time.time() - t0:.1f} s")

    # Sanity: batch matcher must exactly reproduce the scalar matcher.
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.choice(keyframes, 2, replace=False)
        d_ref, s_ref = sc_mod.sc_distance(desc[a]["sc"], desc[b]["sc"])
        d_b, s_b = sc_mod.sc_distances_batch(desc[a]["sc"][None], desc[b]["sc"])
        assert abs(d_b[0] - d_ref) < 1e-6 and s_b[0] == s_ref, "batch mismatch"
    logger.info("sc_distances_batch matches sc_distance on 20 random pairs (1e-6)")

    truth = gt_loop_pairs(gt_poses, keyframes, args.loop_sep, args.gt_loop_dist)
    print(f"\n=== GT: {len(keyframes)} keyframes, {len(truth)} true loop pairs "
          f"(dist < {args.gt_loop_dist} m, sep >= {args.loop_sep}) ===")
    print(f"true pairs: {sorted(truth)}")

    # ---------------- diagnostics: SC separability of the true pairs ------
    print("\n=== True-pair SC diagnostics (full-scan distances at arrival of j) ===")
    print(f"{'pair':>12} {'gt_d':>5} {'sc_d':>6} {'rk_rank':>8} {'sc_rank':>8}")
    for (i, j) in sorted(truth, key=lambda p: p[1]):
        earlier = [k for k in keyframes if k < j and j - k >= args.loop_sep]
        db = np.stack([desc[k]["sc"] for k in earlier])
        dists, _ = sc_mod.sc_distances_batch(db, desc[j]["sc"])
        k_pos = earlier.index(i)
        sc_d = float(dists[k_pos])
        sc_rank = int((dists < sc_d).sum()) + 1
        rk_j = desc[j]["ring_key"]
        rk_sorted = sorted(float(np.linalg.norm(rk_j - desc[k]["ring_key"]))
                           for k in earlier)
        rk_rank = rk_sorted.index(
            float(np.linalg.norm(rk_j - desc[i]["ring_key"]))) + 1
        gt_d = float(np.linalg.norm(gt_poses[i][:3, 3] - gt_poses[j][:3, 3]))
        print(f"{str((i, j)):>12} {gt_d:>5.2f} {sc_d:>6.3f} {rk_rank:>8d} {sc_rank:>8d}")

    # ---------------- run both policies ------------------------------------
    seq_time = n / args.hz
    for name, cls in (("old", OldPolicy), ("new", NewPolicy)):
        policy = cls(args)
        t0 = time.time()
        verified_at, stats = simulate(policy, keyframes, odom_poses, desc,
                                      args.hz, args.t_verify)
        wall = time.time() - t0
        proposed = policy.proposed
        queued = getattr(policy, "queued", policy.proposed)
        verified = set(verified_at)
        tp_p, prec_p, rec_p = prf(proposed, truth)
        tp_q, prec_q, rec_q = prf(queued, truth)
        tp_v, prec_v, rec_v = prf(verified, truth)
        false_v = len(verified - truth)
        gpu_hours = len(verified) * args.t_verify / 3600.0
        print(f"\n=== policy={name} (sim wall {wall:.1f} s) ===")
        print(f"proposed={stats['n_proposed']} queued={stats['n_queued']} "
              f"verified={stats['n_verified']} max_queue_depth={stats['max_depth']}")
        print(f"proposal: tp={tp_p}/{len(truth)} prec={prec_p:.2f} rec={rec_p:.2f}")
        print(f"queued:   tp={tp_q}/{len(truth)} prec={prec_q:.2f} rec={rec_q:.2f}")
        print(f"verified: tp={tp_v}/{len(truth)} prec={prec_v:.2f} rec={rec_v:.2f} "
              f"false_verifications={false_v} (GPU {gpu_hours * 60:.1f} min)")
        print("true-pair verification time (% of sequence "
              f"{seq_time / 60:.1f} min):")
        for p in sorted(truth, key=lambda p: p[1]):
            if p in verified_at:
                print(f"  {p}: verified at {verified_at[p] / 60:.1f} min "
                      f"({100 * verified_at[p] / seq_time:.0f}% of seq, "
                      f"pair arrives at {100 * p[1] / args.hz / seq_time:.0f}%)")
            elif p in queued:
                print(f"  {p}: queued but never verified (queue backlog)")
            elif p in proposed:
                print(f"  {p}: proposed but GATED (cluster never promoted)")
            else:
                print(f"  {p}: NEVER PROPOSED")
        if name == "new" and args.dump_proposals:
            with open(args.dump_proposals, "w") as f:
                for rec in policy.proposal_log:
                    rec = dict(rec)
                    rec["true"] = (rec["i"], rec["j"]) in truth
                    f.write(json.dumps(rec) + "\n")
            logger.info(f"dumped {len(policy.proposal_log)} proposals to "
                        f"{args.dump_proposals}")

    # ---------------- threshold sweep for the new policy -------------------
    if args.sweep:
        print("\n=== new-policy sc_thresh sweep ===")
        print(f"{'thresh':>7} {'prop':>5} {'ver':>5} {'tp_v':>5} {'rec':>5} "
              f"{'false_v':>8} {'gpu_min':>8}")
        for thr in (0.15, 0.18, 0.20, 0.22, 0.25, 0.30):
            args.sc_thresh = thr
            policy = NewPolicy(args)
            verified_at, stats = simulate(policy, keyframes, odom_poses, desc,
                                          args.hz, args.t_verify)
            verified = set(verified_at)
            tp_v, _, rec_v = prf(verified, truth)
            false_v = len(verified - truth)
            print(f"{thr:>7.2f} {stats['n_proposed']:>5} {stats['n_verified']:>5} "
                  f"{tp_v:>5} {rec_v:>5.2f} {false_v:>8} "
                  f"{len(verified) * args.t_verify / 60:>8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
