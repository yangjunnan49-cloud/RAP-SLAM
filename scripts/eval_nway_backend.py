#!/usr/bin/env python3
"""RAP N-way joint global registration backend — comparison against GTSAM PGO.

The roadmap's core research question: instead of verifying loop candidates
with independent 2-way RAP registrations and fusing them with GTSAM PGO,
jointly register a batch of loop-related keyframes in ONE RAP call (its
native multi-part capability) and use the cycle-consistent result as the
backend.

Compared on the POSS segment (APE RMSE vs GT, first-pose aligned):
  - odometry        : raw KISS-ICP
  - pairwise-PGO    : current pipeline (2-way RAP loop edges + GTSAM)
  - Nway-PGO        : same graph, loop-edge measurements taken from the
                      N-way joint registration (inv(T_i) @ T_j)
  - Nway-direct     : no optimization at all — batch keyframe poses from
                      the N-way common frame, anchored at the first batch
                      keyframe's odometry pose, propagated piecewise

Batches:
  B1 = keyframes appearing in accepted loops (POSS: 5)
  B2 = B1 plus one temporal keyframe neighbor on each side
  B3 = ALL keyframes (82 on POSS; fewer points per scan — global attention
       in PointCloudDiT is O(total_points^2))

Usage (WSL):
  python scripts/eval_nway_backend.py --batches B1,B2
  python scripts/eval_nway_backend.py --batches B3 --load-loops output_nway/accepted_loops.json
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rap_slam import evaluate, keyframes as kf_mod, loop_closure, odometry
from rap_slam import pose_graph, real_dataset
from rap_slam import scan_context as sc_mod

logger = logging.getLogger("eval_nway_backend")


def parse_args():
    p = argparse.ArgumentParser(description="RAP N-way backend vs GTSAM PGO")
    p.add_argument("--real-root", default="/mnt/e/githouse/RAP-SLAM/data/poss")
    p.add_argument("--real-seq", default="00")
    p.add_argument("--real-start", type=int, default=3850)
    p.add_argument("--real-end", type=int, default=5268)
    p.add_argument("--odom-poses", default="/mnt/e/githouse/RAP-SLAM/output_poss/odom_poses.txt")
    p.add_argument("--odom-max-range", type=float, default=80.0)
    p.add_argument("--kf-trans", type=float, default=6.0)
    p.add_argument("--kf-rot", type=float, default=12.0)
    p.add_argument("--loop-sep", type=int, default=200)
    p.add_argument("--loop-dist", type=float, default=15.0)
    p.add_argument("--sc-dist-thresh", type=float, default=0.25)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--min-overlap", type=float, default=0.7)
    p.add_argument("--rap-root", default="/mnt/e/githouse/RAP-main")
    p.add_argument("--rap-steps", type=int, default=5)
    p.add_argument("--rap-max-points", type=int, default=4000)
    p.add_argument("--rap-voxel", type=float, default=0.4)
    p.add_argument("--b3-points", type=int, default=1500,
                   help="per-scan point cap for the all-keyframe batch B3")
    p.add_argument("--batches", default="B1,B2", help="comma list: B1,B2,B3")
    p.add_argument("--load-loops", default=None,
                   help="JSON with accepted loops (skips pairwise RAP verification)")
    p.add_argument("--reuse-cache", action="store_true",
                   help="reuse cached batch_<name>.npz N-way results when present")
    p.add_argument("--output", default="/mnt/e/githouse/RAP-SLAM/output_nway")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _rot_err_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rel_error(T_rel_est: np.ndarray, T_i_gt: np.ndarray, T_j_gt: np.ndarray):
    """Translation/rotation error of an estimated relative pose vs GT."""
    T_gt = np.linalg.inv(T_i_gt) @ T_j_gt
    dG = np.linalg.inv(T_gt) @ T_rel_est
    return float(np.linalg.norm(dG[:3, 3])), _rot_err_deg(dG[:3, :3])


def umeyama_se3(src: np.ndarray, dst: np.ndarray):
    """Best-fit SE(3) (R, t) aligning src (N,3) onto dst (N,3) — Kabsch."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def ape_rmse(gt_rel, est_rel) -> float:
    return evaluate.compute_ape_rpe(gt_rel, est_rel)["ape_rmse"]


def build_batches(name: str, keyframes: List[int], loop_pairs) -> List[int]:
    kfs = list(keyframes)
    if name == "B1":
        batch = sorted({k for pair in loop_pairs for k in pair})
    elif name == "B2":
        core = {k for pair in loop_pairs for k in pair}
        pos = {kf: idx for idx, kf in enumerate(kfs)}
        keep = set()
        for kf in core:
            for d in (-1, 0, 1):
                if 0 <= pos[kf] + d < len(kfs):
                    keep.add(kfs[pos[kf] + d])
        batch = sorted(keep)
    elif name == "B3":
        batch = kfs
    else:
        raise ValueError(f"unknown batch {name}")
    return batch


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    t_start = time.time()

    # ------------------------------------------------ data + odometry + keyframes
    data = real_dataset.load_real_sequence(
        args.real_root, sequence_id=args.real_seq, start=args.real_start,
        end=args.real_end, min_range=1.5, stride=1, apply_calib=False)
    gt_poses, scans = data["gt_poses"], data["scans"]
    n = len(scans)
    if os.path.exists(args.odom_poses):
        from rap_slam import sequence as seq_mod
        odom_poses = list(seq_mod.load_kitti_poses(args.odom_poses))
        assert len(odom_poses) == n
        logger.info(f"Loaded cached odometry poses from {args.odom_poses}")
    else:
        odom_poses, backend = odometry.run_odometry(scans, max_range=args.odom_max_range)
    keyframes = kf_mod.select_keyframes(odom_poses, trans_thresh=args.kf_trans,
                                        rot_thresh_deg=args.kf_rot)

    gt_rel = evaluate.align_to_first_pose(list(gt_poses))
    odom_rel = evaluate.align_to_first_pose(odom_poses)
    m_odom = ape_rmse(gt_rel, odom_rel)
    logger.info(f"Odometry APE RMSE: {m_odom:.3f} m")

    # ------------------------------------------------ accepted loop constraints
    accepted = None
    if args.load_loops and os.path.exists(args.load_loops):
        with open(args.load_loops) as f:
            accepted = json.load(f)
        for lc in accepted:
            lc["T_rel"] = np.array(lc["T_rel"])
        logger.info(f"Loaded {len(accepted)} accepted loops from {args.load_loops}")

    rap = None
    if accepted is None:
        # Rebuild candidates (SC + proximity) and verify pairwise with RAP.
        from rap_slam.rap_backend import RAPBackend
        rap = RAPBackend(rap_root=args.rap_root, sampling_steps=args.rap_steps,
                         voxel_size=args.rap_voxel,
                         max_points_per_scan=args.rap_max_points, seed=args.seed)
        prox = loop_closure.find_candidates(
            odom_poses, keyframes, dist_thresh=args.loop_dist,
            min_separation=args.loop_sep, max_candidates=args.max_candidates)
        descriptors = sc_mod.build_descriptors(scans, keyframes)
        sc_pairs = sc_mod.find_candidates_sc(
            descriptors, keyframes, min_separation=args.loop_sep,
            dist_thresh=args.sc_dist_thresh, max_candidates=args.max_candidates)
        candidates = loop_closure.merge_candidates(prox, sc_pairs,
                                                   max_candidates=args.max_candidates)
        accepted = loop_closure.verify_candidates(
            candidates, scans, odom_poses, rap, min_overlap=args.min_overlap,
            gt_poses=gt_poses)
        loops_json = os.path.join(args.output, "accepted_loops.json")
        with open(loops_json, "w") as f:
            json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in lc.items()} for lc in accepted], f, indent=2)
        logger.info(f"Saved accepted loops to {loops_json}")

    loop_pairs = [(lc["i"], lc["j"]) for lc in accepted]
    logger.info(f"Loop pairs: {loop_pairs}")

    # ------------------------------------------------ baseline: pairwise PGO
    opt_kf = pose_graph.optimize_pose_graph(odom_poses, keyframes, accepted)
    opt_rel = evaluate.align_to_first_pose(
        pose_graph.propagate_to_full_trajectory(odom_poses, keyframes, opt_kf))
    m_pgo = ape_rmse(gt_rel, opt_rel)
    logger.info(f"pairwise-PGO APE RMSE: {m_pgo:.3f} m")

    results = {"odometry_ape": m_odom, "pairwise_pgo_ape": m_pgo,
               "loop_pairs": loop_pairs, "batches": {}}

    # ------------------------------------------------ N-way batches
    batch_names = [b.strip() for b in args.batches.split(",") if b.strip()]
    if batch_names and rap is None:
        need_rap = not args.reuse_cache or any(
            not os.path.exists(os.path.join(args.output, f"batch_{b}.npz"))
            for b in batch_names)
        if need_rap:
            from rap_slam.rap_backend import RAPBackend
            rap = RAPBackend(rap_root=args.rap_root, sampling_steps=args.rap_steps,
                             voxel_size=args.rap_voxel,
                             max_points_per_scan=args.rap_max_points, seed=args.seed)

    for name in batch_names:
        batch = build_batches(name, keyframes, loop_pairs)
        pps = args.b3_points if name == "B3" else args.rap_max_points
        logger.info("=" * 60)
        logger.info(f"Batch {name}: {len(batch)} keyframes, {pps} pts/scan: {batch}")
        cache = os.path.join(args.output, f"batch_{name}.npz")
        if args.reuse_cache and os.path.exists(cache):
            z = np.load(cache)
            T_parts = [z["T_parts"][i] for i in range(len(z["T_parts"]))]
            assert [int(k) for k in z["batch"]] == batch, "cached batch mismatch"
            diag = {"n_parts": len(batch), "total_points": -1,
                    "t_inference": -1, "t_preprocess": -1}
            logger.info(f"Loaded cached N-way result from {cache}")
        else:
            try:
                T_parts, diag = rap.register_multi([scans[k] for k in batch],
                                                   max_points_per_scan=pps)
            except Exception as e:
                logger.error(f"Batch {name} failed: {e}")
                results["batches"][name] = {"error": str(e)}
                continue
        idx_of = {kf: i for i, kf in enumerate(batch)}
        logger.info(f"N-way registration done: {diag['n_parts']} parts, "
                    f"{diag['total_points']} pts, infer {diag['t_inference']:.1f} s, "
                    f"preproc {diag['t_preprocess']:.1f} s")
        np.savez(os.path.join(args.output, f"batch_{name}.npz"),
                 T_parts=np.stack(T_parts), batch=np.array(batch))

        # --- oracle global-layout check: best-fit SE(3) to GT positions ---
        nway_pos = np.array([T_parts[idx_of[kf]][:3, 3] for kf in batch])
        gt_pos = np.array([gt_poses[kf][:3, 3] for kf in batch])
        R_fit, t_fit = umeyama_se3(nway_pos, gt_pos)
        resid = np.linalg.norm((R_fit @ nway_pos.T).T + t_fit - gt_pos, axis=1)
        layout = {"rmse": round(float(np.sqrt(np.mean(resid ** 2))), 3),
                  "median": round(float(np.median(resid)), 3),
                  "p90": round(float(np.quantile(resid, 0.9)), 3),
                  "max": round(float(resid.max()), 3),
                  "worst_kf": int(batch[int(resid.argmax())])}
        results["batches"][name] = {"diag": {k: v for k, v in diag.items()},
                                    "oracle_layout": layout}
        logger.info(f"  Oracle SE(3)-fit of N-way layout to GT: rmse={layout['rmse']} m "
                    f"median={layout['median']} p90={layout['p90']} "
                    f"max={layout['max']} (worst kf {layout['worst_kf']})")

        # --- intrinsic accuracy: relative pose of each loop pair vs GT ---
        intrinsic = []
        for lc in accepted:
            i, j = lc["i"], lc["j"]
            if i not in idx_of or j not in idx_of:
                continue
            T_rel_nway = np.linalg.inv(T_parts[idx_of[i]]) @ T_parts[idx_of[j]]
            te, re = rel_error(T_rel_nway, gt_poses[i], gt_poses[j])
            intrinsic.append({"pair": (i, j), "nway_trans_err": round(te, 3),
                              "nway_rot_err_deg": round(re, 2),
                              "pairwise_trans_err": (None if lc["gt_trans_err"] is None
                                                     else round(float(lc["gt_trans_err"]), 3))})
            logger.info(f"  loop ({i}, {j}): N-way vs GT dT={te:.3f} m dR={re:.2f} deg "
                        f"(pairwise was {lc['gt_trans_err']})")
        results["batches"][name]["intrinsic"] = intrinsic

        # --- Path A: PGO with N-way loop-edge measurements ---
        nway_constraints = []
        for lc in accepted:
            i, j = lc["i"], lc["j"]
            lc2 = dict(lc)
            if i in idx_of and j in idx_of:
                lc2["T_rel"] = np.linalg.inv(T_parts[idx_of[i]]) @ T_parts[idx_of[j]]
            nway_constraints.append(lc2)
        opt_kf_a = pose_graph.optimize_pose_graph(odom_poses, keyframes, nway_constraints)
        opt_rel_a = evaluate.align_to_first_pose(
            pose_graph.propagate_to_full_trajectory(odom_poses, keyframes, opt_kf_a))
        m_a = ape_rmse(gt_rel, opt_rel_a)
        results["batches"][name]["nway_pgo_ape"] = m_a
        logger.info(f"  Path A  Nway-PGO    APE RMSE: {m_a:.3f} m")

        # --- Path B: direct poses anchored at first batch keyframe ---
        anchor = batch[0]
        T_anchor_world = odom_poses[anchor] @ np.linalg.inv(T_parts[0])
        direct_poses = {kf: T_anchor_world @ T_parts[idx_of[kf]] for kf in batch}
        # Per-keyframe error in the first-pose frame (world-frame positions of
        # odometry/GT differ by hundreds of meters from the anchored frame).
        direct_kf_rel = {kf: np.linalg.inv(direct_poses[batch[0]]) @ direct_poses[kf]
                         for kf in batch}
        kf_err = np.array([np.linalg.norm(direct_kf_rel[kf][:3, 3] - gt_rel[kf][:3, 3])
                           for kf in batch])
        logger.info(f"  Path B per-keyframe position error: median={np.median(kf_err):.2f} m "
                    f"p90={np.quantile(kf_err, 0.9):.2f} max={kf_err.max():.2f} "
                    f"(worst kf {batch[int(kf_err.argmax())]})")
        direct_rel = evaluate.align_to_first_pose(
            pose_graph.propagate_to_full_trajectory(odom_poses, batch, direct_poses))
        m_b = ape_rmse(gt_rel, direct_rel)
        results["batches"][name]["nway_direct_ape"] = m_b
        results["batches"][name]["direct_kf_err"] = {
            "median": round(float(np.median(kf_err)), 3),
            "p90": round(float(np.quantile(kf_err, 0.9)), 3),
            "max": round(float(kf_err.max()), 3),
            "worst_kf": int(batch[int(kf_err.argmax())])}
        logger.info(f"  Path B  Nway-direct APE RMSE: {m_b:.3f} m")

    # ------------------------------------------------ summary
    results["total_runtime_s"] = round(time.time() - t_start, 1)
    out_json = os.path.join(args.output, "summary_nway.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n===== N-WAY BACKEND vs PGO (APE RMSE, m) =====")
    print(f"{'odometry':>14} {'pairwise-PGO':>14}", end="")
    for name in batch_names:
        print(f" | {name}: PGO / direct", end="")
    print()
    print(f"{m_odom:>14.3f} {m_pgo:>14.3f}", end="")
    for name in batch_names:
        b = results["batches"].get(name, {})
        if "error" in b:
            print(f" | {'FAILED':>16}", end="")
        else:
            print(f" | {b['nway_pgo_ape']:>7.3f} / {b['nway_direct_ape']:>7.3f}", end="")
    print(f"\nSummary written to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
