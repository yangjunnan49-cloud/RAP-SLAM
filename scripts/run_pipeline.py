#!/usr/bin/env python3
"""RAP-SLAM pipeline entry point.

Stages:
  1. Simulate a closed-loop scan sequence from a real aggregated map.
  2. KISS-ICP frame-to-frame odometry (Open3D ICP fallback).
  3. Keyframe selection by displacement.
  4. Loop-closure candidates by odometry proximity + RAP verification.
  5. GTSAM pose-graph optimization.
  6. evo APE/RPE evaluation, trajectory plot, merged map.

Run inside WSL:
  python scripts/run_pipeline.py --output /mnt/e/githouse/RAP-SLAM/output
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rap_slam import (evaluate, keyframes as kf_mod, loop_closure, odometry,
                      pose_graph, real_dataset, scan_context as sc_mod,
                      sequence)

logger = logging.getLogger("run_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="RAP-SLAM offline SLAM pipeline")
    p.add_argument("--map", default="/mnt/e/githouse/RAP-main/demo_example_data/map_kitti/kitti00.ply")
    p.add_argument("--rap-root", default="/mnt/e/githouse/RAP-main")
    p.add_argument("--output", default="/mnt/e/githouse/RAP-SLAM/output")
    p.add_argument("--n-poses", type=int, default=100)
    p.add_argument("--radius", type=float, default=80.0)
    p.add_argument("--center", type=float, nargs=2, default=[250.0, -30.0])
    p.add_argument("--sensor-range", type=float, default=50.0)
    p.add_argument("--points-range", type=int, nargs=2, default=[2000, 4000])
    p.add_argument("--noise-sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kf-trans", type=float, default=6.0, help="keyframe translation threshold (m)")
    p.add_argument("--kf-rot", type=float, default=12.0, help="keyframe rotation threshold (deg)")
    p.add_argument("--loop-dist", type=float, default=15.0, help="loop candidate distance threshold (m)")
    p.add_argument("--loop-sep", type=int, default=5, help="loop candidate min scan separation")
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--loop-method", choices=["proximity", "sc", "both"], default="both",
                   help="loop-candidate retrieval: odometry proximity, "
                        "Scan Context global place recognition, or both merged")
    p.add_argument("--sc-dist-thresh", type=float, default=0.25,
                   help="Scan Context distance acceptance threshold")
    p.add_argument("--sc-max-radius", type=float, default=80.0)
    p.add_argument("--sc-rings", type=int, default=20)
    p.add_argument("--sc-sectors", type=int, default=60)
    p.add_argument("--sc-shortlist", type=int, default=10,
                   help="ring-key shortlist size per query")
    p.add_argument("--rap-steps", type=int, default=5, help="RAP inference sampling steps")
    p.add_argument("--rap-max-points", type=int, default=4000)
    p.add_argument("--rap-voxel", type=float, default=0.4)
    p.add_argument("--trans-consistency", type=float, default=5.0)
    p.add_argument("--rot-consistency", type=float, default=25.0)
    p.add_argument("--min-overlap", type=float, default=0.25)
    p.add_argument("--no-rap-features", action="store_true",
                   help="disable miniSpinNet features (zero features)")
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--skip-loop-closure", action="store_true")
    # --- real sequence (KITTI-format: KITTI odometry / POSS) ---
    p.add_argument("--source", choices=["sim", "real"], default="sim",
                   help="sim = simulated sequence from an aggregated map; "
                        "real = KITTI-format dataset segment")
    p.add_argument("--real-root", default="/mnt/e/githouse/RAP-SLAM/data/poss",
                   help="dataset root with dataset/poses and dataset/sequences")
    p.add_argument("--real-seq", default="00")
    p.add_argument("--real-start", type=int, default=3850)
    p.add_argument("--real-end", type=int, default=5268)
    p.add_argument("--real-stride", type=int, default=1)
    p.add_argument("--real-apply-calib", action="store_true",
                   help="apply Tr velo<->cam conversion (KITTI convention; "
                        "POSS poses are already in the LiDAR/world frame)")
    p.add_argument("--odom-max-range", type=float, default=80.0)
    return p.parse_args()


def get_sequence(args):
    """Load cached sequence, generate a simulated one, or load a real segment."""
    if getattr(args, "source", "sim") == "real":
        logger.info(f"Loading real sequence {args.real_seq} from {args.real_root} "
                    f"frames [{args.real_start}:{args.real_end}] stride={args.real_stride}")
        data = real_dataset.load_real_sequence(
            args.real_root, sequence_id=args.real_seq,
            start=args.real_start, end=args.real_end,
            min_range=1.5, stride=args.real_stride,
            apply_calib=getattr(args, "real_apply_calib", False))
        return data["gt_poses"], data["scans"]
    seq_dir = os.path.join(args.output, "sequence")
    gt_file = os.path.join(seq_dir, "gt_poses.txt")
    meta_file = os.path.join(seq_dir, "meta.json")
    meta = {"n_poses": args.n_poses, "radius": args.radius, "center": args.center,
            "sensor_range": args.sensor_range, "noise_sigma": args.noise_sigma,
            "points_range": args.points_range, "seed": args.seed}
    if (not args.force_regenerate and os.path.exists(gt_file)
            and os.path.exists(meta_file)):
        with open(meta_file) as f:
            cached = json.load(f)
        if cached == meta:
            logger.info(f"Loading cached sequence from {seq_dir}")
            gt_poses = sequence.load_kitti_poses(gt_file)
            scans = []
            import glob as _glob
            import open3d as o3d
            for path in sorted(_glob.glob(os.path.join(seq_dir, "scans", "scan_*.ply"))):
                scans.append(np.asarray(o3d.io.read_point_cloud(path).points,
                                        dtype=np.float32))
            assert len(scans) == len(gt_poses), "cached scan count mismatch"
            return gt_poses, scans
        logger.info("Cached sequence parameters differ; regenerating")
    data = sequence.generate_sequence(
        args.map, seq_dir, center=tuple(args.center), radius=args.radius,
        n_poses=args.n_poses, sensor_range=args.sensor_range,
        points_range=tuple(args.points_range),
        noise_sigma=args.noise_sigma, seed=args.seed)
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    return data["gt_poses"], data["scans"]


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    t_start = time.time()
    timings = {}

    # ------------------------------------------------------------ 1. sequence
    logger.info("=" * 70)
    logger.info("STAGE 1: sequence simulation")
    t0 = time.time()
    gt_poses, scans = get_sequence(args)
    n = len(scans)
    pts = [len(s) for s in scans]
    logger.info(f"Sequence: {n} scans, points/scan min={min(pts)} med={int(np.median(pts))} max={max(pts)}")
    timings["sequence"] = time.time() - t0

    # ------------------------------------------------------------ 2. odometry
    logger.info("=" * 70)
    logger.info("STAGE 2: odometry")
    t0 = time.time()
    odom_poses, backend_name = odometry.run_odometry(
        scans, max_range=getattr(args, "odom_max_range", 80.0))
    sequence.save_kitti_poses(np.stack(odom_poses),
                              os.path.join(args.output, "odom_poses.txt"))
    timings["odometry"] = time.time() - t0
    logger.info(f"Odometry done in {timings['odometry']:.1f} s ({backend_name})")

    # ------------------------------------------------------------ 3. keyframes
    logger.info("=" * 70)
    logger.info("STAGE 3: keyframe selection")
    keyframes = kf_mod.select_keyframes(odom_poses, trans_thresh=args.kf_trans,
                                        rot_thresh_deg=args.kf_rot)

    # ------------------------------------------------------------ 4. loop closure
    logger.info("=" * 70)
    logger.info("STAGE 4: loop closure (candidates + RAP verification)")
    accepted = []
    candidates = []
    t0 = time.time()
    if not args.skip_loop_closure:
        prox_pairs, sc_pairs = [], []
        if args.loop_method in ("proximity", "both"):
            prox_pairs = loop_closure.find_candidates(
                odom_poses, keyframes, dist_thresh=args.loop_dist,
                min_separation=args.loop_sep, max_candidates=args.max_candidates)
        if args.loop_method in ("sc", "both"):
            descriptors = sc_mod.build_descriptors(
                scans, keyframes, n_rings=args.sc_rings,
                n_sectors=args.sc_sectors, max_radius=args.sc_max_radius)
            sc_pairs = sc_mod.find_candidates_sc(
                descriptors, keyframes, min_separation=args.loop_sep,
                dist_thresh=args.sc_dist_thresh, shortlist=args.sc_shortlist,
                max_candidates=args.max_candidates)
        candidates = loop_closure.merge_candidates(
            prox_pairs, sc_pairs, max_candidates=args.max_candidates)
        if candidates:
            from rap_slam.rap_backend import RAPBackend
            rap = RAPBackend(rap_root=args.rap_root,
                             sampling_steps=args.rap_steps,
                             voxel_size=args.rap_voxel,
                             max_points_per_scan=args.rap_max_points,
                             feature_extraction_on=not args.no_rap_features,
                             seed=args.seed)
            accepted = loop_closure.verify_candidates(
                candidates, scans, odom_poses, rap,
                trans_consistency=args.trans_consistency,
                rot_consistency_deg=args.rot_consistency,
                min_overlap=args.min_overlap,
                gt_poses=gt_poses)
        else:
            logger.info("No loop candidates found")
    timings["loop_closure"] = time.time() - t0

    # ------------------------------------------------------------ 5. PGO
    logger.info("=" * 70)
    logger.info("STAGE 5: pose-graph optimization")
    t0 = time.time()
    optimized_kf = pose_graph.optimize_pose_graph(odom_poses, keyframes, accepted)
    opt_poses = pose_graph.propagate_to_full_trajectory(odom_poses, keyframes,
                                                        optimized_kf)
    sequence.save_kitti_poses(np.stack(opt_poses),
                              os.path.join(args.output, "optimized_poses.txt"))
    timings["pgo"] = time.time() - t0

    # ------------------------------------------------------------ 6. evaluation
    logger.info("=" * 70)
    logger.info("STAGE 6: evaluation")
    t0 = time.time()
    gt_rel = evaluate.align_to_first_pose(list(gt_poses))
    odom_rel = evaluate.align_to_first_pose(odom_poses)
    opt_rel = evaluate.align_to_first_pose(opt_poses)

    m_odom = evaluate.compute_ape_rpe(gt_rel, odom_rel)
    m_opt = evaluate.compute_ape_rpe(gt_rel, opt_rel)
    logger.info(f"Odometry  APE rmse={m_odom['ape_rmse']:.3f} m mean={m_odom['ape_mean']:.3f} "
                f"max={m_odom['ape_max']:.3f} | RPE rmse={m_odom['rpe_rmse']:.3f} m")
    logger.info(f"Optimized APE rmse={m_opt['ape_rmse']:.3f} m mean={m_opt['ape_mean']:.3f} "
                f"max={m_opt['ape_max']:.3f} | RPE rmse={m_opt['rpe_rmse']:.3f} m")

    evaluate.plot_trajectories(
        gt_rel, odom_rel, opt_rel, accepted, keyframes,
        os.path.join(args.output, "trajectory_plot.png"),
        title="RAP-SLAM: closed-loop trajectory (first-pose frame)")
    map_stride = 1 if n <= 200 else 3
    evaluate.merge_map(scans, opt_rel,
                       os.path.join(args.output, "merged_map_optimized.ply"),
                       voxel_size=0.3, stride=map_stride)
    evaluate.merge_map(scans, odom_rel,
                       os.path.join(args.output, "merged_map_odometry.ply"),
                       voxel_size=0.3, stride=map_stride)
    timings["evaluation"] = time.time() - t0

    # ------------------------------------------------------------ summary
    total = time.time() - t_start
    summary = {
        "config": {k: v for k, v in vars(args).items()},
        "n_scans": n,
        "n_keyframes": len(keyframes),
        "odometry_backend": backend_name,
        "n_loop_candidates": len(candidates),
        "loop_method": args.loop_method,
        "n_loops_accepted": len(accepted),
        "accepted_loops": [
            {"i": int(lc["i"]), "j": int(lc["j"]),
             "overlap": round(float(lc["overlap"]), 3),
             "gt_trans_err": (None if lc["gt_trans_err"] is None
                              else round(float(lc["gt_trans_err"]), 3)),
             "gt_rot_err_deg": (None if lc["gt_rot_err_deg"] is None
                                else round(float(lc["gt_rot_err_deg"]), 2))}
            for lc in accepted
        ],
        "metrics": {"odometry": m_odom, "optimized": m_opt},
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "total_runtime_s": round(total, 2),
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"DONE in {total:.1f} s. Summary written to {args.output}/summary.json")
    print("\n===== RAP-SLAM SUMMARY =====")
    print(json.dumps({k: summary[k] for k in
                      ["n_scans", "n_keyframes", "odometry_backend",
                       "n_loop_candidates", "n_loops_accepted", "metrics",
                       "timings_s", "total_runtime_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
