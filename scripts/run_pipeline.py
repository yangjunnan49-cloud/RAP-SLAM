#!/usr/bin/env python3
"""RAP-SLAM pipeline entry point.

Stages:
  1. Simulate a closed-loop scan sequence from a real aggregated map.
  2. KISS-ICP frame-to-frame odometry (explicit optional Open3D ICP fallback).
  3. Keyframe selection by displacement.
  4. Loop-closure candidates by odometry proximity + RAP verification.
  5. GTSAM pose-graph optimization.
  6. evo APE/RPE evaluation, trajectory plot, merged map.

Portable launcher: RAP-main/scripts/run.py slam (plan/check by default).
Direct entry: python /path/to/RAP-SLAM/scripts/run_pipeline.py --output /new/output
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rap_slam.cli import parse_args
from rap_slam.runtime_paths import validate_pipeline_paths

logger = logging.getLogger("run_pipeline")




def get_sequence(args):
    """Load cached sequence, generate a simulated one, or load a real segment."""
    import numpy as np
    from rap_slam import real_dataset, sequence, spires_dataset
    if getattr(args, "source", "sim") == "spires":
        logger.info(f"Loading Oxford Spires {args.spires_seq} from "
                    f"{args.spires_root} scans [{args.spires_start}:"
                    f"{args.spires_end}]")
        data = spires_dataset.load_spires_sequence(
            args.spires_root, args.spires_seq, start=args.spires_start,
            end=args.spires_end, min_range=1.5)
        return data["gt_poses"], data["scans"]
    if getattr(args, "source", "sim") == "real":
        logger.info(f"Loading real sequence {args.real_seq} from {args.real_root} "
                    f"frames [{args.real_start}:{args.real_end}] stride={args.real_stride}")
        kitti205 = bool(getattr(args, "apply_kitti_205", False))
        if kitti205 and not getattr(args, "real_apply_calib", False):
            # [meeting-20260921] the correction is now on by default on this
            # branch; auto-disable it for non-KITTI conventions (POSS etc.)
            # instead of raising, so existing POSS commands keep working.
            logger.info("auto-disabling --apply-kitti-205 (no --real-apply-calib)")
            kitti205 = False
        data = real_dataset.load_real_sequence(
            args.real_root, sequence_id=args.real_seq,
            start=args.real_start, end=args.real_end,
            min_range=getattr(args, "odom_min_range", 1.5),  # [meeting-20260921] was hard-coded 1.5
            stride=args.real_stride,
            apply_calib=getattr(args, "real_apply_calib", False),
            kitti205=kitti205)
        manifest = {"source_frame_ids": data["source_frame_ids"],
                    "timestamps": data["timestamps"], "scan_files": data["scan_files"],
                    "source": "real", "sequence": args.real_seq,
                    "pose_reference": "lidar"}
        with open(os.path.join(args.output, "source_frames.json"), "w") as f:
            json.dump(manifest, f, indent=2)
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
    errors = validate_pipeline_paths(args)
    if errors:
        raise SystemExit("Path preflight failed:\n" + "\n".join(errors))
    import numpy as np
    from rap_slam import (evaluate, keyframes as kf_mod, loop_closure, odometry,
                          pose_graph, scan_context as sc_mod, sequence)

    os.makedirs(args.output, exist_ok=False)
    run_id = str(uuid.uuid4())
    incomplete = os.path.join(args.output,"INCOMPLETE")
    with open(incomplete,"w") as f:
        f.write(run_id + "\n")
    with open(os.path.join(args.output,"run_manifest.json"),"w") as f:
        json.dump({"run_id":run_id,"config":vars(args)},f,indent=2)
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
        scans, max_range=getattr(args, "odom_max_range", 80.0),
        deskew=getattr(args, "deskew", False),
        allow_fallback=args.allow_odom_fallback,
        min_range=getattr(args, "odom_min_range", 1.5),  # [meeting-20260921]
        voxel_size=getattr(args, "odom_voxel", 0.8))  # [meeting-20260921]
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
    from rap_slam.verification_log import VerificationLog
    frame_path = os.path.join(args.output,"source_frames.json")
    if args.source == "real":
        with open(frame_path) as f:
            source_ids = json.load(f)["source_frame_ids"]
        dataset_id = "kitti" if args.real_apply_calib else "poss"
        sequence_id = args.real_seq
        frame_binding = "source_frames.json"
    else:
        source_ids = list(range(n))
        dataset_id, sequence_id = args.source, run_id
        frame_binding = "run_local_only"
    measurement_log = VerificationLog(os.path.join(args.output,"verification_measurements.jsonl"),
        run_id,dataset_id,sequence_id,source_ids,gt_poses,
        provenance={"frame_binding":frame_binding,"root_seed":args.seed,
                    "per_generation_seed":None,"seed_policy":"backend_shared_rng_in_recorded_order"})
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
                             ckpt_path=args.rap_ckpt,
                             spinnet_ckpt=args.spinnet_ckpt,
                             sampling_steps=args.rap_steps,
                             voxel_size=args.rap_voxel,
                             max_points_per_scan=args.rap_max_points,
                             feature_extraction_on=not args.no_rap_features,
                             seed=args.seed)
            def file_sha(path):
                digest = hashlib.sha256()
                with open(path,"rb") as f:
                    for block in iter(lambda:f.read(1024*1024),b""):
                        digest.update(block)
                return digest.hexdigest()
            provenance = {"checkpoint_sha256":file_sha(rap.ckpt_path),
                          "spinnet_sha256":file_sha(rap.spinnet_ckpt) if rap.feature_extraction_on else None,
                          "precision":"ieee_float32_matmul_scope",
                          "source_sha256":{name:file_sha(os.path.join(os.path.dirname(__file__),"..",name))
                            for name in ["rap_slam/rap_backend.py","rap_slam/verification_log.py",
                                         "rap_slam/measurement.py","rap_slam/loop_closure.py","scripts/run_pipeline.py"]}}
            with open(os.path.join(args.output,"inference_provenance.json"),"w") as f:
                json.dump(provenance,f,indent=2)
            measurement_log.provenance.update(provenance)
            gate_model = None
            if args.gate_mode == "conformal":
                from rap_slam.conformal_gate import ConformalGate
                gm_path = args.gate_model or os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(
                        __file__))), "data/conformal_gate/A1.json")
                gate_model = ConformalGate.load(gm_path)
                logger.info(f"conformal gate loaded from {gm_path} "
                            f"(Q={gate_model.Q:.4f}, alpha={gate_model.alpha}, "
                            f"features={gate_model.features})")
            accepted = loop_closure.verify_candidates(
                candidates, scans, odom_poses, rap,
                trans_consistency=args.trans_consistency,
                rot_consistency_deg=args.rot_consistency,
                min_overlap=args.min_overlap,
                max_mean_dist=args.max_mean_dist,
                multigen_k=args.multigen_k,
                min_overlap_min=args.min_overlap_min,
                max_disp_r=args.max_disp_r,
                gt_poses=gt_poses,
                gate_model=gate_model,measurement_log=measurement_log)
        else:
            logger.info("No loop candidates found")
    timings["loop_closure"] = time.time() - t0

    # Persist reusable loop-closure products for offline backend experiments
    # (eval_hier_nway.py / eval_submap_nway.py): keyframe list + accepted loop
    # edges with T_rel / overlap / mean_dist (same schema as the ROS 2 live
    # products in output_ros2/step3*/).
    with open(os.path.join(args.output, "keyframes.json"), "w") as f:
        json.dump([int(k) for k in keyframes], f)
    with open(os.path.join(args.output, "accepted_loops.json"), "w") as f:
        json.dump([{k: (v.tolist() if isinstance(v, np.ndarray)
                        else float(v) if isinstance(v, np.floating)
                        else int(v) if isinstance(v, np.integer)
                        else v)
                    for k, v in lc.items()} for lc in accepted], f)

    verification_counts = measurement_log.summary()
    with open(os.path.join(args.output,"verification_summary.json"),"w") as f:
        json.dump(verification_counts,f,indent=2)
    if verification_counts["failed"] or verification_counts["started"]:
        raise RuntimeError("Incomplete candidate verification; evidence saved, research run is not complete")

    # ------------------------------------------------------------ 5. PGO
    logger.info("=" * 70)
    logger.info("STAGE 5: pose-graph optimization")
    t0 = time.time()
    optimized_kf = pose_graph.optimize_pose_graph(
        odom_poses, keyframes, accepted,
        loop_weighting=args.loop_weighting,
        robust_kernel=None if args.robust_kernel == "none"
        else args.robust_kernel,
        huber_k=args.huber_k,
        dcs_phi=args.dcs_phi,
        relfloor_ref=args.relfloor_ref,
        relfloor_cap=args.relfloor_cap,
        loop_nms_width=args.loop_nms_width)
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
        "run_id": run_id,
        "verification_counts": verification_counts,
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
    os.remove(incomplete)
    logger.info(f"DONE in {total:.1f} s. Summary written to {args.output}/summary.json")
    print("\n===== RAP-SLAM SUMMARY =====")
    print(json.dumps({k: summary[k] for k in
                      ["n_scans", "n_keyframes", "odometry_backend",
                       "n_loop_candidates", "n_loops_accepted", "metrics",
                       "timings_s", "total_runtime_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
