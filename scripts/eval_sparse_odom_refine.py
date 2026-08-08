#!/usr/bin/env python
"""RC-5: sparse RAP odometry refinement on no-revisit stretches.

RC-4 localized KITTI-02's residual PGO APE (26.4 m) to the long stretches
without loop closures (frames 0-900, 2000-3400): there, odometry drift
accumulates unchecked. Odometry between-edge errors are temporally
CORRELATED (drift), while RAP registration errors are per-edge INDEPENDENT.
So replacing a sparse set of KISS-ICP between-edges with RAP-registered
ones should convert accumulating drift into sqrt(N)-growing noise.

Experiment (offline, reuses saved live products):
  Phase "sample" (GPU): RAP-register CONSECUTIVE keyframe pairs whose
    endpoints both lie in a no-revisit stretch, plus a control sample from
    the loop-covered region. Resumable JSON cache.
  Phase "analyze" (CPU):
    Q1: per-edge accuracy — RAP T_rel vs odom T_rel, both against the GT
        relative pose, on stretch vs control pairs.
    Q2: PGO APE when stretch between-edges are REPLACED by RAP edges
        (loop-style relfloor+huber noise), vs the 26.423 m baseline and
        odometry-only 31.640 m.
    Q3: stride ablation — refine every pair / every 2nd / every 4th pair.

Run (WSL2, conda env rap-inference):
  python scripts/eval_sparse_odom_refine.py                 # both phases
  python scripts/eval_sparse_odom_refine.py --phase analyze # CPU only
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # eval_pgo_robust

KITTI_SEQ_DIR = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset/sequences/02"
KITTI_ROOT = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset"
SAVE_DIR = os.path.join(REPO_ROOT, "output_ros2", "step3_kitti02")
ODOM_PATH = os.path.join(REPO_ROOT, "output_ros2", "live_poses_kitti02.txt")
CACHE_DEFAULT = os.path.join(REPO_ROOT, "output_odom_refine", "odom_edge_samples.json")

STRETCHES = [(0, 900), (2000, 3400)]   # no-revisit stretches (RC-4 finding)
CONTROL_REGION = (901, 1999)           # loop-covered region (control sample)
CONTROL_EVERY = 8                      # keep every Nth control pair


# ----------------------------------------------------------------------
def pair_lists(keyframes):
    """Consecutive keyframe pairs in the no-revisit stretches + control."""
    kf = sorted(keyframes)
    stretch_pairs, control_pairs = [], []
    for a, b in zip(kf[:-1], kf[1:]):
        in_stretch = any(s <= a < b <= e for s, e in STRETCHES)
        if in_stretch:
            stretch_pairs.append((a, b))
        elif CONTROL_REGION[0] <= a < b <= CONTROL_REGION[1]:
            control_pairs.append((a, b))
    control_pairs = control_pairs[::CONTROL_EVERY]
    return stretch_pairs, control_pairs


def load_scan(idx):
    path = os.path.join(KITTI_SEQ_DIR, "velodyne", f"{idx:06d}.bin")
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


# ----------------------------------------------------------------------
# Phase 1: GPU sampling
# ----------------------------------------------------------------------
def phase_sample(args):
    from rap_slam.rap_backend import RAPBackend

    keyframes = json.load(open(os.path.join(SAVE_DIR, "keyframes.json")))
    stretch_pairs, control_pairs = pair_lists(keyframes)
    pairs = ([{"kind": "stretch", "i": a, "j": b} for a, b in stretch_pairs] +
             [{"kind": "control", "i": a, "j": b} for a, b in control_pairs])

    cache = {}
    if os.path.exists(args.cache) and not args.resample:
        cache = {e["key"]: e for e in json.load(open(args.cache))["entries"]}
    print(f"pairs: {len(stretch_pairs)} stretch + {len(control_pairs)} "
          f"control = {len(pairs)}; {len(cache)} cached")

    rap = RAPBackend()  # defaults: rap_12 ckpt, voxel 0.4, 4000 pts, spinnet
    entries = list(cache.values())
    t0 = time.time()
    n_done = len(cache)
    for p in pairs:
        key = f"{p['kind']}|{p['i']}|{p['j']}"
        if key in cache:
            continue
        T_rel, diag = rap.register(load_scan(p["i"]), load_scan(p["j"]))
        entries.append({
            "key": key, "kind": p["kind"], "i": p["i"], "j": p["j"],
            "T_rel": np.asarray(T_rel).tolist(),
            "overlap": float(diag["overlap_ratio"]),
            "mean_dist": float(diag["mean_dist"]),
            "t_inference": float(diag["t_inference"]),
        })
        n_done += 1
        if n_done % 25 == 0 or n_done == len(pairs):
            el = time.time() - t0
            eta = el / max(n_done - len(cache), 1) * (len(pairs) - n_done)
            print(f"  [{n_done}/{len(pairs)}] {key} | elapsed {el/60:.1f} min"
                  f", ETA {eta/60:.1f} min", flush=True)
            _save(args.cache, entries)
    _save(args.cache, entries)
    print(f"sample phase done: {len(entries)} entries -> {args.cache}")


def _save(path, entries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"entries": entries}, f)
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# Phase 2: analysis
# ----------------------------------------------------------------------
def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def phase_analyze(args):
    import gtsam
    from gtsam import (BetweenFactorPose3, LevenbergMarquardtOptimizer,
                       NonlinearFactorGraph, Pose3, PriorFactorPose3, Values,
                       noiseModel)
    import eval_pgo_robust as epr
    from rap_slam import evaluate, real_dataset

    entries = json.load(open(args.cache))["entries"]
    odom_poses = epr.load_tum(ODOM_PATH)
    keyframes = sorted(json.load(open(os.path.join(SAVE_DIR, "keyframes.json"))))
    loops = json.load(open(os.path.join(SAVE_DIR, "accepted_loops.json")))
    gt = list(real_dataset.load_gt_poses(
        os.path.join(KITTI_ROOT, "poses/02.txt"),
        os.path.join(KITTI_ROOT, "sequences/02/calib.txt"),
        start=0, end=len(odom_poses) - 1, apply_calib=True))
    gt_rel = evaluate.align_to_first_pose(gt)
    print(f"frames {len(odom_poses)}, keyframes {len(keyframes)}, "
          f"loop edges {len(loops)}, RAP odom-edge samples {len(entries)}")

    # ------------------------------------------------------------------
    print("\n=== Q1: RAP edge vs odometry edge accuracy (vs GT relative) ===")
    for kind in ("stretch", "control"):
        es = [e for e in entries if e["kind"] == kind]
        rap_te, rap_re, odom_te, odom_re, ovs = [], [], [], [], []
        for e in es:
            i, j = e["i"], e["j"]
            T_gt = np.linalg.inv(gt_rel[i]) @ gt_rel[j]
            dG = np.linalg.inv(T_gt) @ np.asarray(e["T_rel"])
            rap_te.append(float(np.linalg.norm(dG[:3, 3])))
            rap_re.append(rot_angle_deg(dG[:3, :3]))
            T_od = np.linalg.inv(odom_poses[i]) @ odom_poses[j]
            dO = np.linalg.inv(T_gt) @ T_od
            odom_te.append(float(np.linalg.norm(dO[:3, 3])))
            odom_re.append(rot_angle_deg(dO[:3, :3]))
            ovs.append(e["overlap"])
        rap_te, odom_te = np.array(rap_te), np.array(odom_te)
        print(f"  [{kind:>7s}] {len(es)} pairs, overlap "
              f"{min(ovs):.2f}..{max(ovs):.2f}")
        print(f"           RAP  edge trans err: med {np.median(rap_te):.3f} m "
              f"p90 {np.quantile(rap_te, 0.9):.3f} max {rap_te.max():.3f} | "
              f"rot med {np.median(rap_re):.2f} deg")
        print(f"           odom edge trans err: med {np.median(odom_te):.3f} m "
              f"p90 {np.quantile(odom_te, 0.9):.3f} max {odom_te.max():.3f} | "
              f"rot med {np.median(odom_re):.2f} deg")

    # ------------------------------------------------------------------
    # Q2/Q3: PGO with stretch between-edges replaced by RAP edges
    # ------------------------------------------------------------------
    def run_pgo_refined(refine_map, loops_in):
        """Between-factor for keyframe pair (a,b) uses RAP T_rel when
        (a,b) in refine_map (loop-style relfloor noise + Huber), else the
        odometry relative with the standard odometry sigmas."""
        graph = NonlinearFactorGraph()
        initial = Values()
        kf_set = set(keyframes)
        for k in keyframes:
            initial.insert(k, Pose3(odom_poses[k]))
        graph.add(PriorFactorPose3(
            keyframes[0], Pose3(odom_poses[keyframes[0]]),
            noiseModel.Diagonal.Sigmas(np.array(epr.PRIOR_SIGMAS))))
        odom_noise = noiseModel.Diagonal.Sigmas(np.array(epr.ODOM_SIGMAS))
        for a, b in zip(keyframes[:-1], keyframes[1:]):
            e = refine_map.get((a, b))
            if e is None:
                graph.add(BetweenFactorPose3(
                    a, b, Pose3(np.linalg.inv(odom_poses[a]) @ odom_poses[b]),
                    odom_noise))
            else:
                s = float(np.clip(e["mean_dist"] / 1.0, 1.0, 3.0))
                rn = noiseModel.Robust.Create(
                    noiseModel.mEstimator.Huber(1.0),
                    noiseModel.Diagonal.Sigmas(
                        np.array(epr.LOOP_SIGMAS_BASE) * s))
                graph.add(BetweenFactorPose3(
                    a, b, Pose3(np.asarray(e["T_rel"])), rn))
        # loop edges: current default relfloor + Huber (RC-1)
        n_loops = 0
        for lc in loops_in:
            i, j = lc["i"], lc["j"]
            if i in kf_set and j in kf_set:
                s = float(np.clip(float(lc.get("mean_dist", 0.5)) / 1.0,
                                  1.0, 3.0))
                ln = noiseModel.Robust.Create(
                    noiseModel.mEstimator.Huber(1.0),
                    noiseModel.Diagonal.Sigmas(
                        np.array(epr.LOOP_SIGMAS_BASE) * s))
                graph.add(BetweenFactorPose3(
                    i, j, Pose3(np.asarray(lc["T_rel"])), ln))
                n_loops += 1
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        result = LevenbergMarquardtOptimizer(graph, initial, params).optimize()
        optimized = {k: result.atPose3(k).matrix() for k in keyframes}
        m = epr.full_traj_ape(odom_poses, keyframes, optimized, gt_rel)
        return m, n_loops

    m_odom = evaluate.compute_ape_rpe(
        gt_rel[:len(odom_poses)], evaluate.align_to_first_pose(odom_poses))
    print("\n=== Q2/Q3: PGO with RAP-refined stretch odometry edges ===")
    print(f"  odometry-only APE rmse = {m_odom['ape_rmse']:.3f} m")

    m_base, nl = run_pgo_refined({}, loops)
    print(f"  {'loops only (RC-1 default)':>38s}: APE rmse "
          f"{m_base['ape_rmse']:7.3f} m  (loops {nl})")

    stretch_entries = [e for e in entries if e["kind"] == "stretch"]
    results = {"baseline": m_base["ape_rmse"]}
    for stride in (1, 2, 4):
        rm = {(e["i"], e["j"]): e for e in stretch_entries[::stride]}
        m, _ = run_pgo_refined(rm, loops)
        results[f"stride{stride}"] = m["ape_rmse"]
        print(f"  {'RAP-refined stretch edges stride=' + str(stride):>38s}: "
              f"APE rmse {m['ape_rmse']:7.3f} m  ({len(rm)} refined edges)")
    # control: refine the CONTROL pairs instead (should do ~nothing if the
    # benefit is specific to the unconstrained stretches)
    rm_ctrl = {(e["i"], e["j"]): e for e in entries if e["kind"] == "control"}
    m_ctrl, _ = run_pgo_refined(rm_ctrl, loops)
    results["control_only"] = m_ctrl["ape_rmse"]
    print(f"  {'RAP-refined CONTROL edges (specificity)':>38s}: APE rmse "
          f"{m_ctrl['ape_rmse']:7.3f} m  ({len(rm_ctrl)} refined edges)")

    out = os.path.join(os.path.dirname(args.cache), "odom_refine_analysis.json")
    with open(out, "w") as f:
        json.dump({"odom_ape": m_odom["ape_rmse"], "pgo_variants": results,
                   "n_stretch_pairs": len(stretch_entries)}, f, indent=1)
    print(f"\nanalysis saved to {out}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sample", "analyze", "both"],
                    default="both")
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--resample", action="store_true")
    args = ap.parse_args()
    if args.phase in ("sample", "both"):
        phase_sample(args)
    if args.phase in ("analyze", "both"):
        phase_analyze(args)


if __name__ == "__main__":
    main()
