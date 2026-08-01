"""Offline PGO robustness experiment: RAP-quality-adaptive weighting x robust kernels.

Reuses the saved live KITTI-02 products — no GPU / ROS / point clouds needed:

  output_ros2/live_poses_kitti02.txt              live odometry, TUM format
                                                  (row index == scan index)
  output_ros2/step3_kitti02/keyframes.json        812 keyframe scan indices
  output_ros2/step3_kitti02/accepted_loops.json   182 RAP-verified loop edges
                                                  with per-edge RAP diagnostics
                                                  (overlap, mean_dist)

Questions answered:
  1. Do RAP's per-edge diagnostics (overlap, mean_dist) actually correlate
     with the edge's true relative-pose error (computable here from GT)?
     This is the premise of RAP-quality-adaptive information weighting.
  2. Which loop-edge noise model minimizes PGO APE on the real 182-edge
     dataset: fixed sigmas (live baseline, 27.387 m), RAP-adaptive sigmas,
     Huber kernel, DCS, or adaptive x kernel combinations?
  3. Noise sweep: when a fraction of loop edges is corrupted with synthetic
     translation noise, which scheme degrades the least?

Run (WSL2, conda env rap-inference):
  python scripts/eval_pgo_robust.py
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rap_slam import evaluate, real_dataset  # noqa: E402

# Same sigmas as rap_slam/pose_graph.py (order: rx, ry, rz, tx, ty, tz).
ODOM_SIGMAS = (0.05, 0.05, 0.05, 0.30, 0.30, 0.30)
PRIOR_SIGMAS = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
LOOP_SIGMAS_BASE = (0.03, 0.03, 0.03, 0.20, 0.20, 0.20)


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------
def load_tum(path):
    """Load TUM-format poses (timestamp tx ty tz qx qy qz qw) -> list of 4x4."""
    rows = np.loadtxt(path)
    poses = []
    for r in rows:
        t = r[1:4]
        qx, qy, qz, qw = r[4:8]
        n_ = qx * qx + qy * qy + qz * qz + qw * qw
        s = 2.0 / n_
        xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
        xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
        wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
        T = np.eye(4)
        T[:3, :3] = [[1 - (yy + zz), xy - wz, xz + wy],
                     [xy + wz, 1 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1 - (xx + yy)]]
        T[:3, 3] = t
        poses.append(T)
    return poses


def t_rel_error(T_a, T_b):
    """Translation (m) and rotation (deg) magnitude of a relative transform."""
    dt = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    dR = T_a[:3, :3].T @ T_b[:3, :3]
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return dt, float(np.degrees(np.arccos(cos)))


# ----------------------------------------------------------------------
# PGO with a configurable loop-edge noise model
# ----------------------------------------------------------------------
def run_pgo(odom_poses, keyframes, loops, loop_noise_fn):
    """Mirror of rap_slam/pose_graph.optimize_pose_graph with a pluggable
    per-edge noise model. loop_noise_fn(lc) -> gtsam noise model."""
    import gtsam
    from gtsam import (BetweenFactorPose3, LevenbergMarquardtOptimizer,
                       NonlinearFactorGraph, Pose3, PriorFactorPose3, Values,
                       noiseModel)

    graph = NonlinearFactorGraph()
    initial = Values()
    kf_set = set(keyframes)
    for k in keyframes:
        initial.insert(k, Pose3(odom_poses[k]))

    prior_noise = noiseModel.Diagonal.Sigmas(np.array(PRIOR_SIGMAS))
    graph.add(PriorFactorPose3(keyframes[0], Pose3(odom_poses[keyframes[0]]),
                               prior_noise))

    odom_noise = noiseModel.Diagonal.Sigmas(np.array(ODOM_SIGMAS))
    for a, b in zip(keyframes[:-1], keyframes[1:]):
        graph.add(BetweenFactorPose3(a, b, Pose3(
            np.linalg.inv(odom_poses[a]) @ odom_poses[b]), odom_noise))

    n_edges = 0
    for lc in loops:
        i, j = lc["i"], lc["j"]
        if i in kf_set and j in kf_set:
            graph.add(BetweenFactorPose3(i, j, Pose3(np.asarray(lc["T_rel"])),
                                         loop_noise_fn(lc)))
            n_edges += 1

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    optimizer = LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    optimized = {k: result.atPose3(k).matrix() for k in keyframes}
    return optimized, n_edges


def full_traj_ape(odom_poses, keyframes, optimized, gt_rel):
    from rap_slam.pose_graph import propagate_to_full_trajectory
    opt_poses = propagate_to_full_trajectory(odom_poses, keyframes, optimized)
    m = evaluate.compute_ape_rpe(gt_rel[:len(opt_poses)],
                                 evaluate.align_to_first_pose(opt_poses))
    return m


# ----------------------------------------------------------------------
# Noise-model factories
# ----------------------------------------------------------------------
def make_noise_models():
    """name -> loop_noise_fn(lc). Built lazily so gtsam import stays local."""
    import gtsam
    from gtsam import noiseModel

    def fixed(lc):
        return noiseModel.Diagonal.Sigmas(np.array(LOOP_SIGMAS_BASE))

    def make_adaptive(alpha_t, alpha_r, floor_t=0.10, floor_r=0.05,
                      cap_t=3.0, cap_r=1.0):
        """Sigmas scaled by RAP's per-edge mean_dist (m): the per-point
        residual of the RAP alignment is a direct proxy for its pose error."""
        def adaptive(lc):
            md = float(lc.get("mean_dist", 0.5))
            st = float(np.clip(alpha_t * md, floor_t, cap_t))
            sr = float(np.clip(alpha_r * md, floor_r, cap_r))
            return noiseModel.Diagonal.Sigmas(
                np.array([sr, sr, sr, st, st, st]))
        return adaptive

    def make_huber(k, base_fn=None):
        base_fn = base_fn or fixed

        def huber(lc):
            return noiseModel.Robust.Create(
                noiseModel.mEstimator.Huber(k), base_fn(lc))
        return huber

    def make_rel_floor(md_ref, cap):
        """Floor-relative adaptive: sigma scale = clip(mean_dist/md_ref,
        1.0, cap) applied to the FIXED base sigmas. Excellent edges
        (mean_dist <= md_ref, e.g. all POSS loops at 0.62-1.0 m) keep the
        full base trust; only sloppier edges are loosened. Fixes the
        failure mode of absolute adaptive sigmas, which inflate covariance
        on near-perfect edges and degraded POSS 1.757 -> 1.961 m."""
        def rel(lc):
            md = float(lc.get("mean_dist", 0.5))
            s = float(np.clip(md / md_ref, 1.0, cap))
            return noiseModel.Diagonal.Sigmas(
                np.array(LOOP_SIGMAS_BASE) * s)
        return rel

    def make_dcs(phi, base_fn=None):
        base_fn = base_fn or fixed

        def dcs(lc):
            return noiseModel.Robust.Create(
                noiseModel.mEstimator.DCS(phi), base_fn(lc))
        return dcs

    variants = {"baseline_fixed": fixed}
    for a in (0.5, 1.0, 2.0):
        variants[f"adaptive_t{a}"] = make_adaptive(a, 0.15)
    for k in (0.5, 1.0, 2.0, 5.0):
        variants[f"huber_k{k}"] = make_huber(k)
    for p in (1.0, 5.0, 10.0, 20.0):
        variants[f"dcs_p{p}"] = make_dcs(p)
    # Combinations around sensible defaults.
    for a in (0.5, 1.0):
        ad = make_adaptive(a, 0.15)
        variants[f"ad{a}+huber_k1"] = make_huber(1.0, ad)
        variants[f"ad{a}+dcs_p10"] = make_dcs(10.0, ad)
    # Floor-relative adaptive (safe on excellent edges) x Huber.
    for ref in (0.5, 1.0):
        for cap in (3.0, 5.0):
            rf = make_rel_floor(ref, cap)
            variants[f"relfloor_r{ref}c{cap}"] = rf
            variants[f"relfloor_r{ref}c{cap}+huber"] = make_huber(1.0, rf)
    return variants


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odom", default=os.path.join(
        REPO_ROOT, "output_ros2", "live_poses_kitti02.txt"))
    ap.add_argument("--save-dir", default=os.path.join(
        REPO_ROOT, "output_ros2", "step3_kitti02"))
    ap.add_argument("--kitti-root", default=(
        "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset"))
    ap.add_argument("--seq", default="02")
    ap.add_argument("--dataset", choices=["kitti", "poss"], default="kitti",
                    help="poss: GT from RAP-SLAM/data/poss, frames 3850-5268, "
                         "apply_calib=False (POSS golden regression)")
    ap.add_argument("--noise-sweep", action="store_true",
                    help="also run the synthetic-corruption sweep")
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "output_ros2", "eval_pgo_robust_summary.json"))
    args = ap.parse_args()

    odom_poses = load_tum(args.odom)
    keyframes = json.load(open(os.path.join(args.save_dir, "keyframes.json")))
    loops = json.load(open(os.path.join(args.save_dir, "accepted_loops.json")))
    n = len(odom_poses)
    print(f"odometry frames: {n}, keyframes: {len(keyframes)}, "
          f"loop edges: {len(loops)}")

    if args.dataset == "poss":
        poss_root = os.path.join(REPO_ROOT, "data", "poss", "dataset")
        gt = real_dataset.load_gt_poses(
            os.path.join(poss_root, "poses", "00.txt"),
            None, start=3850, end=3850 + n - 1, apply_calib=False)
    else:
        gt = real_dataset.load_gt_poses(
            os.path.join(args.kitti_root, f"poses/{args.seq}.txt"),
            os.path.join(args.kitti_root, f"sequences/{args.seq}/calib.txt"),
            start=0, end=n - 1, apply_calib=True)
    gt_rel = evaluate.align_to_first_pose(list(gt))

    # ------------------------------------------------------------------
    # Q1: do RAP diagnostics predict true edge error?
    # ------------------------------------------------------------------
    print("\n=== Q1: RAP diagnostics vs true relative-pose error (GT) ===")
    rows = []
    for lc in loops:
        T_gt = np.linalg.inv(gt_rel[lc["i"]]) @ gt_rel[lc["j"]]
        dt, dr = t_rel_error(np.asarray(lc["T_rel"]), T_gt)
        rows.append((lc["overlap"], lc["mean_dist"], dt, dr,
                     float(np.linalg.norm(gt_rel[lc["j"]][:3, 3] -
                                          gt_rel[lc["i"]][:3, 3]))))
    ov = np.array([r[0] for r in rows])
    md = np.array([r[1] for r in rows])
    dt = np.array([r[2] for r in rows])
    dr = np.array([r[3] for r in rows])
    gtd = np.array([r[4] for r in rows])
    for name, x in (("overlap", ov), ("mean_dist", md)):
        r_t = np.corrcoef(x, dt)[0, 1]
        r_r = np.corrcoef(x, dr)[0, 1]
        print(f"  corr({name}, trans_err) = {r_t:+.3f}   "
              f"corr({name}, rot_err) = {r_r:+.3f}")
    qs = np.quantile(dt, [0.5, 0.9, 1.0])
    print(f"  true trans err quantiles: median {qs[0]:.2f} m, "
          f"p90 {qs[1]:.2f} m, max {qs[2]:.2f} m | "
          f"edges with GT dist>10 m: {int(np.sum(gtd > 10))}")

    # ------------------------------------------------------------------
    # Q2: variant comparison on the real 182 edges
    # ------------------------------------------------------------------
    print("\n=== Q2: PGO variants on the real 182-edge dataset ===")
    m_odom = evaluate.compute_ape_rpe(
        gt_rel[:n], evaluate.align_to_first_pose(odom_poses))
    print(f"  odometry-only APE rmse = {m_odom['ape_rmse']:.3f} m")
    results = {}
    for name, fn in make_noise_models().items():
        optimized, n_edges = run_pgo(odom_poses, keyframes, loops, fn)
        m = full_traj_ape(odom_poses, keyframes, optimized, gt_rel)
        results[name] = m
        print(f"  {name:>20s}: APE rmse {m['ape_rmse']:7.3f} m  "
              f"max {m['ape_max']:7.2f} m  (edges {n_edges})")

    best = min(results, key=lambda k: results[k]["ape_rmse"])
    print(f"\n  best variant: {best} "
          f"(rmse {results[best]['ape_rmse']:.3f} m)")

    # ------------------------------------------------------------------
    # Q3: noise sweep — corrupt a fraction of loop edges
    # ------------------------------------------------------------------
    noise_results = {}
    if args.noise_sweep:
        print("\n=== Q3: synthetic-corruption sweep ===")
        models = make_noise_models()
        sweep_variants = ["baseline_fixed", best]
        for extra in ("huber_k1.0", "dcs_p10.0"):
            if extra not in sweep_variants:
                sweep_variants.append(extra)
        rng = np.random.default_rng(7)
        for frac in (0.2, 0.5):
            for sigma_inj in (1.0, 3.0, 5.0):
                corrupted = []
                for lc in loops:
                    lc2 = dict(lc)
                    if rng.random() < frac:
                        T = np.asarray(lc["T_rel"]).copy()
                        T[:3, 3] += rng.normal(0, sigma_inj, 3)
                        lc2["T_rel"] = T.tolist()
                    corrupted.append(lc2)
                key = f"frac{frac}_sig{sigma_inj}"
                noise_results[key] = {}
                line = f"  {key}: "
                for vn in sweep_variants:
                    optimized, _ = run_pgo(odom_poses, keyframes, corrupted,
                                           models[vn])
                    m = full_traj_ape(odom_poses, keyframes, optimized, gt_rel)
                    noise_results[key][vn] = m["ape_rmse"]
                    line += f"{vn}={m['ape_rmse']:.2f} m  "
                print(line)

    summary = {
        "n_frames": n, "n_keyframes": len(keyframes), "n_loops": len(loops),
        "odom_ape_rmse": m_odom["ape_rmse"],
        "diagnostic_correlation": {
            "overlap_vs_trans_err": float(np.corrcoef(ov, dt)[0, 1]),
            "mean_dist_vs_trans_err": float(np.corrcoef(md, dt)[0, 1]),
            "overlap_vs_rot_err": float(np.corrcoef(ov, dr)[0, 1]),
            "mean_dist_vs_rot_err": float(np.corrcoef(md, dr)[0, 1]),
        },
        "variants": {k: v for k, v in results.items()},
        "best_variant": best,
        "noise_sweep": noise_results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nsummary saved to {args.out}")


if __name__ == "__main__":
    main()
