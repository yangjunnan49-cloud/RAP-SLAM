"""GTSAM pose-graph optimization over keyframes.

Loop-edge noise models (RAP-CORE, 2026-08-01): RAP's per-edge diagnostics
(overlap, mean_dist) are consumed HERE, not only at the acceptance gate.
  * loop_weighting="relfloor" (DEFAULT): sigma scale =
    clip(mean_dist / relfloor_ref, 1.0, relfloor_cap) applied to the fixed
    base sigmas — edges at least as clean as relfloor_ref keep full trust,
    only sloppier edges are loosened. Cross-dataset safe.
  * loop_weighting="adaptive": absolute per-edge sigmas from mean_dist
    (sigma_t = clip(alpha_t * mean_dist, 0.10, 3.0)). Best single scheme on
    KITTI 02 (+Huber: 25.688 m) but inflates covariance on near-perfect
    edges — POSS regresses 1.757 -> 1.961 m. NOT the default.
  * robust_kernel="huber"/"dcs": wraps the per-edge noise model in a robust
    mEstimator so outlier loop edges (same-street aliasing, median 2 m /
    worst 15 m true error on KITTI 02) cannot hijack the graph.
Empirical basis (scripts/eval_pgo_robust.py, 182 KITTI-02 + 2 POSS edges):
  default relfloor(ref=1.0, cap=3.0) + Huber(k=1): KITTI 27.387 -> 26.423 m,
  POSS bit-identical 1.757 m; robust kernels stay ~25 m under 50% edge
  corruption at 5 m sigma where the fixed baseline explodes past 80 m.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _loop_noise(lc, base_sigmas, weighting, alpha_t, alpha_r,
                relfloor_ref, relfloor_cap):
    """Per-edge base (Gaussian) noise model for one loop constraint."""
    import gtsam
    from gtsam import noiseModel

    md = float(lc.get("mean_dist", 0.5))
    if weighting == "adaptive":
        st = float(np.clip(alpha_t * md, 0.10, 3.0))
        sr = float(np.clip(alpha_r * md, 0.05, 1.0))
        sigmas = np.array([sr, sr, sr, st, st, st])
    elif weighting == "relfloor":
        s = float(np.clip(md / relfloor_ref, 1.0, relfloor_cap))
        sigmas = np.array(base_sigmas) * s
    else:
        sigmas = np.array(base_sigmas)
    return noiseModel.Diagonal.Sigmas(sigmas)


def optimize_pose_graph(
    odom_poses: List[np.ndarray],
    keyframes: List[int],
    loop_constraints: List[dict],
    prior_sigmas=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
    odom_sigmas=(0.05, 0.05, 0.05, 0.30, 0.30, 0.30),
    loop_sigmas=(0.03, 0.03, 0.03, 0.20, 0.20, 0.20),
    loop_weighting: str = "relfloor",
    robust_kernel: str = "huber",
    huber_k: float = 1.0,
    dcs_phi: float = 10.0,
    adaptive_alpha_t: float = 1.0,
    adaptive_alpha_r: float = 0.15,
    relfloor_ref: float = 1.0,
    relfloor_cap: float = 3.0,
) -> Dict[int, np.ndarray]:
    """Optimize a keyframe pose graph.

    Nodes: keyframe poses. Edges: chained odometry relatives between
    consecutive keyframes + accepted RAP loop constraints.

    Args:
        odom_poses: full odometry pose sequence.
        keyframes: keyframe scan indices (sorted).
        loop_constraints: accepted loops from verify_candidates.
        prior_sigmas: (rx, ry, rz, tx, ty, tz) sigmas for the anchor prior.
        odom_sigmas: sigmas for odometry between-factors.
        loop_sigmas: sigmas for loop between-factors (fixed weighting).
        loop_weighting: "fixed" (loop_sigmas for every edge), "adaptive"
            (absolute per-edge sigmas from RAP mean_dist), or "relfloor"
            (default; floor-relative loosening of edges sloppier than
            relfloor_ref, safe on near-perfect edges).
        robust_kernel: None, "huber", or "dcs" — wraps each loop edge's
            noise model in the corresponding robust mEstimator.
        huber_k: Huber kernel width (residual units of the whitened error).
        dcs_phi: DCS scale parameter.
        adaptive_alpha_t/adaptive_alpha_r: sigma-per-meter scaling of the
            translation/rotation sigmas on the edge's RAP mean_dist.
        relfloor_ref/relfloor_cap: reference mean_dist (m) and max loosening
            factor for the "relfloor" weighting.

    Returns:
        {keyframe_index: optimized 4x4 pose}.
    """
    import gtsam
    from gtsam import (BetweenFactorPose3, LevenbergMarquardtOptimizer,
                       NonlinearFactorGraph, Pose3, PriorFactorPose3, Values,
                       noiseModel)

    graph = NonlinearFactorGraph()
    initial = Values()

    kf_set = set(keyframes)
    for k in keyframes:
        initial.insert(k, Pose3(odom_poses[k]))

    # Anchor prior on the first keyframe.
    prior_noise = noiseModel.Diagonal.Sigmas(np.array(prior_sigmas))
    graph.add(PriorFactorPose3(keyframes[0], Pose3(odom_poses[keyframes[0]]),
                               prior_noise))

    # Odometry edges between consecutive keyframes (chained scan relatives).
    odom_noise = noiseModel.Diagonal.Sigmas(np.array(odom_sigmas))
    for a, b in zip(keyframes[:-1], keyframes[1:]):
        T_rel = np.linalg.inv(odom_poses[a]) @ odom_poses[b]
        graph.add(BetweenFactorPose3(a, b, Pose3(T_rel), odom_noise))

    # RAP loop-closure edges (optionally RAP-adaptive + robust-kernelled).
    for lc in loop_constraints:
        i, j = lc["i"], lc["j"]
        if i not in kf_set or j not in kf_set:
            logger.warning(f"Loop ({i}, {j}) skipped: endpoint not a keyframe")
            continue
        noise = _loop_noise(lc, loop_sigmas, loop_weighting,
                            adaptive_alpha_t, adaptive_alpha_r,
                            relfloor_ref, relfloor_cap)
        if robust_kernel == "huber":
            noise = noiseModel.Robust.Create(
                noiseModel.mEstimator.Huber(huber_k), noise)
        elif robust_kernel == "dcs":
            noise = noiseModel.Robust.Create(
                noiseModel.mEstimator.DCS(dcs_phi), noise)
        graph.add(BetweenFactorPose3(i, j, Pose3(lc["T_rel"]), noise))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    optimizer = LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    logger.info(f"PGO: {graph.size()} factors, {len(keyframes)} nodes, "
                f"error {optimizer.iterations()} iters: "
                f"{graph.error(initial):.4f} -> {graph.error(result):.4f}")

    return {k: result.atPose3(k).matrix() for k in keyframes}


def propagate_to_full_trajectory(odom_poses: List[np.ndarray],
                                 keyframes: List[int],
                                 optimized: Dict[int, np.ndarray]) -> List[np.ndarray]:
    """Apply per-keyframe corrections to the full odometry trajectory.

    Scan s in [kf_k, kf_{k+1}) gets correction C_k = T_opt_k @ T_odom_k^{-1}.
    """
    n = len(odom_poses)
    out = [None] * n
    kf_sorted = sorted(keyframes)
    corrections = {k: optimized[k] @ np.linalg.inv(odom_poses[k]) for k in kf_sorted}
    for s in range(n):
        # Most recent keyframe <= s.
        k = kf_sorted[0]
        for kf in kf_sorted:
            if kf <= s:
                k = kf
            else:
                break
        out[s] = corrections[k] @ odom_poses[s]
    return out
