"""GTSAM pose-graph optimization over keyframes."""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def optimize_pose_graph(
    odom_poses: List[np.ndarray],
    keyframes: List[int],
    loop_constraints: List[dict],
    prior_sigmas=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
    odom_sigmas=(0.05, 0.05, 0.05, 0.30, 0.30, 0.30),
    loop_sigmas=(0.03, 0.03, 0.03, 0.20, 0.20, 0.20),
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
        loop_sigmas: sigmas for loop between-factors.

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

    # RAP loop-closure edges.
    loop_noise = noiseModel.Diagonal.Sigmas(np.array(loop_sigmas))
    for lc in loop_constraints:
        i, j = lc["i"], lc["j"]
        if i in kf_set and j in kf_set:
            graph.add(BetweenFactorPose3(i, j, Pose3(lc["T_rel"]), loop_noise))
        else:
            logger.warning(f"Loop ({i}, {j}) skipped: endpoint not a keyframe")

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
