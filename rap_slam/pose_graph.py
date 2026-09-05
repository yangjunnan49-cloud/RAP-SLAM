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

Loop-edge NMS (2026-08-08, scripts/eval_loop_edge_nms.py): dense same-site
loop clusters from the streaming retrieval (KITTI-06 live: 114 edges at one
revisit site) over-constrain PGO when treated as independent — live PGO
3.221 m was WORSE than odometry 2.777 m. Greedy (i,j)-space NMS (keep the
highest-overlap edge per Chebyshev-w cell) at w=10 scans fixes it (06:
2.573 m, below odometry), improves KITTI-02 (182->38 edges, 26.423->26.018
m) and is a no-op on sparse edge sets (POSS live 1.757 m bit-identical).
Default OFF here (loop_nms_width=0 keeps the offline goldens bit-exact);
the LIVE path opts in at w=10 (its cluster-persistence retrieval is
deliberately uncapped; the offline pipeline is budget-capped).

Clique down-weighting (2026-08-23, P3; scripts/eval_clique_weighting.py):
a PRINCIPLED alternative to hard NMS suppression — keep every edge but
inflate each edge's sigmas by k^exponent, where k counts the correlated
same-site edges around it. Two clique definitions: "component" (union-find
over Chebyshev-w adjacency in (i,j) space; k = component size) and
"degree" (k = 1 + number of OTHER edges within w in both indices — a local
density that does not chain along long clusters). Motivation: k fully
correlated constraints carry ~1 edge's information; scaling sigma^2 by k
(exponent 0.5) makes their total information equal to one edge. NMS
discards information (keeps best-overlap only); clique modes keep all
geometry. Empirically compared on the dense-cluster datasets (KITTI-06/02
live) — see the eval script's report.

GNC-TLS loop-edge rejection (2026-09-05, S0a; scripts/eval_gnc_backend.py):
robust_kernel="gnc_tls"/"gnc_gm" replaces the per-edge robust mEstimator
with GTSAM's graduated non-convexity optimizer (Yang et al. T-RO 2020,
arXiv:1909.08605): odometry/prior factors are pinned via `knownInliers`,
only loop edges are graduated between inlier/outlier using per-factor
inlier cost thresholds derived from each edge's OWN noise model
(setInlierCostThresholdsAtProbability(gnc_prob)) — so relfloor/calibrated
per-edge sigmas directly set each edge's rejection threshold. Unlike Huber
(bounded influence, never rejects), TLS can fully reject the heavy-tailed
same-street aliasing edges (KITTI-02 live: 57 edges with 2.65 m median /
15 m worst true error). Default OFF (robust_kernel="huber" unchanged).
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _loop_noise(lc, base_sigmas, weighting, alpha_t, alpha_r,
                relfloor_ref, relfloor_cap, calib_params=None):
    """Per-edge base (Gaussian) noise model for one loop constraint."""
    import gtsam
    from gtsam import noiseModel

    md = float(lc.get("mean_dist", 0.5))
    if weighting == "adaptive":
        st = float(np.clip(alpha_t * md, 0.10, 3.0))
        sr = float(np.clip(alpha_r * md, 0.05, 1.0))
        sigmas = np.array([sr, sr, sr, st, st, st])
    elif weighting == "calibrated":
        # P2: sigma from the fitted conditional error model
        # sigma = a * mean_dist^b, clipped to [floor, cap]; calib_params
        # carries the fit (scripts/eval_calibrated_weighting.py), rotation
        # params in radians.
        cp = calib_params or {}
        st = float(np.clip(cp.get("a_t", 1.0) * md ** cp.get("b_t", 1.0),
                           cp.get("floor_t", 0.10), cp.get("cap_t", 3.0)))
        sr = float(np.clip(cp.get("a_r", 0.15) * md ** cp.get("b_r", 1.0),
                           cp.get("floor_r", 0.001), cp.get("cap_r", 1.0)))
        sigmas = np.array([sr, sr, sr, st, st, st])
    elif weighting == "per_edge":
        # S1b: explicit per-edge sigmas attached to the loop record as
        # sigma_t (m) / sigma_r (rad) — e.g. conformalized-quantile
        # predictions from scripts/eval_conformal_s1.py. Falls back to
        # relfloor when the record carries no explicit sigmas.
        st, sr = lc.get("sigma_t"), lc.get("sigma_r")
        if st is None or sr is None:
            s = float(np.clip(md / relfloor_ref, 1.0, relfloor_cap))
            sigmas = np.array(base_sigmas) * s
        else:
            sigmas = np.array([sr, sr, sr, st, st, st])
    elif weighting == "relfloor":
        s = float(np.clip(md / relfloor_ref, 1.0, relfloor_cap))
        sigmas = np.array(base_sigmas) * s
    else:
        sigmas = np.array(base_sigmas)
    cs = float(lc.get("clique_scale", 1.0))  # P3 clique down-weighting
    if cs != 1.0:
        sigmas = sigmas * cs
    return noiseModel.Diagonal.Sigmas(sigmas)


def nms_loop_edges(loop_constraints: List[dict],
                   width: float) -> List[dict]:
    """Greedy (i,j)-space NMS over accepted loop edges, highest overlap
    first: suppress a candidate when an already-selected edge lies within
    `width` scans in BOTH indices (|i-i'| <= w AND |j-j'| <= w). width=0
    removes only exact (i,j) duplicates. See the module docstring for the
    empirical basis (w=10 fixes dense-cluster over-weighting).
    """
    kept: List[dict] = []
    for lc in sorted(loop_constraints, key=lambda e: -e.get("overlap", 0.0)):
        if all(abs(lc["i"] - k["i"]) > width or abs(lc["j"] - k["j"]) > width
               for k in kept):
            kept.append(lc)
    return kept


def clique_downweight(loop_constraints: List[dict], width: float,
                      exponent: float = 0.5,
                      mode: str = "component") -> List[dict]:
    """Attach per-edge `clique_scale` = k**exponent for correlated
    same-site edges (see module docstring). mode="component": k = size of
    the (i,j)-space Chebyshev-w connected component (union-find, chains
    merge transitively). mode="degree": k = 1 + local neighbor count
    within w in both indices (no transitive chaining).
    """
    n = len(loop_constraints)
    if n == 0:
        return []
    if mode == "degree":
        out = []
        for a in range(n):
            ea = loop_constraints[a]
            deg = sum(1 for b in range(n) if b != a
                      and abs(ea["i"] - loop_constraints[b]["i"]) <= width
                      and abs(ea["j"] - loop_constraints[b]["j"]) <= width)
            lc2 = dict(ea)
            lc2["clique_size"] = 1 + deg
            lc2["clique_scale"] = float((1 + deg) ** exponent)
            out.append(lc2)
        return out
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(n):
        ea = loop_constraints[a]
        for b in range(a + 1, n):
            eb = loop_constraints[b]
            if abs(ea["i"] - eb["i"]) <= width and \
                    abs(ea["j"] - eb["j"]) <= width:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    sizes: Dict[int, int] = {}
    for a in range(n):
        sizes[find(a)] = sizes.get(find(a), 0) + 1
    out = []
    for a, lc in enumerate(loop_constraints):
        k = sizes[find(a)]
        lc2 = dict(lc)
        lc2["clique_size"] = k
        lc2["clique_scale"] = float(k ** exponent)
        out.append(lc2)
    return out


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
    loop_nms_width: float = 0.0,
    calib_params=None,
    loop_clique_width: float = 0.0,
    loop_clique_exponent: float = 0.5,
    loop_clique_mode: str = "component",
    gnc_prob: float = 0.99,
    gnc_mu_step: float = 1.4,
    gnc_max_iterations: int = 100,
    gnc_out: dict = None,
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
            (absolute per-edge sigmas from RAP mean_dist), "calibrated"
            (P2 parametric model via calib_params), "per_edge" (S1b:
            explicit sigma_t/sigma_r on each loop record, relfloor
            fallback), or "relfloor" (default; floor-relative loosening of
            edges sloppier than relfloor_ref, safe on near-perfect edges).
        robust_kernel: None, "huber", or "dcs" — wraps each loop edge's
            noise model in the corresponding robust mEstimator.
        huber_k: Huber kernel width (residual units of the whitened error).
        dcs_phi: DCS scale parameter.
        adaptive_alpha_t/adaptive_alpha_r: sigma-per-meter scaling of the
            translation/rotation sigmas on the edge's RAP mean_dist.
        relfloor_ref/relfloor_cap: reference mean_dist (m) and max loosening
            factor for the "relfloor" weighting.
        robust_kernel: None, "huber", "dcs", "gnc_tls", or "gnc_gm" — the
            first two wrap each loop edge's noise model in a robust
            mEstimator; the gnc_* modes leave the noise Gaussian and instead
            run GTSAM's GNC optimizer, which graduates each loop edge
            between inlier/outlier (odometry/prior pinned as knownInliers).
        gnc_prob: chi-square probability for each loop edge's inlier cost
            threshold, derived from that edge's own noise model (only used
            by the gnc_* modes).
        gnc_mu_step/gnc_max_iterations: GNC non-convexity growth rate and
            outer-iteration cap.
        gnc_out: optional dict filled with GNC diagnostics: per-factor
            weights, loop-edge weights, and the rejected count (w < 0.5).

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

    # GNC modes: everything before the loop edges (prior + odometry chain)
    # is pinned as a known inlier; only loop edges are graduated.
    use_gnc = robust_kernel in ("gnc_tls", "gnc_gm")
    n_known = graph.size() if use_gnc else 0

    # RAP loop-closure edges (optionally NMS-deduped, clique-down-weighted,
    # RAP-adaptive + robust-kernelled).
    if loop_nms_width and loop_nms_width > 0:
        n_before = len(loop_constraints)
        loop_constraints = nms_loop_edges(loop_constraints, loop_nms_width)
        logger.info(f"Loop-edge NMS (w={loop_nms_width}): "
                    f"{n_before} -> {len(loop_constraints)} edges")
    if loop_clique_width and loop_clique_width > 0:
        loop_constraints = clique_downweight(loop_constraints,
                                             loop_clique_width,
                                             loop_clique_exponent,
                                             loop_clique_mode)
        ks = [lc["clique_size"] for lc in loop_constraints]
        logger.info(f"Loop-edge clique down-weighting (w="
                    f"{loop_clique_width}, mode={loop_clique_mode}): "
                    f"{len(ks)} edges, clique sizes max {max(ks)}, "
                    f"mean {np.mean(ks):.1f}")
    for lc in loop_constraints:
        i, j = lc["i"], lc["j"]
        if i not in kf_set or j not in kf_set:
            logger.warning(f"Loop ({i}, {j}) skipped: endpoint not a keyframe")
            continue
        noise = _loop_noise(lc, loop_sigmas, loop_weighting,
                            adaptive_alpha_t, adaptive_alpha_r,
                            relfloor_ref, relfloor_cap, calib_params)
        if robust_kernel == "huber":
            noise = noiseModel.Robust.Create(
                noiseModel.mEstimator.Huber(huber_k), noise)
        elif robust_kernel == "dcs":
            noise = noiseModel.Robust.Create(
                noiseModel.mEstimator.DCS(dcs_phi), noise)
        # gnc_* modes keep the plain Gaussian noise: GNC derives each edge's
        # inlier cost threshold from it (gnc_prob) and graduates the weight.
        graph.add(BetweenFactorPose3(i, j, Pose3(lc["T_rel"]), noise))

    if use_gnc:
        gnc_params = gtsam.GncLMParams()
        gnc_params.setKnownInliers(list(range(n_known)))
        gnc_params.setLossType(
            gtsam.GncLossType.TLS if robust_kernel == "gnc_tls"
            else gtsam.GncLossType.GM)
        gnc_params.setMuStep(gnc_mu_step)
        gnc_params.setMaxIterations(gnc_max_iterations)
        gnc_params.baseOptimizerParams.setMaxIterations(50)
        optimizer = gtsam.GncLMOptimizer(graph, initial, gnc_params)
        optimizer.setInlierCostThresholdsAtProbability(gnc_prob)
        result = optimizer.optimize()
        weights = np.asarray(optimizer.getWeights())
        loop_w = weights[n_known:]
        n_rej = int(np.sum(loop_w < 0.5))
        logger.info(f"PGO[GNC-{robust_kernel[4:].upper()}]: {graph.size()} "
                    f"factors, {len(keyframes)} nodes; loop edges "
                    f"{len(loop_w)} kept / {n_rej} rejected (w<0.5); "
                    f"error {graph.error(initial):.4f} -> "
                    f"{graph.error(result):.4f}")
        if gnc_out is not None:
            gnc_out["weights"] = weights
            gnc_out["loop_weights"] = loop_w
            gnc_out["n_rejected"] = n_rej
            gnc_out["n_loop"] = len(loop_w)
    else:
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        optimizer = LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()
        logger.info(f"PGO: {graph.size()} factors, {len(keyframes)} nodes, "
                    f"error {optimizer.iterations()} iters: "
                    f"{graph.error(initial):.4f} -> "
                    f"{graph.error(result):.4f}")

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
