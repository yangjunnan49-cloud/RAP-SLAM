"""Loop-closure candidate detection and RAP-based geometric verification."""

import logging
import time
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def find_candidates(poses: List[np.ndarray], keyframes: List[int],
                    dist_thresh: float = 15.0, min_separation: int = 5,
                    max_candidates: int = 12) -> List[Tuple[int, int]]:
    """Find loop-closure candidate keyframe pairs by odometry proximity.

    Args:
        poses: full odometry pose sequence (sensor-to-world).
        keyframes: keyframe scan indices.
        dist_thresh: max distance between keyframe positions (meters).
        min_separation: min scan-index separation (excludes temporal neighbors).
        max_candidates: cap on returned pairs (closest first).

    Returns:
        List of (kf_idx_a, kf_idx_b) scan-index pairs with a < b.
    """
    pos = {kf: poses[kf][:3, 3] for kf in keyframes}
    pairs = []
    for ii in range(len(keyframes)):
        for jj in range(ii + 1, len(keyframes)):
            i, j = keyframes[ii], keyframes[jj]
            if j - i < min_separation:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < dist_thresh:
                pairs.append((i, j, d))
    pairs.sort(key=lambda x: x[2])
    pairs = pairs[:max_candidates]
    logger.info(f"Loop candidates: {len(pairs)} pairs "
                f"(dist<{dist_thresh} m, sep>={min_separation}): "
                f"{[(i, j, round(d, 1)) for i, j, d in pairs]}")
    return [(i, j) for i, j, _ in pairs]


def merge_candidates(proximity_pairs: List[Tuple[int, int]],
                     sc_pairs: List[Tuple[int, int]],
                     max_candidates: int = 12) -> List[Tuple[int, int]]:
    """Merge candidate pairs from proximity and Scan Context retrieval.

    Proximity pairs (near-certain loops under small drift) keep priority;
    Scan Context pairs fill the remaining budget. Duplicates are removed.
    """
    merged, seen = [], set()
    for source, pairs in (("proximity", proximity_pairs), ("scan_context", sc_pairs)):
        for (i, j) in pairs:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            merged.append(key)
            logger.info(f"  candidate {key} proposed by {source}")
    merged = merged[:max_candidates]
    logger.info(f"Merged loop candidates: {len(merged)} pairs {merged}")
    return merged


def _rotation_error_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def verify_candidates(pairs: List[Tuple[int, int]], scans: List[np.ndarray],
                      poses: List[np.ndarray], rap_backend,
                      trans_consistency: float = 5.0,
                      rot_consistency_deg: float = 25.0,
                      min_overlap: float = 0.25,
                      gt_poses: np.ndarray = None) -> List[dict]:
    """Verify loop candidates with RAP 2-way registration.

    A candidate (i, j) is accepted when the scans actually overlap after
    RAP alignment: overlap_ratio >= min_overlap.

    Rationale: the odometry-consistency values (trans/rot discrepancy between
    RAP's transform and the drifting odometry estimate) are logged as
    diagnostics only. Gating acceptance on them is self-defeating — loop
    closure exists precisely to correct odometry drift, so a true loop under
    heavy drift will always disagree with the odometry prior (observed on the
    POSS segment: 13.3 m drift, RAP vs GT error only 0.15-0.47 m).

    Args:
        pairs: candidate (i, j) scan-index pairs.
        scans: all scans.
        poses: odometry poses (for candidate prior and consistency check).
        rap_backend: RAPBackend instance.
        gt_poses: optional GT poses for logging true errors.

    Returns:
        List of accepted constraint dicts: {i, j, T_rel, overlap, t_trans_err,
        t_rot_err_deg, gt_trans_err, gt_rot_err_deg, t_infer}.
    """
    accepted = []
    for (i, j) in pairs:
        t0 = time.time()
        try:
            T_rel, diag = rap_backend.register(scans[i], scans[j])
        except Exception as e:
            logger.warning(f"RAP failed for pair ({i}, {j}): {e}")
            continue

        # Consistency vs odometry estimate.
        T_odom = np.linalg.inv(poses[i]) @ poses[j]
        dT = np.linalg.inv(T_odom) @ T_rel
        trans_err = float(np.linalg.norm(dT[:3, 3]))
        rot_err = _rotation_error_deg(dT[:3, :3])
        overlap = diag["overlap_ratio"]

        msg = (f"pair ({i:3d}, {j:3d}): overlap={overlap:.2f} "
               f"odom-consistency dT={trans_err:.2f} m dR={rot_err:.1f} deg "
               f"[{diag['t_inference']:.1f} s infer]")
        if gt_poses is not None:
            T_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
            dG = np.linalg.inv(T_gt) @ T_rel
            gt_te = float(np.linalg.norm(dG[:3, 3]))
            gt_re = _rotation_error_deg(dG[:3, :3])
            msg += f" | GT error dT={gt_te:.2f} m dR={gt_re:.1f} deg"
        else:
            gt_te = gt_re = None

        ok = overlap >= min_overlap  # odom-consistency is diagnostic-only (see docstring)
        msg += " => ACCEPT" if ok else " => reject"
        logger.info(msg)

        if ok:
            accepted.append({
                "i": i, "j": j, "T_rel": T_rel,
                "overlap": overlap,
                "odom_trans_err": trans_err, "odom_rot_err_deg": rot_err,
                "gt_trans_err": gt_te, "gt_rot_err_deg": gt_re,
                "t_infer": diag["t_inference"],
                "t_total": time.time() - t0,
            })
    logger.info(f"Accepted {len(accepted)}/{len(pairs)} loop closures")
    return accepted
