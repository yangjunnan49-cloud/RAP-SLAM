"""Loop-closure candidate detection and RAP-based geometric verification."""

import logging
import time
from typing import List, Tuple

import numpy as np
from .verification_log import RecordingError

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
                      min_overlap: float = 0.7,
                      max_mean_dist: float = 2.5,
                      multigen_k: int = 1,
                      min_overlap_min: float = 0.4,
                      max_disp_r: float = 30.0,
                      gt_poses: np.ndarray = None,
                      gate_model=None, measurement_log=None) -> List[dict]:
    """Verify loop candidates with RAP 2-way registration.

    A candidate (i, j) is accepted when the scans actually align after RAP
    registration: overlap_ratio >= min_overlap AND mean_dist <= max_mean_dist
    (max_mean_dist <= 0 disables the second gate). Defaults come from the
    KITTI 02 discriminator experiment (scripts/eval_acceptance_gate.py) plus
    the big-budget live run: true loops overlap 0.76-0.96 / mean_dist
    0.39-1.45 m; false loops overlap <= 0.64, with gross misregistrations at
    mean_dist 3.7-22.1 m. Overlap carries the separation (gap 0.64..0.76);
    mean_dist guards against catastrophic misalignments.

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
        gate_model: optional rap_slam.conformal_gate.ConformalGate (S1a arm).
            When given (and multigen_k == 1), acceptance is decided by the
            conformal acceptance region on RAP's intrinsic diagnostics with a
            finite-sample false-accept bound; the heuristic two-metric gate
            is the fallback when a feature is missing.

    Returns:
        List of accepted constraint dicts: {i, j, T_rel, overlap, t_trans_err,
        t_rot_err_deg, gt_trans_err, gt_rot_err_deg, t_infer}.
    """
    accepted = []
    for (i, j) in pairs:
        attempt = measurement_log.start(i,j) if measurement_log is not None else None
        def capture(generation_id, transform, diagnostics):
            measurement_log.generation(attempt, generation_id, transform, diagnostics)
        t0 = time.time()
        try:
            if multigen_k > 1:
                callback = {"generation_callback": capture} if measurement_log is not None else {}
                T_rel, diag = rap_backend.register_consistent(
                    scans[i], scans[j], k=multigen_k, **callback)
            else:
                T_rel, diag = rap_backend.register(scans[i], scans[j])
                if measurement_log is not None:
                    capture(0,T_rel,diag)
        except RecordingError:
            raise
        except Exception as e:
            if measurement_log is not None:
                measurement_log.failure(attempt,e)
            logger.warning(f"RAP failed for pair ({i}, {j}): {e}")
            continue

        # Consistency vs odometry estimate.
        T_odom = np.linalg.inv(poses[i]) @ poses[j]
        dT = np.linalg.inv(T_odom) @ T_rel
        trans_err = float(np.linalg.norm(dT[:3, 3]))
        rot_err = _rotation_error_deg(dT[:3, :3])
        overlap = diag["overlap_ratio"]
        mean_dist = diag.get("mean_dist", float("nan"))

        msg = (f"pair ({i:3d}, {j:3d}): overlap={overlap:.2f} "
               f"mean_dist={mean_dist:.2f} m "
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

        # odom-consistency is diagnostic-only (see docstring)
        # RC-3 (2026-08-08, eval_multigen_confidence.py): with k>1 the gate
        # switches to CROSS-GENERATION consistency — false loops are
        # unstable across generations (min overlap <= 0.16, 90-deg yaw
        # flips) while true loops reproduce (min overlap >= 0.66), so
        # min_overlap_min=0.4 has a huge two-sided margin vs the
        # single-generation gate's 0.06.
        gate_used = "heuristic"
        if multigen_k > 1:
            gate_used = "multigen_consistency"
            ok = (diag["overlap_min"] >= min_overlap_min and
                  (max_mean_dist <= 0.0 or mean_dist <= max_mean_dist) and
                  diag["disp_r"] <= max_disp_r)
            msg += (f" | multigen k={multigen_k} "
                    f"overlap_min={diag['overlap_min']:.2f} "
                    f"disp_t={diag['disp_t']:.2f} m "
                    f"disp_r={diag['disp_r']:.1f} deg")
        elif gate_model is not None:
            # S1a arm: conformal acceptance region (finite-sample
            # false-accept bound; rap_slam/conformal_gate.py). Uses the RAP
            # intrinsic diagnostics only; a None verdict (missing features)
            # falls back to the heuristic gate.
            s = gate_model.score({"overlap": overlap, "mean_dist": mean_dist})
            if s is None:
                gate_used = "heuristic_fallback_missing_conformal_features"
                ok = (overlap >= min_overlap and
                      (max_mean_dist <= 0.0 or mean_dist <= max_mean_dist))
                msg += " | conformal n/a (heuristic fallback)"
            else:
                gate_used = "conformal_legacy_optional"
                ok = s < gate_model.Q
                msg += (f" | conformal danger={s:.3f} "
                        f"Q={gate_model.Q:.3f}")
        else:
            ok = (overlap >= min_overlap and
                  (max_mean_dist <= 0.0 or mean_dist <= max_mean_dist))
        msg += " => ACCEPT" if ok else " => reject"
        logger.info(msg)

        selected_id = None
        if measurement_log is not None:
            selected_id = measurement_log.decision(attempt, int(diag.get("selected_generation",0)),
                                                   bool(ok), gate_used, T_rel, diag)
        if ok:
            entry = {
                "i": i, "j": j, "T_rel": T_rel,
                "overlap": overlap, "mean_dist": mean_dist,
                "odom_trans_err": trans_err, "odom_rot_err_deg": rot_err,
                "gt_trans_err": gt_te, "gt_rot_err_deg": gt_re,
                "t_infer": diag["t_inference"],
                "t_total": time.time() - t0,
            }
            if selected_id is not None:
                entry["measurement_id"] = selected_id
            if multigen_k > 1:
                entry.update({
                    "overlap_min": diag["overlap_min"],
                    "disp_t": diag["disp_t"], "disp_r": diag["disp_r"],
                    "n_generations": diag["n_generations"],
                })
            accepted.append(entry)
    logger.info(f"Accepted {len(accepted)}/{len(pairs)} loop closures")
    return accepted
