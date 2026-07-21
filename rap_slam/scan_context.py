"""Scan Context global place recognition for loop-candidate retrieval.

Implements Kim & Kim, "Scan Context: Egocentric Spatial Descriptor of a
3D Point Cloud Map" (IROS 2018). Each scan is encoded as a polar grid
(rings x sectors) of maximum point heights in the sensor frame, making
the descriptor translation-invariant by construction; rotation shows up
as a column shift that the matcher searches over (which also yields a
yaw estimate as a byproduct).

Unlike the odometry-proximity retrieval in loop_closure.py, this module
never looks at estimated poses, so it keeps finding revisited places
even when odometry drift exceeds any reasonable search radius.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def make_scan_context(points: np.ndarray, n_rings: int = 20,
                      n_sectors: int = 60, max_radius: float = 80.0,
                      z_offset: float = 2.0) -> np.ndarray:
    """Build the Scan Context descriptor of a single scan.

    Args:
        points: (N, 3) points in the sensor frame.
        n_rings: number of radial bins.
        n_sectors: number of azimuth bins.
        max_radius: points beyond this range are dropped.
        z_offset: added to z so heights are non-negative (a car-mounted
            LiDAR sees the ground at about -1.5 to -1.8 m).

    Returns:
        (n_rings, n_sectors) float32 array; empty cells are 0.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((n_rings, n_sectors), dtype=np.float32)
    r = np.linalg.norm(pts[:, :2], axis=1)
    mask = (r > 1e-6) & (r <= max_radius)
    pts = pts[mask]
    r = r[mask]

    ring_idx = np.clip(np.ceil((r / max_radius) * n_rings).astype(np.int64) - 1,
                       0, n_rings - 1)
    azim = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))  # -180..180
    sector_idx = np.clip(
        np.ceil(((azim + 180.0) / 360.0) * n_sectors).astype(np.int64) - 1,
        0, n_sectors - 1)

    heights = np.maximum(pts[:, 2] + z_offset, 0.0).astype(np.float32)
    sc = np.zeros((n_rings, n_sectors), dtype=np.float32)
    np.maximum.at(sc.reshape(-1), ring_idx * n_sectors + sector_idx, heights)
    return sc


def make_ring_key(sc: np.ndarray) -> np.ndarray:
    """Per-ring mean of the descriptor, used as a lightweight retrieval key."""
    return sc.mean(axis=1)


def sc_distance(sc_a: np.ndarray, sc_b: np.ndarray) -> Tuple[float, int]:
    """Column-shift-invariant distance between two Scan Contexts.

    Distance is the mean over columns of (1 - cosine similarity) at the
    best column shift. Columns empty in both descriptors count as fully
    similar; columns empty in exactly one count as fully dissimilar.

    Returns:
        (distance, best_shift): distance in [0, 1]; best_shift is the
        column rotation that aligns sc_b to sc_a (one column corresponds
        to 360 / n_sectors degrees of yaw).
    """
    n_rings, n_sectors = sc_a.shape
    norm_a = np.linalg.norm(sc_a, axis=0)
    norm_b = np.linalg.norm(sc_b, axis=0)

    best_d, best_shift = np.inf, 0
    for shift in range(n_sectors):
        b_shift = np.roll(sc_b, shift, axis=1)
        nb = np.roll(norm_b, shift)
        denom = norm_a * nb
        dot = (sc_a * b_shift).sum(axis=0)

        sim = np.zeros(n_sectors, dtype=np.float64)
        valid = denom > 1e-8
        sim[valid] = dot[valid] / denom[valid]
        sim[(norm_a < 1e-8) & (nb < 1e-8)] = 1.0  # both empty -> similar
        d = 1.0 - float(sim.mean())
        if d < best_d:
            best_d, best_shift = d, shift
    return float(best_d), int(best_shift)


def build_descriptors(scans: List[np.ndarray], indices: List[int],
                      n_rings: int = 20, n_sectors: int = 60,
                      max_radius: float = 80.0,
                      z_offset: float = 2.0) -> Dict[int, dict]:
    """Compute Scan Context + ring key for the given scan indices."""
    desc = {}
    for idx in indices:
        sc = make_scan_context(scans[idx], n_rings=n_rings,
                               n_sectors=n_sectors, max_radius=max_radius,
                               z_offset=z_offset)
        desc[idx] = {"sc": sc, "ring_key": make_ring_key(sc)}
    logger.info(f"Built Scan Contexts for {len(desc)} scans "
                f"({n_rings}x{n_sectors}, max_radius={max_radius} m)")
    return desc


def find_candidates_sc(descriptors: Dict[int, dict], keyframes: List[int],
                       min_separation: int = 5, dist_thresh: float = 0.25,
                       shortlist: int = 10,
                       max_candidates: int = 12) -> List[Tuple[int, int]]:
    """Find loop-candidate keyframe pairs by global place recognition.

    For each keyframe (in scan-index order), earlier keyframes separated
    by at least min_separation are shortlisted by ring-key L2 distance,
    then verified with the column-shift SC distance.

    Args:
        descriptors: scan index -> {"sc", "ring_key"} (build_descriptors).
        keyframes: keyframe scan indices.
        min_separation: min scan-index separation (excludes temporal
            neighbors, which trivially look similar).
        dist_thresh: accept pairs with SC distance below this.
        shortlist: ring-key shortlist size per query.
        max_candidates: cap on returned pairs (most confident first).

    Returns:
        List of (kf_idx_a, kf_idx_b) scan-index pairs with a < b.
    """
    kfs = [k for k in keyframes if k in descriptors]
    pairs = []
    for jj in range(1, len(kfs)):
        j = kfs[jj]
        rk_j = descriptors[j]["ring_key"]
        cand = []
        for ii in range(jj):
            i = kfs[ii]
            if j - i < min_separation:
                continue
            cand.append((i, float(np.linalg.norm(rk_j - descriptors[i]["ring_key"]))))
        cand.sort(key=lambda x: x[1])
        for i, _ in cand[:shortlist]:
            d, shift = sc_distance(descriptors[i]["sc"], descriptors[j]["sc"])
            if d < dist_thresh:
                pairs.append((i, j, d, shift))
    pairs.sort(key=lambda x: x[2])
    pairs = pairs[:max_candidates]
    deg_per_col = 360.0 / next(iter(descriptors.values()))["sc"].shape[1]
    logger.info(f"Scan Context candidates: {len(pairs)} pairs "
                f"(dist<{dist_thresh}, sep>={min_separation}): "
                f"{[(i, j, round(d, 3)) for i, j, d, _ in pairs]}")
    for i, j, d, shift in pairs:
        logger.debug(f"  SC pair ({i}, {j}): dist={d:.3f} "
                     f"yaw_hint={shift * deg_per_col:.1f} deg")
    return [(i, j) for i, j, _, _ in pairs]
