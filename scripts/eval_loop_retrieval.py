#!/usr/bin/env python3
"""Evaluate loop-candidate retrieval methods against ground truth.

Compares odometry-proximity retrieval vs Scan Context global place
recognition on a real KITTI-format sequence (default: the POSS segment).
A keyframe pair counts as a true loop when the GT positions are closer
than --gt-loop-dist and the scan indices are separated by >= --loop-sep.

Metrics: precision / recall / F1 of the proposed candidate pairs, swept
over the proximity radius and the SC distance threshold. Odometry poses
are loaded from a previous pipeline run when available (--odom-poses),
otherwise KISS-ICP is re-run.

Run inside WSL:
  python scripts/eval_loop_retrieval.py
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rap_slam import keyframes as kf_mod
from rap_slam import odometry, real_dataset
from rap_slam import scan_context as sc_mod

logger = logging.getLogger("eval_loop_retrieval")


def parse_args():
    p = argparse.ArgumentParser(description="Loop-retrieval precision/recall evaluation")
    p.add_argument("--real-root", default="/mnt/e/githouse/RAP-SLAM/data/poss")
    p.add_argument("--real-seq", default="00")
    p.add_argument("--real-start", type=int, default=3850)
    p.add_argument("--real-end", type=int, default=5268)
    p.add_argument("--real-stride", type=int, default=1)
    p.add_argument("--real-apply-calib", action="store_true")
    p.add_argument("--kf-trans", type=float, default=6.0)
    p.add_argument("--kf-rot", type=float, default=12.0)
    p.add_argument("--loop-sep", type=int, default=200)
    p.add_argument("--gt-loop-dist", type=float, default=3.0,
                   help="GT position distance below which a keyframe pair is a true loop (m)")
    p.add_argument("--odom-poses", default="/mnt/e/githouse/RAP-SLAM/output_poss/odom_poses.txt",
                   help="cached odometry poses from a previous pipeline run; "
                        "re-run KISS-ICP when missing")
    p.add_argument("--odom-max-range", type=float, default=80.0)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--sc-max-radius", type=float, default=80.0)
    p.add_argument("--sc-rings", type=int, default=20)
    p.add_argument("--sc-sectors", type=int, default=60)
    p.add_argument("--sc-shortlist", type=int, default=10)
    return p.parse_args()


def gt_loop_pairs(gt_poses: np.ndarray, keyframes: List[int],
                  min_sep: int, dist: float) -> Set[Tuple[int, int]]:
    """True loop pairs: keyframes separated in time but close in GT space."""
    pairs = set()
    pos = {kf: gt_poses[kf][:3, 3] for kf in keyframes}
    for ii in range(len(keyframes)):
        for jj in range(ii + 1, len(keyframes)):
            i, j = keyframes[ii], keyframes[jj]
            if j - i < min_sep:
                continue
            if np.linalg.norm(pos[i] - pos[j]) < dist:
                pairs.add((i, j))
    return pairs


def nearest_gt_pairs(gt_poses: np.ndarray, keyframes: List[int],
                     min_sep: int, top: int = 12) -> List[Tuple[int, int, float]]:
    """Closest keyframe pairs in GT space, for inspecting the loop structure."""
    pos = {kf: gt_poses[kf][:3, 3] for kf in keyframes}
    pairs = []
    for ii in range(len(keyframes)):
        for jj in range(ii + 1, len(keyframes)):
            i, j = keyframes[ii], keyframes[jj]
            if j - i < min_sep:
                continue
            pairs.append((i, j, float(np.linalg.norm(pos[i] - pos[j]))))
    pairs.sort(key=lambda x: x[2])
    return pairs[:top]


def prf1(proposed: Set[Tuple[int, int]], truth: Set[Tuple[int, int]]):
    tp = len(proposed & truth)
    prec = tp / len(proposed) if proposed else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info(f"Loading real sequence frames [{args.real_start}:{args.real_end}]")
    data = real_dataset.load_real_sequence(
        args.real_root, sequence_id=args.real_seq, start=args.real_start,
        end=args.real_end, min_range=1.5, stride=args.real_stride,
        apply_calib=args.real_apply_calib)
    gt_poses, scans = data["gt_poses"], data["scans"]
    n = len(scans)
    logger.info(f"{n} scans loaded")

    # Odometry poses: reuse cached pipeline output when possible.
    if os.path.exists(args.odom_poses):
        from rap_slam import sequence as seq_mod
        odom_poses = list(seq_mod.load_kitti_poses(args.odom_poses))
        assert len(odom_poses) == n, "cached odom pose count mismatch"
        logger.info(f"Loaded cached odometry poses from {args.odom_poses}")
    else:
        logger.info("No cached odometry poses; running KISS-ICP")
        odom_poses, backend = odometry.run_odometry(scans, max_range=args.odom_max_range)
        logger.info(f"Odometry done ({backend})")

    keyframes = kf_mod.select_keyframes(
        odom_poses, trans_thresh=args.kf_trans, rot_thresh_deg=args.kf_rot)

    # ------------------------------------------------ GT loop structure
    truth = gt_loop_pairs(gt_poses, keyframes, args.loop_sep, args.gt_loop_dist)
    near = nearest_gt_pairs(gt_poses, keyframes, args.loop_sep)
    print("\n=== GT loop structure ===")
    print(f"keyframes={len(keyframes)}, true loop pairs "
          f"(GT dist < {args.gt_loop_dist} m, sep >= {args.loop_sep}): {sorted(truth)}")
    print("closest GT keyframe pairs: "
          f"{[(i, j, round(d, 2)) for i, j, d in near]}")

    # ------------------------------------------------ proximity sweep
    print("\n=== Proximity retrieval sweep (odometry-based) ===")
    print(f"{'radius_m':>9} {'n_prop':>7} {'prec':>6} {'rec':>6} {'f1':>6}")
    odom_pos = {kf: odom_poses[kf][:3, 3] for kf in keyframes}
    prox_dist = {}
    for ii in range(len(keyframes)):
        for jj in range(ii + 1, len(keyframes)):
            i, j = keyframes[ii], keyframes[jj]
            if j - i < args.loop_sep:
                continue
            prox_dist[(i, j)] = float(np.linalg.norm(odom_pos[i] - odom_pos[j]))
    for radius in (3.0, 5.0, 8.0, 10.0, 15.0, 20.0):
        proposed = {p for p, d in sorted(prox_dist.items(), key=lambda x: x[1])
                    if d < radius}
        proposed = set(list(proposed)[:args.max_candidates])
        prec, rec, f1 = prf1(proposed, truth)
        print(f"{radius:>9.1f} {len(proposed):>7} {prec:>6.2f} {rec:>6.2f} {f1:>6.2f}")

    # ------------------------------------------------ Scan Context sweep
    t0 = time.time()
    descriptors = sc_mod.build_descriptors(
        scans, keyframes, n_rings=args.sc_rings, n_sectors=args.sc_sectors,
        max_radius=args.sc_max_radius)
    logger.info(f"Descriptors built in {time.time() - t0:.1f} s")

    # Ring-key shortlist + SC distance computed once; thresholds swept after.
    t0 = time.time()
    sc_dist = {}
    kfs = list(keyframes)
    for jj in range(1, len(kfs)):
        j = kfs[jj]
        rk_j = descriptors[j]["ring_key"]
        cand = []
        for ii in range(jj):
            i = kfs[ii]
            if j - i < args.loop_sep:
                continue
            cand.append((i, float(np.linalg.norm(rk_j - descriptors[i]["ring_key"]))))
        cand.sort(key=lambda x: x[1])
        for i, _ in cand[:args.sc_shortlist]:
            d, shift = sc_mod.sc_distance(descriptors[i]["sc"], descriptors[j]["sc"])
            sc_dist[(i, j)] = d
    logger.info(f"SC matching ({len(sc_dist)} shortlisted pairs) in "
                f"{time.time() - t0:.1f} s")

    print("\n=== Scan Context retrieval sweep ===")
    print(f"{'thresh':>7} {'n_prop':>7} {'prec':>6} {'rec':>6} {'f1':>6}")
    for thresh in (0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40):
        proposed = {p for p, d in sorted(sc_dist.items(), key=lambda x: x[1])
                    if d < thresh}
        proposed = set(list(proposed)[:args.max_candidates])
        prec, rec, f1 = prf1(proposed, truth)
        print(f"{thresh:>7.2f} {len(proposed):>7} {prec:>6.2f} {rec:>6.2f} {f1:>6.2f}")

    # SC distances of the true pairs (are they separable from the rest?)
    print("\n=== SC distance of true pairs vs shortlisted false pairs ===")
    true_d = sorted((p, round(d, 3)) for p, d in sc_dist.items() if p in truth)
    false_d = sorted((round(d, 3), p) for p, d in sc_dist.items()
                     if p not in truth)[:10]
    print(f"true pairs in shortlist: {true_d}")
    print(f"10 lowest false-pair distances: {false_d}")
    missing = truth - set(sc_dist)
    print(f"true pairs NOT in ring-key shortlist: {sorted(missing)}")

    # ------------------------------------------------ head-to-head
    print("\n=== Head-to-head at pipeline defaults (proximity 15 m vs SC 0.25) ===")
    prox_def = set(list({p for p, d in sorted(prox_dist.items(), key=lambda x: x[1])
                         if d < 15.0})[:args.max_candidates])
    sc_def = set(list({p for p, d in sorted(sc_dist.items(), key=lambda x: x[1])
                       if d < 0.25})[:args.max_candidates])
    for name, prop in (("proximity", prox_def), ("scan_context", sc_def)):
        prec, rec, f1 = prf1(prop, truth)
        print(f"{name:>12}: proposed={sorted(prop)} prec={prec:.2f} rec={rec:.2f} f1={f1:.2f}")
    print(f"SC-only rescues (missed by proximity): {sorted(sc_def - prox_def)}")
    print(f"proximity-only (missed by SC): {sorted(prox_def - sc_def)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
