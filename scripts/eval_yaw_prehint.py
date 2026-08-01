#!/usr/bin/env python
"""RC-2: measure RAP accuracy with vs without Scan Context yaw pre-rotation.

Hypothesis: RAP's global registration can land in poor local minima when the
relative yaw between two scans is large (e.g. opposite-direction urban
streets on KITTI 02). Scan Context already yields a per-pair yaw hint
(column shift x 6 deg) that is currently only logged. This script re-scores
loop pairs two ways:
  vanilla : T0 = register(scan_i, scan_j)
  prehint : T1 = register(scan_i, Rz(psi_hint) @ scan_j) @ Rz(psi_hint)
and reports T_rel error vs GT, overlap, mean_dist, and whether the
acceptance gate (overlap >= 0.7 AND mean_dist <= 2.5 m) changes its verdict.

Convention check: psi_hint should approximate yaw_gt = yaw(T_i^-1 @ T_j)
(derivation: sc_distance rolls sc_b by `shift` columns to match sc_a, and
one column = 360/n_sectors deg of relative yaw). The script prints the
per-pair hint error so a sign/convention flip is visible immediately.

Run in the conda env:
  /root/miniconda3/envs/rap-inference/bin/python scripts/eval_yaw_prehint.py
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rap_slam import real_dataset  # noqa: E402
from rap_slam.scan_context import make_scan_context, sc_distance  # noqa: E402

KITTI_SEQ_DIR = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset/sequences/02"
POSES_FILE = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset/poses/02.txt"
FALSE_LOOPS_JSON = os.path.join(
    REPO_ROOT, "output_ros2/step3_kitti02_bigbudget/accepted_loops.json")
OUT_JSON = os.path.join(REPO_ROOT, "output_ros2/eval_yaw_prehint.json")

OVERLAP_MIN, MEAN_DIST_MAX = 0.7, 2.5  # live acceptance gate

# True pairs from the GT revisit clusters (same 8 as eval_acceptance_gate).
TRUE_PAIRS = [
    (930, 4200), (950, 4220), (980, 4240), (990, 4250),
    (1010, 4260), (1800, 4400), (1810, 4400), (1820, 4410),
]

# Extra large-yaw true pairs mined from the GT clusters (filled in by
# mine_large_yaw_pairs() at runtime, deterministic).
N_EXTRA_TRUE = 6


def load_scan(seq_dir: str, idx: int) -> np.ndarray:
    path = os.path.join(seq_dir, "velodyne", f"{idx:06d}.bin")
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def yaw_deg(T: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def rotz4(psi_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(psi_deg)), np.sin(np.radians(psi_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return T


def rotation_error_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def mine_large_yaw_pairs(gt) -> list:
    """Stride-10 pairs inside the two GT revisit clusters, sorted by |yaw|."""
    windows = [(range(930, 1031, 10), range(4200, 4271, 10)),
               (range(1800, 1821, 10), range(4400, 4411, 10))]
    cand = []
    for wa, wb in windows:
        for i in wa:
            for j in wb:
                d = float(np.linalg.norm(gt[i][:3, 3] - gt[j][:3, 3]))
                if d < 5.0 and (i, j) not in TRUE_PAIRS:
                    y = abs(yaw_deg(np.linalg.inv(gt[i]) @ gt[j]))
                    cand.append((i, j, d, y))
    cand.sort(key=lambda x: -x[3])
    return [(i, j) for i, j, _, _ in cand[:N_EXTRA_TRUE]]


def gate(overlap: float, mean_dist: float) -> bool:
    return overlap >= OVERLAP_MIN and mean_dist <= MEAN_DIST_MAX


def main():
    from rap_slam.rap_backend import RAPBackend
    import torch

    with open(FALSE_LOOPS_JSON) as f:
        false_pairs = [(lc["i"], lc["j"]) for lc in json.load(f)]

    gt = list(real_dataset.load_gt_poses(
        POSES_FILE, os.path.join(KITTI_SEQ_DIR, "calib.txt"),
        start=0, end=4660, apply_calib=True))

    extra_true = mine_large_yaw_pairs(gt)
    true_pairs = TRUE_PAIRS + extra_true
    print(f"pairs: {len(true_pairs)} TRUE ({len(extra_true)} mined large-yaw: "
          f"{extra_true}) + {len(false_pairs)} FALSE")

    rap = RAPBackend()  # defaults: rap_12 ckpt, voxel 0.4, 4000 pts, spinnet

    needed = sorted({s for pair in true_pairs + false_pairs for s in pair})
    scans = {s: load_scan(KITTI_SEQ_DIR, s) for s in needed}
    scs = {s: make_scan_context(scans[s], max_radius=80.0) for s in needed}
    print(f"loaded {len(scans)} scans\n")

    def reseed():
        torch.manual_seed(42)
        np.random.seed(42)

    hdr = (f"{'pair':>13} {'kind':>5} {'yaw_gt':>7} {'hint':>6} {'herr':>6} | "
           f"{'ov0':>5} {'md0':>5} {'dT0':>5} {'dR0':>5} {'g0':>2} | "
           f"{'ov1':>5} {'md1':>5} {'dT1':>5} {'dR1':>5} {'g1':>2}")
    print(hdr)
    rows = []
    for kind, pairs in (("TRUE", true_pairs), ("FALSE", false_pairs)):
        for i, j in pairs:
            T_gt = np.linalg.inv(gt[i]) @ gt[j]
            y_gt = yaw_deg(T_gt)
            d_sc, shift = sc_distance(scs[i], scs[j])
            psi = shift * (360.0 / scs[i].shape[1])
            herr = wrap_deg(psi - y_gt)

            reseed()
            T0, dg0 = rap.register(scans[i], scans[j])
            dG0 = np.linalg.inv(T_gt) @ T0
            te0 = float(np.linalg.norm(dG0[:3, 3]))
            re0 = rotation_error_deg(dG0[:3, :3])
            g0 = gate(dg0["overlap_ratio"], dg0["mean_dist"])

            Rz = rotz4(psi)
            b_rot = (Rz[:3, :3] @ scans[j].T).T
            reseed()
            T1p, dg1 = rap.register(scans[i], b_rot)
            T1 = T1p @ Rz
            dG1 = np.linalg.inv(T_gt) @ T1
            te1 = float(np.linalg.norm(dG1[:3, 3]))
            re1 = rotation_error_deg(dG1[:3, :3])
            g1 = gate(dg1["overlap_ratio"], dg1["mean_dist"])

            rows.append(dict(i=i, j=j, kind=kind, yaw_gt=y_gt, hint=psi,
                             hint_err=herr, sc_dist=d_sc,
                             ov0=dg0["overlap_ratio"], md0=dg0["mean_dist"],
                             te0=te0, re0=re0, gate0=bool(g0),
                             ov1=dg1["overlap_ratio"], md1=dg1["mean_dist"],
                             te1=te1, re1=re1, gate1=bool(g1)))
            print(f"({i:5d},{j:5d}) {kind:>5} {y_gt:7.1f} {psi:6.1f} "
                  f"{herr:6.1f} | {dg0['overlap_ratio']:5.2f} "
                  f"{dg0['mean_dist']:5.2f} {te0:5.2f} {re0:5.1f} "
                  f"{'Y' if g0 else 'n':>2} | {dg1['overlap_ratio']:5.2f} "
                  f"{dg1['mean_dist']:5.2f} {te1:5.2f} {re1:5.1f} "
                  f"{'Y' if g1 else 'n':>2}")

    def sel(k):
        return [r for r in rows if r["kind"] == k]

    print("\n--- summary ---")
    for k in ("TRUE", "FALSE"):
        rs = sel(k)
        te0 = [r["te0"] for r in rs]
        te1 = [r["te1"] for r in rs]
        re0 = [r["re0"] for r in rs]
        re1 = [r["re1"] for r in rs]
        print(f"{k}: dT err med {np.median(te0):.2f} -> {np.median(te1):.2f} m "
              f"(max {max(te0):.2f} -> {max(te1):.2f}) | "
              f"dR err med {np.median(re0):.1f} -> {np.median(re1):.1f} deg | "
              f"gate accepts {sum(r['gate0'] for r in rs)} -> "
              f"{sum(r['gate1'] for r in rs)}/{len(rs)}")
        flips = [(r["i"], r["j"], r["gate0"], r["gate1"])
                 for r in rs if r["gate0"] != r["gate1"]]
        if flips:
            print(f"  gate flips: {flips}")
    t = sel("TRUE")
    he = [abs(r["hint_err"]) for r in t]
    print(f"TRUE hint err vs GT yaw: med {np.median(he):.1f} deg, "
          f"max {max(he):.1f} deg")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
