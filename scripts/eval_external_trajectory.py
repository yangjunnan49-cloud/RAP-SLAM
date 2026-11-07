#!/usr/bin/env python3
"""Evaluate an external method's KITTI pose files against RAP baseline metrics.

Additive maintenance tooling for the 2026-09-19 horizontal comparison (HC-V1).
Reads 12-float KITTI-format pose files produced by an external pipeline and
evaluates them with the exact same metric code as the frozen core-baseline-v1
registry (eval_paper_metrics.evaluate_pair + kitti_drift), so numbers are
directly comparable.

Conventions (verified against kiss_icp 1.3.0 kitti dataloader and
rap_slam.real_dataset):
  * External pose files are in the CAMERA frame (kiss_icp apply_calibration:
    cam = Tr @ velo @ inv(Tr)).
  * Camera metrics: GT = raw KITTI poses file (already camera frame).
  * LiDAR-frame metrics: both sides inverse-transformed, velo = inv(Tr) @ cam @ Tr,
    matching load_gt_poses(apply_calib=True) = inv(Tr) @ T_cam @ Tr.

Frame counts must match the full sequences exactly; a mismatch aborts (no
patching, per protocol).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_paper_metrics as epm
from rap_slam import real_dataset

EXPECTED_FRAMES = {"00": 4541, "05": 2761, "06": 1101}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_sequence(pose_file: Path, gt_file: Path, calib_file: Path, seq: str,
                      rel_to: Path):
    est_cam = epm.load_kitti_format(str(pose_file))
    if len(est_cam) != EXPECTED_FRAMES[seq]:
        raise SystemExit(f"{seq}: pose rows {len(est_cam)} != "
                         f"{EXPECTED_FRAMES[seq]}; aborting (no patching)")
    gt_cam = list(real_dataset.load_gt_poses(str(gt_file), str(calib_file),
                                             apply_calib=False))
    m_cam = epm.evaluate_pair(gt_cam, est_cam)

    tr = real_dataset._read_calib_transform(str(calib_file))
    tr_inv = np.linalg.inv(tr)
    to_velo = lambda ps: [tr_inv @ p @ tr for p in ps]
    gt_velo = list(real_dataset.load_gt_poses(str(gt_file), str(calib_file),
                                              apply_calib=True))
    m_velo = epm.evaluate_pair(gt_velo, to_velo(est_cam))

    return {
        "pose_file": str(pose_file.resolve().relative_to(rel_to.resolve())),
        "pose_file_sha256": sha256(pose_file),
        "pose_frame": "camera",
        "n_frames": len(est_cam),
        "ape_unaligned_rmse": m_cam["ape_unaligned_rmse"],
        "ape_aligned_rmse": m_cam["ape_aligned_rmse"],
        "kitti_ter_pct": m_cam["kitti_ter_pct"],
        "kitti_rer_deg100m": m_cam["kitti_rer_deg100m"],
        "kitti_segments": m_cam["kitti_segments"],
        "kitti_ter_pct_lidar": m_velo["kitti_ter_pct"],
        "kitti_rer_deg100m_lidar": m_velo["kitti_rer_deg100m"],
        "kitti_segments_lidar": m_velo["kitti_segments"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--poses-dir", required=True,
                    help="directory containing kitti<seq>_poses_kitti.txt")
    ap.add_argument("--dataset-root", required=True,
                    help="KITTI odometry dataset root (contains poses/ and sequences/)")
    ap.add_argument("--sequences", default="00,05,06")
    ap.add_argument("--method-label", required=True)
    ap.add_argument("--short-label", default=None,
                    help="short display label for tables (default: method label)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", default="{}",
                    help="JSON string recorded verbatim into the output")
    args = ap.parse_args()

    poses_dir = Path(args.poses_dir)
    root = Path(args.dataset_root)
    seqs = [s.strip() for s in args.sequences.split(",")]
    sequences, gt_files = {}, {}
    out_path = Path(args.out)
    for seq in seqs:
        pose_file = poses_dir / f"kitti{seq}_poses_kitti.txt"
        gt_file = root / "poses" / f"{seq}.txt"
        calib_file = root / "sequences" / seq / "calib.txt"
        for p in (pose_file, gt_file, calib_file):
            if not p.exists():
                raise SystemExit(f"missing input: {p}")
        sequences[seq] = evaluate_sequence(pose_file, gt_file, calib_file, seq,
                                           rel_to=out_path.parent)
        gt_files[seq] = str(gt_file)

    out = {
        "method_label": args.method_label,
        "short_label": args.short_label or args.method_label,
        "provenance": json.loads(args.provenance),
        "metric_code": "eval_paper_metrics.evaluate_pair (core-baseline-v1 code path)",
        "runs": [{"sequences": sequences, "gt_files": gt_files}],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for seq in seqs:
        s = sequences[seq]
        print(f"{seq}: ATE_aligned {s['ape_aligned_rmse']:.4f} m | "
              f"ter_cam {s['kitti_ter_pct']:.3f}% | rer_cam {s['kitti_rer_deg100m']:.4f} | "
              f"ter_lidar {s['kitti_ter_pct_lidar']:.3f}% | frames {s['n_frames']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
