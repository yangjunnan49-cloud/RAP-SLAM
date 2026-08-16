#!/usr/bin/env python3
"""Probe: which azimuth->timestamp convention (if any) does KISS-ICP deskew
want for a given dataset?

Context (2026-08-16): with the KISS-ICP kitti_raw convention
(yaw = -atan2(y, x), time increasing as the azimuth DECREASES), deskew improved
POSS odometry (APE 8.257 -> 7.040) but badly degraded KITTI-06
(2.777 -> 10.334, RPE doubled). This script runs a short segment under three
modes — off / neg (-atan2) / pos (+atan2) — and reports APE/RPE so the sign
(and the question whether the scans are already motion-compensated) can be
decided empirically per dataset.

Run inside WSL (conda env rap-inference):
  python scripts/eval_deskew_sign.py --dataset poss
  python scripts/eval_deskew_sign.py --dataset kitti06
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rap_slam import evaluate, odometry, real_dataset  # noqa: E402

logger = logging.getLogger("eval_deskew_sign")

DATASETS = {
    "poss": dict(root="/mnt/e/githouse/RAP-SLAM/data/poss", sequence_id="00",
                 start=3850, end=4050, apply_calib=False),
    "poss_full": dict(root="/mnt/e/githouse/RAP-SLAM/data/poss",
                      sequence_id="00", start=3850, end=5268,
                      apply_calib=False),
    "kitti06": dict(root="/mnt/e/githouse/Datasets/data_odometry_velodyne",
                    sequence_id="06", start=0, end=200, apply_calib=True),
    "kitti02": dict(root="/mnt/e/githouse/Datasets/data_odometry_velodyne",
                    sequence_id="02", start=1000, end=1200, apply_calib=True),
}

_ORIGINAL_TS_FN = odometry.synthesize_timestamps_from_azimuth


def _make_ts_fn(sign: float):
    def fn(scan: np.ndarray) -> np.ndarray:
        yaw = sign * np.arctan2(scan[:, 1], scan[:, 0])
        return 0.5 * (yaw / np.pi + 1.0)
    return fn


def run_segment(seq_kwargs: dict, mode: str, max_range: float) -> dict:
    data = real_dataset.load_real_sequence(**seq_kwargs)
    scans, gt = data["scans"], data["gt_poses"]
    if mode == "off":
        odom = odometry.KissOdometry(max_range=max_range, deskew=False)
    else:
        odometry.synthesize_timestamps_from_azimuth = _make_ts_fn(
            -1.0 if mode == "neg" else 1.0)
        odom = odometry.KissOdometry(max_range=max_range, deskew=True)
    try:
        poses = [odom.register(s) for s in scans]
    finally:
        odometry.synthesize_timestamps_from_azimuth = _ORIGINAL_TS_FN
    gt_a = evaluate.align_to_first_pose([np.asarray(T) for T in gt])
    est_a = evaluate.align_to_first_pose(poses)
    return evaluate.compute_ape_rpe(gt_a, est_a, align=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--max-range", type=float, default=80.0)
    p.add_argument("--modes", default="off,neg,pos")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)  # keep loader INFO noise down

    print(f"dataset={args.dataset} frames "
          f"{DATASETS[args.dataset]['start']}-{DATASETS[args.dataset]['end']}")
    for mode in args.modes.split(","):
        m = run_segment(DATASETS[args.dataset], mode, args.max_range)
        print(f"  {mode:4s}  APE rmse {m['ape_rmse']:8.3f} m   "
              f"RPE rmse {m['rpe_rmse']:7.4f} m")


if __name__ == "__main__":
    main()
