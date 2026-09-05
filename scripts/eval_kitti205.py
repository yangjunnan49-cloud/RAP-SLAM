#!/usr/bin/env python
"""S0b — KITTI 0.205-deg intrinsic correction: odometry ter/APE deltas.

Baselines: saved goldens for 00/05/06 (odometry side is loop-budget
independent), a freshly-run uncorrected baseline for 02 (kitti02_base).
Corrected arms: output_prods/kitti*_205 (run_s0b_kitti205.sh).

Predeclared GATE (AGENTS.md 2026-09-05): adopt as a documented reporting
arm if ter improves on >=3 of {00,05,06,02} without odometry-APE
regression >2% on any of them. Canonical goldens are NEVER touched by
this arm (correction changes the input scans).

Run (WSL2 conda rap-inference, CPU):
  python scripts/eval_kitti205.py
"""

import json
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from eval_paper_metrics import evaluate_pair, gt_from_summary  # noqa: E402
from eval_calibrated_weighting import load_poses_auto  # noqa: E402

logger = logging.getLogger("eval_kitti205")

PAIRS = {  # seq: (baseline dir, corrected dir)
    "06": ("output_prods/kitti06", "output_prods/kitti06_205"),
    "05": ("output_prods/kitti05", "output_prods/kitti05_205"),
    "02": ("output_prods/kitti02_base", "output_prods/kitti02_205"),
    "00": ("output_prods/kitti00_bb", "output_prods/kitti00_205"),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    out_dir = os.path.join(REPO_ROOT, "output_kitti205")
    os.makedirs(out_dir, exist_ok=True)

    lines = ["# S0b — KITTI 0.205-deg intrinsic correction arm (odometry "
             "metrics)", "",
             "| seq | ter % base | ter % 205 | APE-u base | APE-u 205 | "
             "APE-a base | APE-a 205 |", "|---|---|---|---|---|---|---|"]
    report = {}
    n_ter_win, worst_ape_ratio = 0, 1.0
    for seq, (base_d, corr_d) in PAIRS.items():
        base_d = os.path.join(REPO_ROOT, base_d)
        corr_d = os.path.join(REPO_ROOT, corr_d)
        if not os.path.exists(os.path.join(corr_d, "odom_poses.txt")):
            logger.warning(f"[{seq}] missing {corr_d} — skipped")
            continue
        gt = gt_from_summary(corr_d)
        odom_b = load_poses_auto(os.path.join(base_d, "odom_poses.txt"))
        odom_c = load_poses_auto(os.path.join(corr_d, "odom_poses.txt"))
        mb = evaluate_pair(gt, odom_b)
        mc = evaluate_pair(gt, odom_c)
        report[seq] = {"base": mb, "corr": mc}
        lines.append(f"| {seq} | {mb['kitti_ter_pct']:.3f} | "
                     f"{mc['kitti_ter_pct']:.3f} | "
                     f"{mb['ape_unaligned_rmse']:.3f} | "
                     f"{mc['ape_unaligned_rmse']:.3f} | "
                     f"{mb['ape_aligned_rmse']:.3f} | "
                     f"{mc['ape_aligned_rmse']:.3f} |")
        logger.info(f"[{seq}] ter {mb['kitti_ter_pct']:.3f} -> "
                    f"{mc['kitti_ter_pct']:.3f} %, APE-u "
                    f"{mb['ape_unaligned_rmse']:.3f} -> "
                    f"{mc['ape_unaligned_rmse']:.3f} m")
        if mc["kitti_ter_pct"] < mb["kitti_ter_pct"]:
            n_ter_win += 1
        worst_ape_ratio = max(
            worst_ape_ratio,
            mc["ape_unaligned_rmse"] / max(mb["ape_unaligned_rmse"], 1e-9))

    n_eval = len(report)
    gate = (n_ter_win >= 3 and worst_ape_ratio <= 1.02)
    lines += ["", "## Gate (predeclared 2026-09-05)", "",
              f"ter improved on {n_ter_win}/{n_eval} sequences "
              f"(need >=3); worst odometry-APE ratio "
              f"{worst_ape_ratio:.3f} (need <=1.02)", "",
              f"**GATE {'PASS — document as the KITTI-convention reporting arm' if gate else 'FAIL — canonical goldens untouched, finding recorded'}**"]
    with open(os.path.join(out_dir, "kitti205_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "kitti205_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote {out_dir}/kitti205_report.{{md,json}}; "
                f"GATE {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
