"""P3a — clique down-weighting vs NMS vs raw dense edges.

Same inputs (saved odometry + keyframes + accepted loops), same relfloor +
Huber back-end; only the dense-cluster handling changes:

  nms          — current live canonical: greedy NMS w=10 (hard suppression)
  raw          — no cluster handling at all (pre-2026-08-08 behavior)
  clique_comp  — keep all edges, sigma scale = sqrt(component size),
                 Chebyshev-w=10 union-find components (transitive chaining)
  clique_deg   — keep all edges, sigma scale = sqrt(1 + local degree)

Test beds: the dense-cluster live sets (KITTI-06 114 edges at one revisit
site — the 2026-08-08 lesson; KITTI-02 182 edges; POSS 2) and the offline
goldens (POSS 4, KITTI-06 12, 05 12, 00_bb 40 — sparser; clique modes
should be near-no-ops there, POSS offline must stay bit-exact).

Run (WSL2 conda rap-inference; gtsam, CPU):
  python scripts/eval_clique_weighting.py
"""

import json
import logging
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from rap_slam import pose_graph  # noqa: E402
from eval_paper_metrics import evaluate_pair, gt_for_live, gt_from_summary  # noqa: E402
from eval_calibrated_weighting import GOLDENS, load_poses_auto  # noqa: E402

logger = logging.getLogger("eval_clique_weighting")

ARMS = {
    # arm: (nms_width, clique_width, clique_mode)
    "nms": (10.0, 0.0, None),
    "raw": (0.0, 0.0, None),
    "clique_comp": (0.0, 10.0, "component"),
    "clique_deg": (0.0, 10.0, "degree"),
}

# live sets are the dense test beds; offline goldens check for regressions
LIVE = ["kitti06_live", "kitti02_live", "poss_live"]
OFFLINE = ["poss_off", "kitti06_off", "kitti05", "kitti00_bb"]


def run(name: str, arm: str) -> dict:
    odom_f, kf_f, loops_f, source, seq, nms_canon = GOLDENS[name]
    nms_w, cliq_w, cliq_mode = ARMS[arm]
    if name in OFFLINE and arm == "nms":
        nms_w = 0.0  # offline canonical has no NMS
    odom = load_poses_auto(os.path.join(REPO_ROOT, odom_f))
    keyframes = json.load(open(os.path.join(REPO_ROOT, kf_f)))
    loops = json.load(open(os.path.join(REPO_ROOT, loops_f)))
    for lc in loops:
        lc["T_rel"] = np.asarray(lc["T_rel"], dtype=float)
    optimized = pose_graph.optimize_pose_graph(
        odom, keyframes, loops,
        loop_weighting="relfloor", robust_kernel="huber",
        loop_nms_width=nms_w, loop_clique_width=cliq_w,
        loop_clique_mode=cliq_mode or "component")
    full = pose_graph.propagate_to_full_trajectory(odom, keyframes,
                                                   optimized)
    if source:
        gt = gt_for_live(source, seq, len(odom))
    else:
        gt = gt_from_summary(os.path.dirname(os.path.join(REPO_ROOT,
                                                          odom_f)))
    return evaluate_pair(gt, full)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: "
                               "%(message)s", datefmt="%H:%M:%S")
    out_dir = os.path.join(REPO_ROOT, "output_clique_weighting")
    os.makedirs(out_dir, exist_ok=True)

    report = {}
    lines = ["# P3a — clique down-weighting vs NMS (APE unaligned / "
             "aligned [m])", ""]
    lines.append("| golden | n_edges | raw | nms (canonical) | "
                 "clique-comp √k | clique-deg √(1+deg) |")
    lines.append("|---|---|---|---|---|---|")
    for name in LIVE + OFFLINE:
        _, _, loops_f, _, _, _ = GOLDENS[name]
        n_edges = len(json.load(open(os.path.join(REPO_ROOT, loops_f))))
        row = {}
        for arm in ARMS:
            m = run(name, arm)
            row[arm] = m
            logger.info(f"[{name}][{arm}] APE "
                        f"{m['ape_unaligned_rmse']:.3f} / "
                        f"{m['ape_aligned_rmse']:.3f} m")
        report[name] = row
        lines.append(
            f"| {name} | {n_edges} | "
            f"{row['raw']['ape_unaligned_rmse']:.3f} / "
            f"{row['raw']['ape_aligned_rmse']:.3f} | "
            f"{row['nms']['ape_unaligned_rmse']:.3f} / "
            f"{row['nms']['ape_aligned_rmse']:.3f} | "
            f"{row['clique_comp']['ape_unaligned_rmse']:.3f} / "
            f"{row['clique_comp']['ape_aligned_rmse']:.3f} | "
            f"{row['clique_deg']['ape_unaligned_rmse']:.3f} / "
            f"{row['clique_deg']['ape_aligned_rmse']:.3f} |")
    lines.append("")
    lines.append("POSS live (2 edges) and offline goldens are sparse — "
                 "clique modes must be near-no-ops there (POSS offline "
                 "golden 1.821 m must stay bit-exact).")
    with open(os.path.join(out_dir, "clique_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "clique_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote {out_dir}/clique_report.{{md,json}}")


if __name__ == "__main__":
    main()
