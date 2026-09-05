#!/usr/bin/env python
"""S0a — GNC-TLS loop-edge rejection vs the canonical relfloor+Huber backend.

Program context (AGENTS.md 2026-09-05 S0-S5 addendum): GTSAM's graduated
non-convexity optimizer (Yang et al. T-RO 2020) graduates each LOOP edge
between inlier/outlier with per-edge inlier thresholds derived from that
edge's own noise model, while prior/odometry factors are pinned via
knownInliers. Unlike Huber (bounded influence, never rejects), TLS can
fully reject the heavy-tailed same-street aliasing edges (KITTI-02 live:
57 accepted edges at GT distance 10-37 m, T_rel error median 2.65 m /
worst 15 m).

Arms per golden (canonical NMS width preserved from GOLDENS):
  canonical   — relfloor + Huber(k=1) [must reproduce stored references]
  gnc_a1      — GNC-TLS, relfloor sigmas (Arm 1: current noise model)
  gnc_a2      — GNC-TLS, calibrated sigmas (Arm 2: P2 conditional model,
                fit re-derived from output_calibrated/edge_dataset.json)
  gnc_a1_nms0 — dense live sets only: GNC-TLS with NMS OFF, testing
                whether GNC subsumes NMS (keeps all geometry)

Predeclared GATE (AGENTS.md 2026-09-05): S0a is adopted if the best GNC
arm reaches kitti02_live unaligned APE <= 24.0 m (canonical 26.018) or
aligned <= 3.8 m (canonical 4.035), AND >=5/7 goldens within +5% of
canonical, AND no sparse-loop golden (poss_off, poss_live, kitti06_off,
kitti05) worse by >2%.

Run (WSL2 conda rap-inference; gtsam, CPU):
  python scripts/eval_gnc_backend.py
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
from eval_calibrated_weighting import GOLDENS, load_poses_auto, fit_calibrated  # noqa: E402

logger = logging.getLogger("eval_gnc_backend")

# Stored canonical references (unaligned APE rmse [m], current backend):
# P0-closed addendum 2026-08-22 / P3a 2026-08-23.
CANONICAL_REF = {
    "poss_off": 1.821, "kitti06_off": 2.441, "kitti05": 3.417,
    "kitti00_bb": 10.920, "kitti02_live": 26.018, "kitti06_live": 2.581,
    "poss_live": 1.757,
}
SPARSE_GOLDENS = ("poss_off", "poss_live", "kitti06_off", "kitti05")
DENSE_LIVE = ("kitti02_live", "kitti06_live")

GNC_PROB = 0.99


def run_arm(name: str, arm: str, calib: dict) -> dict:
    odom_f, kf_f, loops_f, source, seq, nms_canon = GOLDENS[name]
    odom = load_poses_auto(os.path.join(REPO_ROOT, odom_f))
    keyframes = json.load(open(os.path.join(REPO_ROOT, kf_f)))
    loops = json.load(open(os.path.join(REPO_ROOT, loops_f)))
    for lc in loops:
        lc["T_rel"] = np.asarray(lc["T_rel"], dtype=float)

    nms_w = nms_canon
    weighting, kernel = "relfloor", "huber"
    if arm == "gnc_a1":
        kernel = "gnc_tls"
    elif arm == "gnc_a2":
        kernel, weighting = "gnc_tls", "calibrated"
    elif arm == "gnc_a1_nms0":
        kernel, nms_w = "gnc_tls", 0.0

    gnc_out = {}
    optimized = pose_graph.optimize_pose_graph(
        odom, keyframes, loops,
        loop_weighting=weighting, robust_kernel=kernel,
        loop_nms_width=nms_w, calib_params=calib if weighting == "calibrated"
        else None,
        gnc_prob=GNC_PROB,
        gnc_out=gnc_out if kernel.startswith("gnc") else None)
    full = pose_graph.propagate_to_full_trajectory(odom, keyframes, optimized)
    if source:
        gt = gt_for_live(source, seq, len(odom))
    else:
        gt = gt_from_summary(os.path.dirname(os.path.join(REPO_ROOT, odom_f)))
    m = evaluate_pair(gt, full)
    if gnc_out:
        m["n_rejected"] = gnc_out["n_rejected"]
        m["n_loop"] = gnc_out["n_loop"]
        m["loop_weight_min"] = float(np.min(gnc_out["loop_weights"])) \
            if len(gnc_out["loop_weights"]) else None
    return m


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    out_dir = os.path.join(REPO_ROOT, "output_gnc_backend")
    os.makedirs(out_dir, exist_ok=True)

    # Arm 2 noise model: re-derive the P2 calibrated fit (fit set only).
    edges = json.load(open(os.path.join(REPO_ROOT, "output_calibrated",
                                        "edge_dataset.json")))
    fit_edges = [e for e in edges
                 if e["ds"] in ("kitti02", "poss_off", "poss_live", "possb")
                 and np.isfinite(e["mean_dist"])]
    calib = fit_calibrated(fit_edges)
    logger.info(f"calibrated sigma model: a_t={calib['a_t']:.3f} "
                f"b_t={calib['b_t']:.3f} a_r={calib['a_r']:.5f} "
                f"b_r={calib['b_r']:.3f} (n={len(fit_edges)})")

    report = {}
    lines = ["# S0a — GNC-TLS backend vs canonical relfloor+Huber "
             f"(gnc_prob={GNC_PROB})", "",
             "APE unaligned / aligned [m]; rej = loop edges rejected "
             "(GNC weight < 0.5)", "",
             "| golden | n_edges | canonical | gnc_a1 (relfloor) | "
             "gnc_a2 (calibrated) | gnc_a1 NMS-off |",
             "|---|---|---|---|---|---|"]
    for name in GOLDENS:
        _, _, loops_f, _, _, _ = GOLDENS[name]
        n_edges = len(json.load(open(os.path.join(REPO_ROOT, loops_f))))
        row = {}
        arms = ["canonical", "gnc_a1", "gnc_a2"] + \
            (["gnc_a1_nms0"] if name in DENSE_LIVE else [])
        for arm in arms:
            m = run_arm(name, arm, calib)
            row[arm] = m
            logger.info(f"[{name}][{arm}] APE {m['ape_unaligned_rmse']:.3f} / "
                        f"{m['ape_aligned_rmse']:.3f} m"
                        + (f", rejected {m.get('n_rejected')}/{m.get('n_loop')}"
                           if "n_rejected" in m else ""))
        report[name] = row

        # canonical regression check against the stored reference
        ref = CANONICAL_REF[name]
        got = row["canonical"]["ape_unaligned_rmse"]
        ok = abs(got - ref) / ref < 0.005
        logger.info(f"[{name}] canonical regression vs {ref}: "
                    f"{got:.3f} -> {'OK' if ok else 'MISMATCH'}")

        def cell(a):
            r = row[a]
            s = f"{r['ape_unaligned_rmse']:.3f} / {r['ape_aligned_rmse']:.3f}"
            if "n_rejected" in r:
                s += f" (rej {r['n_rejected']})"
            return s

        lines.append(f"| {name} | {n_edges} | {cell('canonical')} | "
                     f"{cell('gnc_a1')} | {cell('gnc_a2')} | "
                     f"{cell('gnc_a1_nms0') if 'gnc_a1_nms0' in row else '—'} |")

    # Predeclared gate evaluation
    lines += ["", "## Gate (predeclared 2026-09-05)", ""]
    best_arms = [a for a in ("gnc_a1", "gnc_a2", "gnc_a1_nms0")]
    gate_rows = []
    for arm in best_arms:
        k02 = report["kitti02_live"].get(arm)
        if k02 is None:
            continue
        cond_02 = (k02["ape_unaligned_rmse"] <= 24.0
                   or k02["ape_aligned_rmse"] <= 3.8)
        n_ok5, sparse_bad = 0, []
        for g in GOLDENS:
            if arm not in report[g]:
                continue
            c = report[g]["canonical"]["ape_unaligned_rmse"]
            a = report[g][arm]["ape_unaligned_rmse"]
            if a <= 1.05 * c:
                n_ok5 += 1
            if g in SPARSE_GOLDENS and a > 1.02 * c:
                sparse_bad.append(g)
        gate_rows.append((arm, cond_02, n_ok5, sparse_bad))
        lines.append(f"- {arm}: kitti02_live unaligned "
                     f"{k02['ape_unaligned_rmse']:.3f} / aligned "
                     f"{k02['ape_aligned_rmse']:.3f} -> 02-criterion "
                     f"{'PASS' if cond_02 else 'FAIL'}; goldens within +5%: "
                     f"{n_ok5}/7; sparse regressions >2%: "
                     f"{sparse_bad or 'none'}")
    verdict = any(c and n >= 5 and not s for _, c, n, s in gate_rows)
    lines.append("")
    lines.append(f"**GATE {'PASS — absorb gnc arm as default candidate' if verdict else 'FAIL — flags stay OFF, Huber+relfloor stays default'}**")

    with open(os.path.join(out_dir, "gnc_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "gnc_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote {out_dir}/gnc_report.{{md,json}}")


if __name__ == "__main__":
    main()
