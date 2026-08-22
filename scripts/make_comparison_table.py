"""Merge our paper metrics with literature baselines into the paper draft
comparison tables.

Inputs:
  - output_paper_metrics/paper_metrics.json   (our numbers, eval_paper_metrics.py)
  - data/literature/kitti_baselines.json      (transcribed literature numbers)

Outputs: <out>/comparison_table.md + <out>/comparison_table.json

Our canonical products per KITTI sequence (non-deskew front-end; the deskew
arm was falsified on KITTI, see README 2026-08-16 addendum):
  00 output_prods/kitti00_bb  (offline, loop budget 40)
  01 step3_kitti01            (ROS 2 live; highway, no revisits -> 0 loops)
  02 step3_kitti02_current    (live edges, backend re-run with current
                               defaults: relfloor+Huber, NMS w=10;
                               scripts/rerun_pgo_saved.py)
  03 step3_kitti03            (live; 0 loops exist)
  04 step3_kitti04            (live; 0 loops exist)
  05 output_prods/kitti05     (offline)
  06 output_prods/kitti06     (offline)

CPU-only. Run anywhere (no evo/gtsam needed):
    python scripts/make_comparison_table.py [--out output_paper_metrics]
"""

import argparse
import json
import os
from typing import Dict, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEQ_ORDER = ["00", "01", "02", "03", "04", "05", "06"]
ALL_SEQS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

# KITTI seq -> (paper_metrics key, kind)
OUR_PRODUCTS = {
    "00": ("prods_kitti00_bb", "offline"),
    "01": ("step3_kitti01", "live"),
    "02": ("step3_kitti02_current", "live"),
    "03": ("step3_kitti03", "live"),
    "04": ("step3_kitti04", "live"),
    "05": ("prods_kitti05", "offline"),
    "06": ("prods_kitti06", "offline"),
}

POSS_PRODUCTS = {
    "POSS seg-A (1419 fr, 420 m)": ("prods_poss", "offline"),
    "POSS seg-B (2751 fr, 844 m)": ("possb_smoke", "offline"),
}


def ours(metrics: Dict, seq: str, traj: str) -> Optional[Dict]:
    key, kind = OUR_PRODUCTS[seq]
    entry = metrics[kind].get(key)
    if entry is None or traj not in entry:
        return None
    return entry[traj]


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, nd=2, dash="–"):
    return dash if v is None else f"{v:.{nd}f}"


def build(metrics: Dict, lit: Dict) -> str:
    seqs_lit = lit["seqs"]
    idx = {s: i for i, s in enumerate(seqs_lit)}
    methods = lit["methods"]

    out = []
    out.append("# RAP-SLAM vs literature — comparison table draft")
    out.append("")
    out.append("Our numbers: `output_paper_metrics/paper_metrics.json` "
               "(eval_paper_metrics.py). Literature numbers transcribed from "
               "the cited papers (`data/literature/kitti_baselines.json`, "
               "accessed " + lit.get("_accessed", "?") + "; convention notes "
               "there). ter = KITTI official avg. relative translational "
               "drift [%], 100–800 m subsegments; rer = rotational drift "
               "[deg/100 m]. Aligned ATE = evo APE translation RMSE after "
               "SE(3) Umeyama alignment (literature 'ATE RMSE' convention).")
    out.append("")
    out.append("**Front-end note:** our odometry is KISS-ICP with a single "
               "POSS-tuned config (max_range 80 m, min_range 1.5 m, voxel "
               "0.8 m), not the KISS-ICP paper config — our odometry rows "
               "are therefore weaker than the published KISS-ICP (0.50 % "
               "mean ter on 00–10). All 'ours (full)' numbers add Scan "
               "Context + proximity retrieval, RAP geometric verification "
               "(overlap ≥ 0.7 & mean_dist ≤ 2.5 m gate), loop-edge NMS "
               "(w=10) and relfloor+Huber-weighted GTSAM PGO. No intrinsic "
               "angle correction is applied (several literature rows apply "
               "0.2°).")

    # ---------------- Table 1: ter per seq ----------------
    out.append("")
    out.append("## Table 1 — KITTI relative translational drift ter [%] "
               "(per sequence, train split)")
    out.append("")
    hdr = "| method | " + " | ".join(ALL_SEQS) + " | mean 00–06 |"
    out.append(hdr)
    out.append("|" + "---|" * (len(ALL_SEQS) + 2))

    def lit_row(name, m):
        ter = m.get("ter")
        cells = [fmt(ter[idx[s]]) if ter else "–" for s in ALL_SEQS]
        m6 = mean([ter[idx[s]] for s in SEQ_ORDER]) if ter else None
        return (f"| {name} | " + " | ".join(cells) + f" | {fmt(m6)} |")

    for name, m in methods.items():
        if m.get("ter"):
            tag = " †" if m["type"] == "slam" else ""
            out.append(lit_row(name + tag, m))
    for name, m in methods.items():
        if not m.get("ter") and m.get("mean_00_10_ter") is not None:
            out.append(f"| {name} (00–10 mean only: "
                       f"{m['mean_00_10_ter']:.2f}) | " +
                       " | ".join(["–"] * len(ALL_SEQS)) + " | – |")

    odom_cells, pgo_cells = [], []
    for s in ALL_SEQS:
        if s in OUR_PRODUCTS:
            mo = ours(metrics, s, "odometry")
            mp = ours(metrics, s, "pgo")
            odom_cells.append(fmt(mo["kitti_ter_pct"] if mo else None))
            pgo_cells.append(fmt(mp["kitti_ter_pct"] if mp else None))
        else:
            odom_cells.append("–")
            pgo_cells.append("–")
    odom_m6 = mean([ours(metrics, s, "odometry")["kitti_ter_pct"]
                    for s in SEQ_ORDER])
    pgo_m6 = mean([ours(metrics, s, "pgo")["kitti_ter_pct"]
                   for s in SEQ_ORDER])
    out.append(f"| **ours odometry** | " + " | ".join(odom_cells) +
               f" | **{fmt(odom_m6)}** |")
    out.append(f"| **ours full †** | " + " | ".join(pgo_cells) +
               f" | **{fmt(pgo_m6)}** |")
    out.append("")
    out.append("† = full SLAM with loop closure. Note: ter is a *local* "
               "(≤800 m) drift metric — loop closure mainly improves global "
               "consistency (Table 3); MULLS-SLAM is likewise ≈ MULLS-LO on "
               "ter. seq 01/03/04 contain no true revisits, so 'ours full' "
               "= odometry there by construction.")

    # ---------------- Table 2: rer per seq ----------------
    out.append("")
    out.append("## Table 2 — KITTI relative rotational drift rer "
               "[deg/100 m]")
    out.append("")
    out.append(hdr)
    out.append("|" + "---|" * (len(ALL_SEQS) + 2))
    for name, m in methods.items():
        if m.get("rer"):
            rer = m["rer"]
            cells = [fmt(rer[idx[s]], 2) for s in ALL_SEQS]
            m6 = mean([rer[idx[s]] for s in SEQ_ORDER])
            out.append(f"| {name} † | " + " | ".join(cells) +
                       f" | {fmt(m6)} |")
    odom_cells, pgo_cells = [], []
    for s in ALL_SEQS:
        if s in OUR_PRODUCTS:
            mo = ours(metrics, s, "odometry")
            mp = ours(metrics, s, "pgo")
            odom_cells.append(fmt(mo["kitti_rer_deg100m"], 2) if mo else "–")
            pgo_cells.append(fmt(mp["kitti_rer_deg100m"], 2) if mp else "–")
        else:
            odom_cells.append("–")
            pgo_cells.append("–")
    out.append(f"| **ours odometry** | " + " | ".join(odom_cells) +
               f" | **{fmt(mean([ours(metrics, s, 'odometry')['kitti_rer_deg100m'] for s in SEQ_ORDER]))}** |")
    out.append(f"| **ours full †** | " + " | ".join(pgo_cells) +
               f" | **{fmt(mean([ours(metrics, s, 'pgo')['kitti_rer_deg100m'] for s in SEQ_ORDER]))}** |")
    out.append("")

    # ---------------- Table 3: aligned ATE ----------------
    out.append("")
    out.append("## Table 3 — Global consistency: aligned ATE RMSE [m] "
               "(evo `--align`, SE(3) Umeyama)")
    out.append("")
    out.append("| dataset | odometry | **ours full** | loops "
               "accepted | reference points |")
    out.append("|---|---|---|---|---|")
    loops_note = {"00": "40/40 true (budget 40)",
                  "01": "0 (no revisits)",
                  "02": "182 (125 ≤10 m GT + 57 same-street)",
                  "03": "0 (no revisits)", "04": "0 (no revisits)",
                  "05": "12/12 true", "06": "12/12 true"}
    ref = {"00": "CT-ICP+LC: 6.22 → 0.66 m *mean* ATE after rigid fit "
                 "(KITTI-raw 00; different convention — mean not RMSE, "
                 "raw scans)"}
    for s in SEQ_ORDER:
        mo = ours(metrics, s, "odometry")
        mp = ours(metrics, s, "pgo")
        out.append(f"| KITTI-{s} | {fmt(mo['ape_aligned_rmse'])} | "
                   f"**{fmt(mp['ape_aligned_rmse'])}** | "
                   f"{loops_note.get(s, '')} | {ref.get(s, '')} |")
    for label, (key, kind) in POSS_PRODUCTS.items():
        e = metrics[kind].get(key)
        if not e:
            continue
        out.append(f"| {label} | {fmt(e['odometry']['ape_aligned_rmse'])} | "
                   f"**{fmt(e['pgo']['ape_aligned_rmse'])}** | "
                   f"{'4/4 true' if key == 'prods_poss' else '12/12 true'} | "
                   "no literature baseline on POSS |")
    out.append("")
    out.append("POSS (PKU campus, Velodyne-32e) has no published baselines; "
               "it serves as our cross-sensor generalization set.")

    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics",
                   default=os.path.join(REPO_ROOT, "output_paper_metrics",
                                        "paper_metrics.json"))
    p.add_argument("--lit",
                   default=os.path.join(REPO_ROOT, "data", "literature",
                                        "kitti_baselines.json"))
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "output_paper_metrics"))
    args = p.parse_args()

    metrics = json.load(open(args.metrics))
    lit = json.load(open(args.lit))
    md = build(metrics, lit)
    md_path = os.path.join(args.out, "comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    json_path = os.path.join(args.out, "comparison_table.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"products": OUR_PRODUCTS, "poss": POSS_PRODUCTS,
                   "literature": lit}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
