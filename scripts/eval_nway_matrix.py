#!/usr/bin/env python
"""P3c — set-level N-way matrix: hierarchical N-way joint registration
across ALL golden datasets x multiple RAP seeds x replacement arms.

RC-4 evaluated the hierarchical N-way backend on KITTI-02 + POSS only.
This script completes the matrix: every golden dataset with loop edges,
three seeds (sampling variance), and THREE arms per seed:

  baseline : canonical pairwise PGO (relfloor + Huber + golden NMS) —
             regression-checked against the known authoritative numbers.
  hier     : for every cluster with >= --min-cluster members, replace each
             accepted edge's T_rel by the cluster-consistent relative
             inv(T_i) @ T_j from ONE register_multi call per cluster
             (trimmed to the --max-parts highest-degree scans).
  hier_gate: same, but a cluster's replacement is applied ONLY if the
             post-registration N-way diagnostics pass the hazard re-gate
             (median per-edge N-way mean_dist <= --gate-md, from the
             2026-08-09 submap eval) — a warped/wrong joint registration
             poisons all its edges, so gating is per-cluster.

Per cluster we record the failure-taxonomy fields: n_members -> trimmed,
points/scan, oracle SE(3) layout warp rmse vs GT, per-edge intrinsic
T_rel errors (pairwise vs N-way), and N-way overlap/mean_dist medians.

GPU required (register_multi). Run in the conda env:
  python scripts/eval_nway_matrix.py [--goldens all] [--seeds 42,43,44] \
      [--reuse-cache]

Products: output_nway_matrix/matrix_report.{md,json} + per-cluster npz
caches (reuse across seeds/arms with --reuse-cache).
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from rap_slam import evaluate, pose_graph  # noqa: E402
from eval_paper_metrics import gt_for_live, gt_from_summary  # noqa: E402
from eval_calibrated_weighting import GOLDENS, load_poses_auto  # noqa: E402
from eval_verifier_swap import ScanProvider  # noqa: E402
from eval_hier_nway import mine_clusters, rel_err, umeyama_se3  # noqa: E402

logger = logging.getLogger("eval_nway_matrix")

# Authoritative baseline APEs (unaligned / aligned, current backend) for the
# built-in regression check — see output_clique_weighting/clique_report.md.
BASELINE_REF = {
    "poss_off": (1.821, 0.665),
    "kitti06_off": (2.441, 0.869),
    "kitti05": (3.417, 1.129),
    "kitti00_bb": (10.920, 2.264),
    "kitti02_live": (26.018, 4.035),
    "kitti06_live": (2.581, 0.893),
    "poss_live": (1.757, 0.651),
}


def parse_args():
    p = argparse.ArgumentParser(description="P3c N-way matrix eval")
    p.add_argument("--goldens", default="all",
                   help="comma list of golden names or 'all'")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--min-cluster", type=int, default=3)
    p.add_argument("--max-parts", type=int, default=16)
    p.add_argument("--points-budget", type=int, default=30000)
    p.add_argument("--gate-md", type=float, default=2.0,
                   help="hazard re-gate: median N-way mean_dist <= this (m)")
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--output",
                   default=os.path.join(REPO_ROOT, "output_nway_matrix"))
    return p.parse_args()


def golden_scan_cfg(name, source, seq, odom_f) -> dict:
    """Dataset identity + frame offset for a golden (same logic as
    eval_map_consistency; POSS segA stream frame 0 = POSS frame 3850)."""
    if source:
        return {"source": source, "seq": seq,
                "offset": 3850 if source == "poss" else 0}
    d = os.path.dirname(os.path.join(REPO_ROOT, odom_f))
    c = json.load(open(os.path.join(d, "summary.json")))["config"]
    if c.get("real_seq") == "00" and not c.get("real_apply_calib"):
        return {"source": "poss", "seq": "A",
                "offset": int(c.get("real_start", 3850))}
    return {"source": "kitti", "seq": c["real_seq"],
            "offset": int(c.get("real_start", 0))}


def full_traj_metrics(odom_poses, keyframes, optimized, gt_rel):
    opt = evaluate.align_to_first_pose(
        pose_graph.propagate_to_full_trajectory(odom_poses, keyframes,
                                                optimized))
    m_un = evaluate.compute_ape_rpe(gt_rel, opt)["ape_rmse"]
    m_al = evaluate.compute_ape_rpe(gt_rel, opt, align=True)["ape_rmse"]
    return m_un, m_al


def canonical_pgo(odom_poses, keyframes, loops, nms):
    return pose_graph.optimize_pose_graph(
        odom_poses, keyframes, loops, robust_kernel="huber",
        loop_nms_width=nms)


def taxonomy_tag(n_members, n_trimmed, pps, warp_rmse):
    tags = []
    tags.append("trimmed" if n_trimmed < n_members else "full")
    # RC-4 safe regime: <=16 parts x <=4000 pts did not warp
    if n_trimmed <= 16 and pps <= 4000:
        tags.append("within-safe-regime")
    else:
        tags.append("outside-safe-regime")
    tags.append("warped" if warp_rmse >= 1.0 else "no-warp")
    return "+".join(tags)


def edge_cover_trim(members, edge_ids, loops, max_parts):
    """Trim a cluster to <= max_parts members maximizing the number of
    edges with BOTH endpoints kept (greedy set cover, degree tie-break).

    Degree-top-N trimming (RC-4) has a bipartite trap: at a two-pass
    revisit site the highest-degree nodes can all sit on the SAME pass,
    leaving zero intra-set edges to replace (kitti06_live: 97 members,
    degree-top-16 -> 0/114 edges coverable, and the one-sided batch
    warped 61.9 m). Greedy edge cover keeps the batch connected and
    maximizes replaced edges.
    """
    if len(members) <= max_parts:
        return sorted(members)
    deg = {}
    for k in edge_ids:
        for s in (loops[k]["i"], loops[k]["j"]):
            deg[s] = deg.get(s, 0) + 1
    chosen: List[int] = []
    chosen_set = set()
    remaining = set(members)
    while len(chosen) < max_parts and remaining:
        best, best_key = None, None
        for s in remaining:
            new_edges = sum(
                1 for k in edge_ids
                if (loops[k]["i"] == s and loops[k]["j"] in chosen_set)
                or (loops[k]["j"] == s and loops[k]["i"] in chosen_set))
            key = (new_edges, deg.get(s, 0))
            if best_key is None or key > best_key:
                best, best_key = s, key
        chosen.append(best)
        chosen_set.add(best)
        remaining.discard(best)
    return sorted(chosen)


def run_golden(name, seeds, args, out_dir):
    from rap_slam.rap_backend import RAPBackend

    odom_f, kf_f, loops_f, source, seq, nms = GOLDENS[name]
    odom = load_poses_auto(os.path.join(REPO_ROOT, odom_f))
    keyframes = json.load(open(os.path.join(REPO_ROOT, kf_f)))
    loops = json.load(open(os.path.join(REPO_ROOT, loops_f)))
    for lc in loops:
        lc["T_rel"] = np.asarray(lc["T_rel"], dtype=float)
    if source:
        gt = gt_for_live(source, seq, len(odom))
    else:
        gt = gt_from_summary(os.path.dirname(os.path.join(REPO_ROOT,
                                                          odom_f)))
    gt_rel = evaluate.align_to_first_pose(gt)
    m_odom = evaluate.compute_ape_rpe(
        gt_rel, evaluate.align_to_first_pose(odom))["ape_rmse"]

    # baseline canonical pairwise PGO + regression check
    opt = canonical_pgo(odom, keyframes, loops, nms)
    b_un, b_al = full_traj_metrics(odom, keyframes, opt, gt_rel)
    ref = BASELINE_REF.get(name)
    reg = (None if ref is None
           else (abs(b_un - ref[0]) < 0.02 and abs(b_al - ref[1]) < 0.02))
    logger.info(f"[{name}] odom {m_odom:.3f} | baseline {b_un:.3f}/"
                f"{b_al:.3f} (ref {ref}, regression {'PASS' if reg else 'DIFF'})")

    clusters = mine_clusters(loops)
    big = [c for c in clusters if len(c["members"]) >= args.min_cluster]
    logger.info(f"[{name}] {len(loops)} edges -> {len(clusters)} clusters, "
                f"{len(big)} with >= {args.min_cluster} members")

    scans = ScanProvider(golden_scan_cfg(name, source, seq, odom_f))
    needed = sorted({s for c in big for s in c["members"]})
    raw = {s: scans.raw(s) for s in needed}
    logger.info(f"[{name}] loaded {len(raw)} cluster scans")

    res = {"odometry_ape": m_odom, "baseline": {"un": b_un, "al": b_al,
                                                "regression": reg},
           "n_edges": len(loops), "n_clusters": len(clusters),
           "clusters": {}, "seeds": {}}
    rap = None
    for seed in seeds:
        t_seed = time.time()
        hier_loops = [dict(lc) for lc in loops]
        gate_loops = [dict(lc) for lc in loops]
        edge_stats = {}

        # Pass 1 — register_multi for all clusters. Nothing but inference
        # touches the backend here, so the per-seed RNG stream (RAP sampling
        # is stochastic) is independent of the gate-metric preprocessing in
        # pass 2 and identical across arms / cache reruns.
        per_cluster = {}
        for c in big:
            members = c["members"]
            trimmed = edge_cover_trim(members, c["edges"], loops,
                                      args.max_parts)
            n_coverable = sum(1 for k in c["edges"]
                              if loops[k]["i"] in trimmed
                              and loops[k]["j"] in trimmed)
            pps = min(4000, args.points_budget // len(trimmed))
            cache = os.path.join(out_dir, f"{name}_c{c['id']}_s{seed}.npz")
            if args.reuse_cache and os.path.exists(cache):
                z = np.load(cache)
                T_parts = [z["T_parts"][k] for k in range(len(z["T_parts"]))]
                assert [int(k) for k in z["members"]] == trimmed
                t_infer = -1.0
            else:
                if rap is None or rap.seed != seed:
                    rap = RAPBackend(seed=seed)
                T_parts, diag = rap.register_multi(
                    [raw[s] for s in trimmed], max_points_per_scan=pps)
                t_infer = diag["t_inference"]
                np.savez(cache, T_parts=np.stack(T_parts),
                         members=np.array(trimmed))
            per_cluster[c["id"]] = {
                "c": c, "members": members, "trimmed": trimmed,
                "n_coverable": n_coverable, "pps": pps,
                "T_parts": T_parts, "t_infer": t_infer}

        # Pass 2 — oracle warp, gate metrics (re-preprocesses parts, CPU +
        # encoder only), edge replacement, taxonomy.
        for cid, pc in per_cluster.items():
            c, members, trimmed = pc["c"], pc["members"], pc["trimmed"]
            pps, T_parts = pc["pps"], pc["T_parts"]
            idx = {s: k for k, s in enumerate(trimmed)}

            # oracle layout warp (GT)
            pos = np.array([T_parts[idx[s]][:3, 3] for s in trimmed])
            gtpos = np.array([gt[s][:3, 3] for s in trimmed])
            R_f, t_f = umeyama_se3(pos, gtpos)
            resid = np.linalg.norm((R_f @ pos.T).T + t_f - gtpos, axis=1)
            warp = float(np.sqrt(np.mean(resid ** 2)))

            # post-registration N-way diagnostics for the hazard re-gate:
            # preprocess each part the same way register_multi did, then
            # per-edge overlap / mean_dist on the N-way-aligned clouds.
            if rap is None or rap.seed != seed:  # all-cache runs skip pass 1
                rap = RAPBackend(seed=seed)
            ds_map = {s: rap.preprocess(raw[s], max_points=pps)[0]
                      for s in trimmed}
            nw_ov, nw_md = [], []
            for k in c["edges"]:
                i, j = loops[k]["i"], loops[k]["j"]
                if i in idx and j in idx:
                    T_nw = np.linalg.inv(T_parts[idx[i]]) @ T_parts[idx[j]]
                    ov, md = RAPBackend._alignment_metrics(
                        ds_map[i], ds_map[j], T_nw, thresh=1.0)
                    nw_ov.append(ov)
                    nw_md.append(md)
                    hier_loops[k]["T_rel"] = T_nw
                    te_n, re_n = rel_err(T_nw, gt[i], gt[j])
                    te_p, re_p = rel_err(loops[k]["T_rel"], gt[i], gt[j])
                    edge_stats[k] = {"pair": (i, j), "cluster": c["id"],
                                     "pw_te": te_p, "nw_te": te_n,
                                     "pw_re": re_p, "nw_re": re_n,
                                     "nw_ov": ov, "nw_md": md}
            med_md = float(np.median(nw_md)) if nw_md else float("inf")
            med_ov = float(np.median(nw_ov)) if nw_ov else 0.0
            gate_pass = med_md <= args.gate_md
            if gate_pass:
                for k in c["edges"]:
                    if k in edge_stats and edge_stats[k]["cluster"] == cid:
                        gate_loops[k]["T_rel"] = hier_loops[k]["T_rel"]
            tag = taxonomy_tag(len(members), len(trimmed), pps, warp)
            ckey = str(cid)
            if ckey not in res["clusters"]:
                res["clusters"][ckey] = {
                    "n_members": len(members), "n_edges": len(c["edges"]),
                    "span": [members[0], members[-1]],
                    "trimmed_to": len(trimmed), "taxonomy": {}}
            res["clusters"][ckey]["taxonomy"][str(seed)] = {
                "pps": pps, "warp_rmse": round(warp, 3),
                "nw_md_med": round(med_md, 3), "nw_ov_med": round(med_ov, 3),
                "gate": "PASS" if gate_pass else "FAIL", "tag": tag,
                "n_coverable": pc["n_coverable"],
                "t_infer": round(pc["t_infer"], 1)}
            logger.info(f"[{name}][s{seed}] cluster {ckey}: "
                        f"{len(members)}->{len(trimmed)} parts "
                        f"({pc['n_coverable']}/{len(c['edges'])} edges "
                        f"coverable), {pps} pts, "
                        f"warp {warp:.2f} m, nw md/ov med {med_md:.2f}/"
                        f"{med_ov:.2f} gate {'PASS' if gate_pass else 'FAIL'}")

        opt_h = canonical_pgo(odom, keyframes, hier_loops, nms)
        h_un, h_al = full_traj_metrics(odom, keyframes, opt_h, gt_rel)
        opt_g = canonical_pgo(odom, keyframes, gate_loops, nms)
        g_un, g_al = full_traj_metrics(odom, keyframes, opt_g, gt_rel)
        te_pw = [v["pw_te"] for v in edge_stats.values()]
        te_nw = [v["nw_te"] for v in edge_stats.values()]
        better = sum(1 for v in edge_stats.values()
                     if v["nw_te"] < v["pw_te"])
        med_pw = float(np.median(te_pw)) if te_pw else float("nan")
        med_nw = float(np.median(te_nw)) if te_nw else float("nan")
        logger.info(f"[{name}][s{seed}] hier {h_un:.3f}/{h_al:.3f} | "
                    f"hier_gate {g_un:.3f}/{g_al:.3f} | baseline "
                    f"{b_un:.3f}/{b_al:.3f} | edge T err pw med "
                    f"{med_pw:.2f} -> nw med {med_nw:.2f} "
                    f"(nw better {better}/{len(edge_stats)}) | "
                    f"{time.time() - t_seed:.0f} s")
        res["seeds"][str(seed)] = {
            "hier": {"un": h_un, "al": h_al},
            "hier_gate": {"un": g_un, "al": g_al},
            "n_edges_replaced": len(edge_stats),
            "edge_pw_te_med": float(np.median(te_pw)) if te_pw else None,
            "edge_nw_te_med": float(np.median(te_nw)) if te_nw else None,
            "edges": {str(k): {kk: (round(vv, 3) if isinstance(vv, float)
                                    else vv)
                               for kk, vv in v.items()}
                      for k, v in edge_stats.items()}}
    return res


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: "
                               "%(message)s", datefmt="%H:%M:%S")
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    names = list(GOLDENS) if args.goldens == "all" else args.goldens.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    report: Dict[str, dict] = {}
    for name in names:
        logger.info(f"================ {name} ================")
        report[name] = run_golden(name, seeds, args, out_dir)

    lines = ["# P3c — hierarchical N-way matrix (APE unaligned / aligned [m])",
             ""]
    hdr = "| golden | odom | baseline | " + " | ".join(
        f"hier s{s} | hier_gate s{s}" for s in seeds) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (4 + 2 * len(seeds)))
    for name, r in report.items():
        row = (f"| {name} | {r['odometry_ape']:.3f} | "
               f"{r['baseline']['un']:.3f} / {r['baseline']['al']:.3f} | ")
        row += " | ".join(
            f"{r['seeds'][str(s)]['hier']['un']:.3f} / "
            f"{r['seeds'][str(s)]['hier']['al']:.3f} | "
            f"{r['seeds'][str(s)]['hier_gate']['un']:.3f} / "
            f"{r['seeds'][str(s)]['hier_gate']['al']:.3f}" for s in seeds)
        lines.append(row + " |")
    lines.append("")
    lines.append("Per-cluster taxonomy (warp rmse vs GT, re-gate "
                 f"md<={args.gate_md}):")
    lines.append("")
    lines.append("| golden | cluster | members->trimmed | seed | warp [m] "
                 "| nw md/ov med | gate | tag |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, r in report.items():
        for cid, c in r["clusters"].items():
            for s in seeds:
                tx = c["taxonomy"].get(str(s))
                if tx is None:
                    continue
                lines.append(
                    f"| {name} | {cid} | {c['n_members']}->"
                    f"{c['trimmed_to']} | {s} | {tx['warp_rmse']:.2f} | "
                    f"{tx['nw_md_med']:.2f}/{tx['nw_ov_med']:.2f} | "
                    f"{tx['gate']} | {tx['tag']} |")

    with open(os.path.join(out_dir, "matrix_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "matrix_report.json"), "w") as f:
        json.dump({"args": vars(args), "goldens": report}, f, indent=1,
                  default=str)
    logger.info(f"Wrote {out_dir}/matrix_report.{{md,json}} in "
                f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
