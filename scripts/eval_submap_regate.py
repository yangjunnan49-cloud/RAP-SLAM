#!/usr/bin/env python
"""Re-gate threshold analysis for Direction A submap-N-way (2026-08-09).

The strict two-metric re-gate (overlap >= 0.7 AND mean_dist <= 2.5) added to
eval_submap_nway.py after the kitti06_live warp disaster was calibrated on
RAW scans and misfires on SUBMAP parts (POSS: 1 of 4 good submap replacement
edges rejected, APE 1.705 -> 1.838). This script dumps, per candidate edge,
the measured (overlap, mean_dist) under the cluster-consistent T_rel so a
CATASTROPHIC-only threshold can be chosen on evidence: keep good submap
edges (POSS/KITTI-00) while still rejecting warped garbage (kitti06_live).

Reuses the npz caches in output_submap_nway/ (register_multi results) —
GPU-free. Run in the conda env:
  /root/miniconda3/envs/rap-inference/bin/python scripts/eval_submap_regate.py
"""

import argparse
import json
import logging
import os
import sys
import types

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_submap_nway as esn  # noqa: E402

logger = logging.getLogger("eval_submap_regate")


def parse_args():
    p = argparse.ArgumentParser(description="Submap-N-way re-gate analysis")
    p.add_argument("--datasets",
                   default="poss,kitti05,kitti00_bb,kitti06_live")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-parts", type=int, default=16)
    p.add_argument("--min-cluster", type=int, default=3)
    p.add_argument("--output", default=os.path.join(REPO_ROOT,
                                                    "output_submap_nway"))
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    # eval_submap_nway.Dataset needs a small args-like namespace
    dargs = types.SimpleNamespace(scan_min_range=1.5, scan_max_range=80.0,
                                  submap_half_width=4, submap_voxel=0.2,
                                  submap_max_points=30000)
    from rap_slam.rap_backend import RAPBackend
    rows = []
    for name in args.datasets.split(","):
        D = esn.Dataset(name, esn.DATASETS[name], dargs)
        big = [c for c in D.clusters if len(c["members"]) >= args.min_cluster]
        members = sorted({s for c in big for s in c["members"]})
        rng = np.random.default_rng(0)
        submaps = {m: D.build_submap(m, 4, 0.2, 30000, rng) for m in members}
        D.drop_scan_cache()
        for c in big:
            memb, deg = c["members"], c["deg"]
            trimmed = memb
            if len(memb) > args.max_parts:
                trimmed = sorted(sorted(memb, key=lambda s: -deg[s])
                                 [:args.max_parts])
            idx = None
            for ptype in ("raw", "submap"):
                cache = os.path.join(
                    args.output,
                    f"{name}_{ptype}_c{c['id']}_seed{args.seed}.npz")
                if not os.path.exists(cache):
                    continue
                z = np.load(cache)
                T_parts = [z["T_parts"][k] for k in range(len(z["T_parts"]))]
                assert [int(k) for k in z["members"]] == trimmed
                idx = {s: k for k, s in enumerate(trimmed)}
                ds_parts = {s: (esn.prep_for_metrics(D.load_scan(s))
                                if ptype == "raw"
                                else esn.prep_for_metrics(submaps[s]))
                            for s in trimmed}
                # accepted edges with both endpoints in the batch
                for k in c["edges"]:
                    i, j = D.loops[k]["i"], D.loops[k]["j"]
                    if i not in idx or j not in idx:
                        continue
                    T_new = (np.linalg.inv(T_parts[idx[i]])
                             @ T_parts[idx[j]])
                    ov, md = RAPBackend._alignment_metrics(
                        ds_parts[i], ds_parts[j], T_new, thresh=1.0)
                    te, re = esn.rel_err(T_new, D.gt[i], D.gt[j])
                    rows.append(dict(dataset=name, cluster=c["id"],
                                     ptype=ptype, kind="replace", i=i, j=j,
                                     overlap=ov, mean_dist=md,
                                     gt_te=te, gt_re=re))
                # clique candidates (sep >= 200, not already accepted)
                have = {tuple(sorted((e["i"], e["j"]))) for e in D.loops}
                for ai in range(len(trimmed)):
                    for bi in range(ai + 1, len(trimmed)):
                        i, j = trimmed[ai], trimmed[bi]
                        if j - i < 200 or (i, j) in have:
                            continue
                        T_new = (np.linalg.inv(T_parts[idx[i]])
                                 @ T_parts[idx[j]])
                        ov, md = RAPBackend._alignment_metrics(
                            ds_parts[i], ds_parts[j], T_new, thresh=1.0)
                        te, re = esn.rel_err(T_new, D.gt[i], D.gt[j])
                        rows.append(dict(dataset=name, cluster=c["id"],
                                         ptype=ptype, kind="clique", i=i, j=j,
                                         overlap=ov, mean_dist=md,
                                         gt_te=te, gt_re=re))
        D.drop_scan_cache()

    print("\n===== RE-GATE METRICS (seed 42) =====")
    hdr = f"{'dataset':12s} {'ptype':6s} {'kind':8s} {'pair':>15s} " \
          f"{'ov':>6s} {'md':>7s} {'GT_te':>7s} {'GT_re':>6s}"
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["dataset"], r["ptype"],
                                         r["kind"], r["i"], r["j"])):
        print(f"{r['dataset']:12s} {r['ptype']:6s} {r['kind']:8s} "
              f"{str((r['i'], r['j'])):>15s} {r['overlap']:6.3f} "
              f"{r['mean_dist']:7.3f} {r['gt_te']:7.3f} {r['gt_re']:6.2f}")
    # separability summary
    print("\n===== SEPARATION SUMMARY =====")
    for name in args.datasets.split(","):
        for ptype in ("raw", "submap"):
            sel = [r for r in rows if r["dataset"] == name
                   and r["ptype"] == ptype]
            if not sel:
                continue
            good = [r for r in sel if r["gt_te"] < 5.0]
            bad = [r for r in sel if r["gt_te"] >= 5.0]
            print(f"{name:12s} {ptype:6s}: n={len(sel)} "
                  f"good(GT<5m)={len(good)} bad(GT>=5m)={len(bad)}")
            for label, grp in (("good", good), ("bad", bad)):
                if grp:
                    ovs = [r["overlap"] for r in grp]
                    mds = [r["mean_dist"] for r in grp]
                    tes = [r["gt_te"] for r in grp]
                    print(f"  {label}: ov [{min(ovs):.3f}..{max(ovs):.3f}] "
                          f"md [{min(mds):.2f}..{max(mds):.2f}] "
                          f"GT_te [{min(tes):.2f}..{max(tes):.2f}]")
    out = os.path.join(args.output, "regate_analysis.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
