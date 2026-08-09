import json

rows = json.load(open("output_submap_nway/regate_analysis.json"))
for name in ["poss", "kitti05", "kitti06_live"]:
    for ptype in ("raw", "submap"):
        sel = [r for r in rows if r["dataset"] == name and r["ptype"] == ptype]
        if not sel:
            continue
        print("== %s %s (%d edges) ==" % (name, ptype, len(sel)))
        for r in sorted(sel, key=lambda r: r["overlap"]):
            print("  %-8s (%5d,%5d) ov=%.3f md=%.3f GT_te=%.3f GT_re=%.2f"
                  % (r["kind"], r["i"], r["j"], r["overlap"],
                     r["mean_dist"], r["gt_te"], r["gt_re"]))
