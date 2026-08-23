#!/usr/bin/env python
"""P4 — pick Oxford Spires sequences by revisit structure from GT only.

For every downloaded gt-tum.txt: path length, duration, and revisit sites
(pose pairs closer than R m with time separation >= SEP s, clustered by
Chebyshev distance in index space, like the loop-candidate analysis).
We want: walking-scale path (0.3-3 km), >=1 clean revisit site, and
enough frames for a meaningful KISS-ICP drift.

Pure stdlib + numpy. Run on Windows managed python:
  python scripts/select_spires_sequence.py
"""

import glob
import os

import numpy as np

GT_DIR = r"E:\githouse\Datasets\oxford_spires\gt"
R_LIST = (3.0, 5.0)
SEP_S = 30.0          # min time separation for a revisit pair
CLUSTER_WIN = 50      # index-window for Chebyshev clustering of pairs


def load_tum(path):
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None, :]
    t = rows[:, 0]
    xyz = rows[:, 1:4]
    return t, xyz


def revisit_pairs(t, xyz, r, sep_s):
    n = len(t)
    # separation in frames ~ sep in seconds at ~constant rate
    dt = np.median(np.diff(t))
    sep_f = int(np.ceil(sep_s / dt))
    pairs = []
    for i in range(0, n - sep_f):
        d = np.linalg.norm(xyz[i + sep_f:] - xyz[i], axis=1)
        js = np.nonzero(d < r)[0]
        for joff in js[:8]:  # cap per-pose matches
            pairs.append((i, i + sep_f + int(joff)))
    return pairs


def cluster_pairs(pairs, win):
    """Greedy Chebyshev clustering in (i, j) index space."""
    clusters = []
    for p in sorted(pairs):
        placed = False
        for c in clusters:
            if abs(p[0] - c["ci"]) <= win and abs(p[1] - c["cj"]) <= win:
                c["pairs"].append(p)
                c["ci"] = int(np.mean([q[0] for q in c["pairs"]]))
                c["cj"] = int(np.mean([q[1] for q in c["pairs"]]))
                placed = True
                break
        if not placed:
            clusters.append({"ci": p[0], "cj": p[1], "pairs": [p]})
    return clusters


def main():
    files = sorted(glob.glob(os.path.join(GT_DIR, "*.txt")))
    print(f"{len(files)} GT files\n")
    hdr = (f"{'sequence':38s} {'frames':>6s} {'dur[s]':>7s} {'path[m]':>8s} "
           f"{'pairs<3m':>8s} {'sites<3m':>8s} {'pairs<5m':>8s} "
           f"{'sites<5m':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        name = os.path.basename(f)[:-4]
        try:
            t, xyz = load_tum(f)
        except Exception as e:
            print(f"{name:38s}  LOAD FAIL: {e}")
            continue
        if len(t) < 100:
            print(f"{name:38s}  too short ({len(t)} poses)")
            continue
        path = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
        dur = float(t[-1] - t[0])
        row = (f"{name:38s} {len(t):6d} {dur:7.0f} {path:8.0f} ")
        for r in R_LIST:
            pairs = revisit_pairs(t, xyz, r, SEP_S)
            cl = [c for c in cluster_pairs(pairs, CLUSTER_WIN)
                  if len(c["pairs"]) >= 3]
            row += f"{len(pairs):8d} {len(cl):8d} "
        print(row)
    print("\nsites = Chebyshev clusters (win 50 frames) with >=3 pairs; "
          "pairs capped at 8 per pose")


if __name__ == "__main__":
    main()
