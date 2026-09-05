#!/usr/bin/env python
"""Fit the deployment ConformalGate models (S1a arm) on ALL labeled saved
candidates and serialize them to data/conformal_gate/{A1,A2}.json.

The evaluation (split-conformal, TEST-held metrics) lives in
scripts/eval_conformal_s1.py; here every dangerous sample calibrates the
deployment threshold (standard post-validation practice).

Run (WSL2 conda rap-inference):
  python scripts/fit_conformal_gate.py
"""

import json
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rap_slam.conformal_gate import ConformalGate  # noqa: E402

OUT = os.path.join(REPO_ROOT, "data/conformal_gate")
ARMS = {"A1": ["overlap", "mean_dist"],
        "A2": ["overlap", "mean_dist", "sc_d1", "sc_margin",
               "deg_loc_t", "deg_loc_r", "deg_scat"]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cands = json.load(open(os.path.join(REPO_ROOT,
                                        "output_s1/candidates.json")))
    os.makedirs(OUT, exist_ok=True)
    for name, feats in ARMS.items():
        gate = ConformalGate.fit(cands, feats)
        path = os.path.join(OUT, f"{name}.json")
        gate.save(path)
        # in-band sanity: acceptance behaviour on the same records
        n_da = n_d = n_sa = n_s = 0
        for c in cands:
            v = gate.accept(c)
            if v is None or c.get("label") not in ("safe", "dangerous"):
                continue
            if c["label"] == "dangerous":
                n_d += 1
                n_da += int(v)
            else:
                n_s += 1
                n_sa += int(v)
        print(f"[{name}] Q={gate.Q:.4f} (n_calib_d={gate.n_calib_d}) "
              f"in-band danger-acc {n_da}/{n_d}, safe recall "
              f"{n_sa}/{n_s} = {n_sa / max(n_s, 1):.3f} -> {path}")


if __name__ == "__main__":
    main()
