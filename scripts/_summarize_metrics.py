#!/usr/bin/env python3
"""Extract odometry/PGO APE-RPE from RAP-SLAM summary.json files.

Usage: python3 scripts/_summarize_metrics.py <output_dir> [<output_dir> ...]
Prints one line per dir: name, odometry APE/RPE, optimized APE/RPE, loops.
"""
import json
import os
import sys


def find_metrics(summary):
    """Locate the dict holding 'odometry' and 'optimized' evo blocks."""
    if "odometry" in summary and isinstance(summary["odometry"], dict):
        return summary
    for v in summary.values():
        if isinstance(v, dict) and "odometry" in v:
            return v
    return {}


def main():
    for d in sys.argv[1:]:
        path = os.path.join(d, "summary.json")
        name = os.path.basename(d.rstrip("/"))
        try:
            s = json.load(open(path))
        except Exception as e:
            print(f"{name:16s} ERR {e}")
            continue
        m = find_metrics(s)
        o, p = m.get("odometry", {}), m.get("optimized", {})
        loops = s.get("accepted_loops", s.get("n_accepted_loops", "?"))
        print(f"{name:16s} odom APE {o.get('ape_rmse', -1):8.3f}  RPE {o.get('rpe_rmse', -1):7.4f}"
              f"  ->  PGO APE {p.get('ape_rmse', -1):8.3f}  RPE {p.get('rpe_rmse', -1):7.4f}"
              f"  loops={loops}")


if __name__ == "__main__":
    main()
