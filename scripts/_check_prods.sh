#!/usr/bin/env bash
cd /mnt/e/githouse/RAP-SLAM/output_prods
for d in poss kitti06 kitti05 kitti00_bb; do
  if [ ! -f "$d/summary.json" ]; then echo "=== $d === (not done)"; continue; fi
  echo "=== $d ==="
  ls "$d/" | tr '\n' ' '; echo
  /root/miniconda3/envs/rap-inference/bin/python - "$d" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1] + "/summary.json"))
print("candidates", s["n_loop_candidates"], "accepted", s["n_loops_accepted"],
      "keyframes", s["n_keyframes"])
print("odom APE %.3f -> PGO APE %.3f (RPE %.3f)" % (
    s["metrics"]["odometry"]["ape_rmse"], s["metrics"]["optimized"]["ape_rmse"],
    s["metrics"]["odometry"]["rpe_rmse"]))
import os
print("products:", os.path.exists(sys.argv[1] + "/keyframes.json"),
      os.path.exists(sys.argv[1] + "/accepted_loops.json"))
EOF
done
