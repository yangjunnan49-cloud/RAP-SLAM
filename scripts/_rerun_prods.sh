#!/usr/bin/env bash
# One-shot: regenerate offline pipeline products (odom poses + keyframes.json
# + accepted_loops.json) for the Direction-A submap-N-way evaluation.
# Golden flags per AGENTS.md verified-results table; outputs go to NEW dirs
# output_prods/<dataset>/ (the golden output_* dirs are left untouched).
set -u
source /root/miniconda3/bin/activate rap-inference
cd /mnt/e/githouse/RAP-SLAM
PY=/root/miniconda3/envs/rap-inference/bin/python
KITTI=/mnt/e/githouse/Datasets/data_odometry_velodyne

echo "=== [1/4] POSS segA (3850-5268) ==="
$PY scripts/run_pipeline.py --source real --loop-sep 200 \
    --output ./output_prods/poss > output_prods/poss.log 2>&1
echo "poss exit=$?"

echo "=== [2/4] KITTI 06 (full) ==="
$PY scripts/run_pipeline.py --source real --real-root $KITTI --real-seq 06 \
    --real-start 0 --real-end 1100 --real-apply-calib --loop-sep 200 \
    --output ./output_prods/kitti06 > output_prods/kitti06.log 2>&1
echo "kitti06 exit=$?"

echo "=== [3/4] KITTI 05 (full) ==="
$PY scripts/run_pipeline.py --source real --real-root $KITTI --real-seq 05 \
    --real-start 0 --real-end 2760 --real-apply-calib --loop-sep 200 \
    --output ./output_prods/kitti05 > output_prods/kitti05.log 2>&1
echo "kitti05 exit=$?"

echo "=== [4/4] KITTI 00 (full, bigbudget 40) ==="
$PY scripts/run_pipeline.py --source real --real-root $KITTI --real-seq 00 \
    --real-start 0 --real-end 4540 --real-apply-calib --loop-sep 200 \
    --max-candidates 40 \
    --output ./output_prods/kitti00_bb > output_prods/kitti00_bb.log 2>&1
echo "kitti00_bb exit=$?"

echo "ALL_DONE"
