#!/usr/bin/env bash
# Deskew evaluation batch (2026-08-16, owner-approved front-end upgrade).
# Same golden flags as _rerun_prods.sh, PLUS --deskew (azimuth-synthesized
# per-point timestamps + KISS-ICP motion compensation). Outputs go to NEW
# dirs output_deskew/<dataset>/ so they can be diffed against output_prods/*.
# KITTI-02 additionally gets a no-deskew offline baseline (no offline golden
# existed for it; the 27.387 m reference is from the LIVE stack).
set -u
source /root/miniconda3/bin/activate rap-inference
cd /mnt/e/githouse/RAP-SLAM
PY=/root/miniconda3/envs/rap-inference/bin/python
KITTI=/mnt/e/githouse/Datasets/data_odometry_velodyne

echo "=== [1/6] POSS segA (3850-5268) deskew ==="
$PY scripts/run_pipeline.py --source real --loop-sep 200 --deskew \
    --output ./output_deskew/poss > output_deskew/poss.log 2>&1
echo "poss exit=$?"

echo "=== [2/6] KITTI 06 (full) deskew ==="
$PY scripts/run_pipeline.py --source real --real-root $KITTI --real-seq 06 \
    --real-start 0 --real-end 1100 --real-apply-calib --loop-sep 200 --deskew \
    --output ./output_deskew/kitti06 > output_deskew/kitti06.log 2>&1
echo "kitti06 exit=$?"

echo "=== [3/6] KITTI 05 (full) deskew ==="
$PY scripts/run_pipeline.py --source real --real-root $KITTI --real-seq 05 \
    --real-start 0 --real-end 2760 --real-apply-calib --loop-sep 200 --deskew \
    --output ./output_deskew/kitti05 > output_deskew/kitti05.log 2>&1
echo "kitti05 exit=$?"

echo "=== [4/6] KITTI 00 (full, bigbudget 40) deskew ==="
$PY scripts/run_pipeline.py --source real --real-root $KITTI --real-seq 00 \
    --real-start 0 --real-end 4540 --real-apply-calib --loop-sep 200 \
    --max-candidates 40 --deskew \
    --output ./output_deskew/kitti00_bb > output_deskew/kitti00_bb.log 2>&1
echo "kitti00_bb exit=$?"

echo "=== [5/6] KITTI 02 (full) deskew ==="
$PY scripts/run_pipeline.py --source real --real-root $KITTI --real-seq 02 \
    --real-start 0 --real-end 4660 --real-apply-calib --loop-sep 200 --deskew \
    --output ./output_deskew/kitti02 > output_deskew/kitti02.log 2>&1
echo "kitti02 exit=$?"

echo "=== [6/6] KITTI 02 (full) NO deskew (offline baseline) ==="
$PY scripts/run_pipeline.py --source real --real-root $KITTI --real-seq 02 \
    --real-start 0 --real-end 4660 --real-apply-calib --loop-sep 200 \
    --output ./output_deskew/kitti02_off > output_deskew/kitti02_off.log 2>&1
echo "kitti02_off exit=$?"

echo "ALL_DONE"
