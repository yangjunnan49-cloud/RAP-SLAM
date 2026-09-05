#!/usr/bin/env bash
# S0b — KITTI 0.205-deg intrinsic vertical-angle correction arm
# (AGENTS.md 2026-09-05 S0-S5 program). Runs the canonical OFFLINE pipeline
# with --apply-kitti-205 on KITTI 06/05/02/00, plus a matching uncorrected
# baseline for seq 02 (no saved offline 02 baseline exists; 00/05/06 baselines
# are the saved goldens output_prods/{kitti00_bb,kitti05,kitti06} — odometry
# side is budget-independent, so the budget-40 golden is a valid odom
# baseline). Sequential to avoid GPU contention on RAP verification.
#
# Detached usage (WSL, from repo root):
#   setsid bash scripts/run_s0b_kitti205.sh > output_prods/s0b_driver.log 2>&1 < /dev/null &
set -u
source /root/miniconda3/bin/activate rap-inference
cd /mnt/e/githouse/RAP-SLAM
ROOT=/mnt/e/githouse/Datasets/data_odometry_velodyne

run_seq() { # seq last_frame tag extra_flag
  local seq="$1" last="$2" tag="$3" flag="$4"
  if [ -s "output_prods/${tag}/summary.json" ]; then
    echo "[$(date +%T)] SKIP $tag (summary.json exists)"
    return 0
  fi
  echo "[$(date +%T)] START $tag"
  # shellcheck disable=SC2086
  python scripts/run_pipeline.py --source real --real-root "$ROOT" \
    --real-seq "$seq" --real-start 0 --real-end "$last" --real-apply-calib \
    $flag --loop-sep 200 --output "./output_prods/${tag}" \
    > "output_prods/${tag}.log" 2>&1
  echo "[$(date +%T)] DONE $tag rc=$?"
}

run_seq 06 1100 kitti06_205 --apply-kitti-205
run_seq 05 2760 kitti05_205 --apply-kitti-205
run_seq 02 4660 kitti02_205 --apply-kitti-205
run_seq 02 4660 kitti02_base ""
run_seq 00 4540 kitti00_205 --apply-kitti-205
echo "[$(date +%T)] ALL DONE"
