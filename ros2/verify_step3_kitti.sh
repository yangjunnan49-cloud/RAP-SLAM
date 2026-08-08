#!/usr/bin/env bash
# Headless Step-3 verification on a KITTI odometry sequence (no GUI):
# play velodyne scans through rap_server + odometry node + loop-closure
# node, then compare the live products against KITTI ground truth
# (apply_calib=True, aligned to the first pose).
#
#   bash /mnt/e/githouse/RAP-SLAM/ros2/verify_step3_kitti.sh <seq> [n_frames] [rate]
#   e.g.  bash verify_step3_kitti.sh 04 271 12.0
#
# Products land in output_ros2/step3_kitti<seq>/ (logs alongside in
# output_ros2/ with a _kitti<seq> suffix; POSS artifacts are untouched).
#
# PASS criteria (KITTI-tuned, NOT the POSS numbers):
#   - all n odometry frames present
#   - "map published" in the loop-node log
#   - merged_map_live.ply exists with >50k points
#   - live odometry APE rmse < 10.0 m
# Loop count is reported but never fails the run (KITTI 04 has no true
# loops). If loops were accepted and PGO APE > odometry APE a warning is
# printed (not a failure).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

SEQ="${1:?usage: verify_step3_kitti.sh <seq> [n_frames] [rate]}"
# Canonical full-sequence frame counts (verified against the dataset tree).
case "$SEQ" in
  00) N_FULL=4541 ;;
  01) N_FULL=1101 ;;
  02) N_FULL=4661 ;;
  03) N_FULL=801 ;;
  04) N_FULL=271 ;;
  05) N_FULL=2761 ;;
  06) N_FULL=1101 ;;
  *) echo "FAIL: unknown KITTI sequence '$SEQ'"; exit 1 ;;
esac
N="${2:-$N_FULL}"
RATE="${3:-12.0}"
# rclpy parameter typing: `rate` is declared double, and a bare integer like
# `-p rate:=15` is rejected as INTEGER. Always pass a decimal form.
case "$RATE" in *.*) ;; *) RATE="$RATE.0" ;; esac

KITTI_ROOT=/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset
DATA_DIR="$KITTI_ROOT/sequences/$SEQ/velodyne"
OUT_DIR="$REPO_ROOT/output_ros2"
# Optional SAVE_SUFFIX distinguishes re-runs (e.g. SAVE_SUFFIX=_bigbudget ->
# output_ros2/step3_kitti02_bigbudget/, logs *_kitti02_bigbudget.log).
SUFFIX="kitti$SEQ${SAVE_SUFFIX:-}"
SAVE_DIR="$OUT_DIR/step3_$SUFFIX"
LOG_DIR="$OUT_DIR"
CONDA_PY=/root/miniconda3/envs/rap-inference/bin/python
mkdir -p "$SAVE_DIR"

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

# Player frame range: 0 .. N-1.
END=$((N - 1))

rm -f "$SAVE_DIR/accepted_loops.json" "$SAVE_DIR/keyframes.json" \
      "$SAVE_DIR/map_poses.txt" "$SAVE_DIR/merged_map_live.ply" \
      "$LOG_DIR/live_poses_$SUFFIX.txt" \
      "$LOG_DIR/rap_server_$SUFFIX.log" "$LOG_DIR/odom_node_$SUFFIX.log" \
      "$LOG_DIR/loop_node_$SUFFIX.log" "$LOG_DIR/player_$SUFFIX.log"

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

# 1. GPU worker (model load ~30-40 s).
"$CONDA_PY" "$SCRIPT_DIR/rap_server.py" > "$LOG_DIR/rap_server_$SUFFIX.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 120); do
  grep -q "RAP server listening" "$LOG_DIR/rap_server_$SUFFIX.log" 2>/dev/null && break
  sleep 0.5
done
grep -q "RAP server listening" "$LOG_DIR/rap_server_$SUFFIX.log" || {
  echo "FAIL: rap_server did not start; see $LOG_DIR/rap_server_$SUFFIX.log"; exit 1; }

# 2. Odometry node.
python3 "$SCRIPT_DIR/odometry_node.py" \
  --ros-args -p save_poses:="$LOG_DIR/live_poses_$SUFFIX.txt" \
  > "$LOG_DIR/odom_node_$SUFFIX.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  grep -q "KISS-ICP odometry up" "$LOG_DIR/odom_node_$SUFFIX.log" 2>/dev/null && break
  sleep 0.5
done

# 3. Loop-closure node.
# 3. Loop-closure node. Extra node args can be injected via LOOP_ARGS,
# e.g. LOOP_ARGS="--ros-args -p max_total:=120" (must come after save_dir
# so an explicit -p save_dir there would win).
# shellcheck disable=SC2086
python3 "$SCRIPT_DIR/loop_closure_node.py" \
  --ros-args -p save_dir:="$SAVE_DIR" ${LOOP_ARGS:-} \
  > "$LOG_DIR/loop_node_$SUFFIX.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  grep -q "Loop-closure node up" "$LOG_DIR/loop_node_$SUFFIX.log" 2>/dev/null && break
  sleep 0.5
done

# 4. Player on the KITTI velodyne directory.
python3 "$SCRIPT_DIR/pointcloud_player.py" --ros-args \
  -p data_dir:="$DATA_DIR" -p start:=0 -p end:=$END -p rate:=$RATE \
  > "$LOG_DIR/player_$SUFFIX.log" 2>&1 &
PIDS+=($!)

# Wait for playback to finish (RATE Hz + margin), then for the verification
# queue to drain (RAP checks still running after the last scan). Note:
# "Loop worker idle" is only logged after at least one verification ran;
# if no candidates are ever proposed this wait simply times out.
# Drain detection counts "Loop worker idle" lines: an idle message logged
# DURING playback must not end the wait early (the priority queue can go
# momentarily idle mid-run and then refill). DRAIN_SECS (default 180)
# bounds the wait — the 2026-07-31 retrieval rework deliberately leaves a
# bounded backlog on long urban sequences (KITTI 02: ~3-5 min of drain).
sleep $((N / ${RATE%.*} + 15))
IDLE0=$(grep -c "Loop worker idle" "$LOG_DIR/loop_node_$SUFFIX.log" 2>/dev/null || true)
for _ in $(seq 1 "${DRAIN_SECS:-180}"); do
  IDLE1=$(grep -c "Loop worker idle" "$LOG_DIR/loop_node_$SUFFIX.log" 2>/dev/null || true)
  [ "$IDLE1" -gt "$IDLE0" ] && break
  sleep 1
done

# Stop the loop node first so it flushes its products on SIGTERM, then the rest.
kill "$(pgrep -f loop_closure_node.py)" 2>/dev/null || true
sleep 3
kill "${PIDS[@]}" 2>/dev/null || true
sleep 1

grep -q "map published" "$LOG_DIR/loop_node_$SUFFIX.log" || {
  echo "FAIL: /map_cloud was never published"; exit 1; }

# Evaluate with the conda python (evo + the offline GT loader).
# KITTI ground truth MUST use apply_calib=True (T_velo = Tr^-1 @ T_cam @ Tr).
"$CONDA_PY" - "$SAVE_DIR" "$LOG_DIR/live_poses_$SUFFIX.txt" "$REPO_ROOT" \
            "$SEQ" "$N" "$KITTI_ROOT" <<'EOF'
import json
import os
import sys

import numpy as np

(save_dir, odom_tum, repo_root, seq, n, kitti_root) = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
    int(sys.argv[5]), sys.argv[6])
sys.path.insert(0, repo_root)
from rap_slam import evaluate, real_dataset  # noqa: E402

gt = real_dataset.load_gt_poses(
    os.path.join(kitti_root, f"poses/{seq}.txt"),
    os.path.join(kitti_root, f"sequences/{seq}/calib.txt"),
    start=0, end=n - 1, apply_calib=True)
gt_rel = evaluate.align_to_first_pose(list(gt))
print(f"GT poses: {len(gt_rel)} (seq {seq}, apply_calib=True)")


def load_tum(path):
    rows = np.loadtxt(path)
    poses = []
    for r in rows:
        t = r[1:4]
        qx, qy, qz, qw = r[4:8]
        n_ = qx*qx + qy*qy + qz*qz + qw*qw
        s = 2.0 / n_
        xx, yy, zz = qx*qx*s, qy*qy*s, qz*qz*s
        xy, xz, yz = qx*qy*s, qx*qz*s, qy*qz*s
        wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s
        T = np.eye(4)
        T[:3, :3] = [[1-(yy+zz), xy-wz, xz+wy],
                     [xy+wz, 1-(xx+zz), yz-wx],
                     [xz-wy, yz+wx, 1-(xx+yy)]]
        T[:3, 3] = t
        poses.append(T)
    return poses


failures = []

loops_file = os.path.join(save_dir, "accepted_loops.json")
loops = json.load(open(loops_file)) if os.path.exists(loops_file) else []
print(f"accepted loops ({len(loops)}): "
      f"{[(lc['i'], lc['j'], round(lc['overlap'], 2)) for lc in loops]}")
# Loop count is informational only: KITTI 04 has no true loops.

odom_poses = load_tum(odom_tum)
print(f"odometry frames: {len(odom_poses)} / expected {n}")
if len(odom_poses) != n:
    failures.append(f"dropped scans: {len(odom_poses)} != {n}")
m_odom = evaluate.compute_ape_rpe(gt_rel[:len(odom_poses)],
                                  evaluate.align_to_first_pose(odom_poses))
print(f"live odometry APE rmse = {m_odom['ape_rmse']:.3f} m "
      f"(RPE {m_odom.get('rpe_rmse', float('nan')):.4f} m)")
if m_odom["ape_rmse"] >= 10.0:
    failures.append(f"odometry APE too high: {m_odom['ape_rmse']:.3f}")

map_file = os.path.join(save_dir, "map_poses.txt")
if not os.path.exists(map_file):
    print("WARNING: map_poses.txt missing (loop node did not save); "
          "skipping PGO eval")
else:
    map_poses = load_tum(map_file)
    print(f"corrected frames: {len(map_poses)}")
    if len(map_poses) != len(odom_poses):
        failures.append(f"map/odom frame mismatch: {len(map_poses)} vs {len(odom_poses)}")
    m_opt = evaluate.compute_ape_rpe(gt_rel[:len(map_poses)],
                                     evaluate.align_to_first_pose(map_poses))
    print(f"live PGO APE rmse = {m_opt['ape_rmse']:.3f} m")
    if loops and m_opt["ape_rmse"] > m_odom["ape_rmse"]:
        print(f"WARNING: loops were accepted but PGO APE "
              f"({m_opt['ape_rmse']:.3f}) > odometry APE "
              f"({m_odom['ape_rmse']:.3f})")

ply_file = os.path.join(save_dir, "merged_map_live.ply")
if not os.path.exists(ply_file):
    failures.append("merged_map_live.ply missing (map never built)")
else:
    with open(ply_file, "rb") as f:
        head = f.read(300).decode("ascii", errors="replace")
    n_pts = int(head.split("element vertex")[1].split()[0])
    print(f"merged map: {n_pts} points in merged_map_live.ply")
    if n_pts < 50000:
        failures.append(f"merged map too small: {n_pts} points")

if failures:
    print("FAIL: " + "; ".join(failures))
    sys.exit(1)
print(f"PASS: KITTI seq {seq} live run OK")
EOF
