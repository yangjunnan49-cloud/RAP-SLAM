#!/usr/bin/env bash
# Headless Step-3 verification (no GUI): play a POSS segment through
# rap_server + odometry node + loop-closure node, then compare the live
# products against ground truth:
#   1. all N frames processed (map_poses.txt lines)
#   2. loop closures recovered (accepted_loops.json)
#   3. live odometry APE ~ offline 8.26 m, live PGO APE ~ offline 1.82 m
#
#   bash /mnt/e/githouse/RAP-SLAM/ros2/verify_step3_headless.sh [n_frames] [rate]
#   Env knobs: SAVE_SUFFIX=_deskew (separate output_ros2/step3_deskew/ + logs,
#   golden step3/ untouched), ODOM_ARGS="-p deskew:=true", DRAIN_SECS=240.
#
# Products land in output_ros2/step3/ (logs alongside in output_ros2/).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
N="${1:-1419}"
RATE="${2:-10}"
# rclpy parameter typing: `rate` is declared double, and a bare integer like
# `-p rate:=15` is rejected as INTEGER. Always pass a decimal form.
case "$RATE" in *.*) ;; *) RATE="$RATE.0" ;; esac
OUT_DIR="$REPO_ROOT/output_ros2"
# Optional SAVE_SUFFIX distinguishes re-runs (e.g. SAVE_SUFFIX=_deskew ->
# output_ros2/step3_deskew/, logs *_deskew.log) without deleting the golden.
SUFFIX="${SAVE_SUFFIX:-}"
SAVE_DIR="$OUT_DIR/step3$SUFFIX"
LOG_DIR="$OUT_DIR"
CONDA_PY=/root/miniconda3/envs/rap-inference/bin/python
mkdir -p "$SAVE_DIR"

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

# Last POSS frame index covered by the offline run: 3850 + 1419 - 1 = 5268.
END=$((3850 + N - 1))

rm -f "$SAVE_DIR/accepted_loops.json" "$SAVE_DIR/keyframes.json" \
      "$SAVE_DIR/map_poses.txt" "$SAVE_DIR/merged_map_live.ply" \
      "$LOG_DIR/live_poses_step3$SUFFIX.txt" \
      "$LOG_DIR/rap_server$SUFFIX.log" "$LOG_DIR/odom_node$SUFFIX.log" \
      "$LOG_DIR/loop_node$SUFFIX.log" "$LOG_DIR/player$SUFFIX.log"

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

# 1. GPU worker (model load ~30 s).
"$CONDA_PY" "$SCRIPT_DIR/rap_server.py" > "$LOG_DIR/rap_server$SUFFIX.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 120); do
  grep -q "RAP server listening" "$LOG_DIR/rap_server$SUFFIX.log" 2>/dev/null && break
  sleep 0.5
done
grep -q "RAP server listening" "$LOG_DIR/rap_server$SUFFIX.log" || {
  echo "FAIL: rap_server did not start; see $LOG_DIR/rap_server$SUFFIX.log"; exit 1; }

# 2. Odometry node. queue_depth=20: verification requires EVERY scan for
# GT-comparable eval; depth 1 drops frames whenever a register spike exceeds
# the frame period (observed up to 260 ms under the full Step-3 stack).
# Extra node args can be injected via ODOM_ARGS (later -p wins),
# e.g. ODOM_ARGS="-p deskew:=true".
# shellcheck disable=SC2086
python3 "$SCRIPT_DIR/odometry_node.py" \
  --ros-args -p save_poses:="$LOG_DIR/live_poses_step3$SUFFIX.txt" -p queue_depth:=20 ${ODOM_ARGS:-} \
  > "$LOG_DIR/odom_node$SUFFIX.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  grep -q "KISS-ICP odometry up" "$LOG_DIR/odom_node$SUFFIX.log" 2>/dev/null && break
  sleep 0.5
done

# 3. Loop-closure node.
python3 "$SCRIPT_DIR/loop_closure_node.py" \
  --ros-args -p save_dir:="$SAVE_DIR" \
  > "$LOG_DIR/loop_node$SUFFIX.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  grep -q "Loop-closure node up" "$LOG_DIR/loop_node$SUFFIX.log" 2>/dev/null && break
  sleep 0.5
done

# 4. Player.
python3 "$SCRIPT_DIR/pointcloud_player.py" --ros-args -p end:=$END -p rate:=$RATE \
  > "$LOG_DIR/player$SUFFIX.log" 2>&1 &
PIDS+=($!)

# Wait for playback to finish (RATE Hz + margin), then for the verification
# queue to drain (RAP checks still running after the last scan). Counts
# "Loop worker idle" lines so an idle message logged DURING playback does
# not end the wait early; DRAIN_SECS (default 240) bounds the wait.
sleep $((N / ${RATE%.*} + 15))
IDLE0=$(grep -c "Loop worker idle" "$LOG_DIR/loop_node$SUFFIX.log" 2>/dev/null || echo 0)
for _ in $(seq 1 "${DRAIN_SECS:-240}"); do
  IDLE1=$(grep -c "Loop worker idle" "$LOG_DIR/loop_node$SUFFIX.log" 2>/dev/null || echo 0)
  [ "$IDLE1" -gt "$IDLE0" ] && break
  sleep 1
done

# Stop the loop node first so it flushes its products on SIGTERM, then the rest.
kill "$(pgrep -f loop_closure_node.py)" 2>/dev/null || true
sleep 3
kill "${PIDS[@]}" 2>/dev/null || true
sleep 1

grep -q "map published" "$LOG_DIR/loop_node$SUFFIX.log" || {
  echo "FAIL: /map_cloud was never published"; exit 1; }

# Evaluate with the conda python (evo + the offline GT loader).
"$CONDA_PY" - "$SAVE_DIR" "$LOG_DIR/live_poses_step3$SUFFIX.txt" "$REPO_ROOT" "$N" <<'EOF'
import json
import os
import sys

import numpy as np

save_dir, odom_tum, repo_root, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, repo_root)
from rap_slam import evaluate, real_dataset  # noqa: E402

gt = real_dataset.load_gt_poses(
    os.path.join(repo_root, "data/poss/dataset/poses/00.txt"),
    os.path.join(repo_root, "data/poss/dataset/sequences/00/calib.txt"),
    start=3850, end=3850 + n - 1, apply_calib=False)
gt_rel = evaluate.align_to_first_pose(list(gt))


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

loops = json.load(open(os.path.join(save_dir, "accepted_loops.json")))
print(f"accepted loops ({len(loops)}): "
      f"{[(lc['i'], lc['j'], round(lc['overlap'], 2)) for lc in loops]}")
print("offline reference: [(0, 1394), (0, 1415), (0, 1418), (21, 1418)], APE 8.26 -> 1.82 m")
if len(loops) < 2:
    failures.append(f"too few loops accepted: {len(loops)}")

odom_poses = load_tum(odom_tum)
print(f"odometry frames: {len(odom_poses)} / expected {n}")
if len(odom_poses) != n:
    failures.append(f"dropped scans: {len(odom_poses)} != {n}")
m_odom = evaluate.compute_ape_rpe(gt_rel[:len(odom_poses)],
                                  evaluate.align_to_first_pose(odom_poses))
print(f"live odometry APE rmse = {m_odom['ape_rmse']:.3f} m (offline 8.257)")
if not 5.0 < m_odom["ape_rmse"] < 12.0:
    failures.append(f"odometry APE out of range: {m_odom['ape_rmse']:.3f}")

map_file = os.path.join(save_dir, "map_poses.txt")
if not os.path.exists(map_file):
    failures.append("map_poses.txt missing (loop node did not save)")
else:
    map_poses = load_tum(map_file)
    print(f"corrected frames: {len(map_poses)}")
    if len(map_poses) != len(odom_poses):
        failures.append(f"map/odom frame mismatch: {len(map_poses)} vs {len(odom_poses)}")
    m_opt = evaluate.compute_ape_rpe(gt_rel[:len(map_poses)],
                                     evaluate.align_to_first_pose(map_poses))
    print(f"live PGO APE rmse = {m_opt['ape_rmse']:.3f} m (offline 1.821)")
    if m_opt["ape_rmse"] > 3.0:
        failures.append(f"PGO APE too high: {m_opt['ape_rmse']:.3f}")
    if m_opt["ape_rmse"] >= m_odom["ape_rmse"]:
        failures.append("PGO did not improve over odometry")

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
print("PASS: live loop closure + PGO matches the offline pipeline")
EOF
