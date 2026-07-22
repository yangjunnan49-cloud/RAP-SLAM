#!/usr/bin/env bash
# One-command Step-3 demo: rap_server (GPU worker) + KISS-ICP odometry node
# + loop-closure/PGO node + .bin player + RViz.
# The cyan loop edges appear when RAP-verified loops close, and the whole
# trajectory snaps into place via the map->odom correction TF.
# Ctrl-C (or closing RViz) stops everything.
#
#   bash /mnt/e/githouse/RAP-SLAM/ros2/run_step3.sh
#   bash run_step3.sh --ros-args -p rate:=5.0 -p loop:=true   # extra args -> player
#   ODOM_ARGS="..." LOOP_ARGS="..." SERVER_ARGS="--port 5558" bash run_step3.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/output_ros2"
mkdir -p "$LOG_DIR"
CONDA_PY=/root/miniconda3/envs/rap-inference/bin/python

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

# 1. GPU worker first — model load takes ~30 s; everything else waits on it.
# shellcheck disable=SC2086
"$CONDA_PY" "$SCRIPT_DIR/rap_server.py" ${SERVER_ARGS:-} \
  > "$LOG_DIR/rap_server.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 120); do
  grep -q "RAP server listening" "$LOG_DIR/rap_server.log" 2>/dev/null && break
  sleep 0.5
done
grep -q "RAP server listening" "$LOG_DIR/rap_server.log" || {
  echo "rap_server failed to start; see $LOG_DIR/rap_server.log"; exit 1; }

# 2. Odometry node (must be up before the player or the first scans are lost).
# shellcheck disable=SC2086
python3 "$SCRIPT_DIR/odometry_node.py" ${ODOM_ARGS:-} \
  > "$LOG_DIR/odom_node.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  grep -q "KISS-ICP odometry up" "$LOG_DIR/odom_node.log" 2>/dev/null && break
  sleep 0.5
done

# 3. Loop-closure node.
# shellcheck disable=SC2086
python3 "$SCRIPT_DIR/loop_closure_node.py" ${LOOP_ARGS:-} \
  > "$LOG_DIR/loop_node.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  grep -q "Loop-closure node up" "$LOG_DIR/loop_node.log" 2>/dev/null && break
  sleep 0.5
done

# 4. Player last (extra CLI args are forwarded to it).
python3 "$SCRIPT_DIR/pointcloud_player.py" "$@" \
  > "$LOG_DIR/player.log" 2>&1 &
PIDS+=($!)

sleep 1
# RVIZ_SOFTWARE=1 forces Mesa llvmpipe rendering. Use it when the RViz window
# is invisible/blank under WSLg (taskbar icon present but no window shown).
if [ "${RVIZ_SOFTWARE:-0}" = "1" ]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  echo "RVIZ_SOFTWARE=1 -> software OpenGL (llvmpipe), slower but robust"
fi
echo "Starting RViz (needs WSLg). Fixed frame: map"
echo "  gray cloud: /points_raw | green: /odom_path | cyan lines: /loop_edges"
rviz2 -d "$SCRIPT_DIR/rviz/rap_slam_step3.rviz"
