#!/usr/bin/env bash
# Headless RViz screenshot capture: full Step-3 POSS run under Xvfb.
# Screenshots land in output_ros2/rviz_shots/. Do NOT use set -u (ROS setup.bash).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/output_ros2/rviz_capture"
SHOT_DIR="$REPO_ROOT/output_ros2/rviz_shots"
mkdir -p "$LOG_DIR" "$SHOT_DIR"
CONDA_PY=/root/miniconda3/envs/rap-inference/bin/python

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 2

PIDS=()
trap 'kill "${PIDS[@]}" "$XVFB_PID" 2>/dev/null || true' EXIT

shot() { import -window root "$SHOT_DIR/$1" 2>>"$LOG_DIR/import.log"; echo "shot $1 taken"; }

# 1. GPU worker (model load ~30 s)
"$CONDA_PY" "$SCRIPT_DIR/rap_server.py" > "$LOG_DIR/rap_server.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 240); do
  grep -q "RAP server listening" "$LOG_DIR/rap_server.log" 2>/dev/null && break
  sleep 0.5
done
grep -q "RAP server listening" "$LOG_DIR/rap_server.log" || { echo "rap_server failed"; exit 1; }

# 2. Odometry node
python3 "$SCRIPT_DIR/odometry_node.py" > "$LOG_DIR/odom_node.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  grep -q "KISS-ICP odometry up" "$LOG_DIR/odom_node.log" 2>/dev/null && break
  sleep 0.5
done

# 3. Loop-closure node (scratch product dir — baseline output_ros2/step3 untouched)
python3 "$SCRIPT_DIR/loop_closure_node.py" --ros-args -p save_dir:="$LOG_DIR/products" \
  > "$LOG_DIR/loop_node.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
  grep -q "Loop-closure node up" "$LOG_DIR/loop_node.log" 2>/dev/null && break
  sleep 0.5
done

# 4. RViz (software GL), give it time to render before playback
LIBGL_ALWAYS_SOFTWARE=1 rviz2 -d "$SCRIPT_DIR/rviz/rap_slam_step3.rviz" > "$LOG_DIR/rviz.log" 2>&1 &
PIDS+=($!)
sleep 25
shot shot_0_startup.png

# 5. Player (POSS default data dir, 15 Hz)
python3 "$SCRIPT_DIR/pointcloud_player.py" --ros-args -p rate:=15.0 > "$LOG_DIR/player.log" 2>&1 &
PIDS+=($!)

# Shot A: mid-run, drifted map (before any loop closure)
sleep 55
shot shot_A_middrift.png

# Shot B: first loop accepted + PGO + map rebuilt
for _ in $(seq 1 300); do grep -q "=> ACCEPT" "$LOG_DIR/loop_node.log" 2>/dev/null && break; sleep 0.5; done
for _ in $(seq 1 120); do grep -q "PGO done" "$LOG_DIR/loop_node.log" 2>/dev/null && break; sleep 0.5; done
sleep 8
shot shot_B_loop_closed.png

# Shot C: playback finished + final PGO (second loop) + final map
for _ in $(seq 1 300); do grep -q "Finished all scans" "$LOG_DIR/player.log" 2>/dev/null && break; sleep 1; done
for _ in $(seq 1 240); do
  n=$(grep -c "PGO done" "$LOG_DIR/loop_node.log" 2>/dev/null || echo 0)
  [ "$n" -ge 2 ] && break
  sleep 1
done
sleep 10
shot shot_C_final.png

echo "CAPTURE DONE"
ls -la "$SHOT_DIR"
