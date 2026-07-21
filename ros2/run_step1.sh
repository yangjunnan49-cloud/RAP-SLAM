#!/usr/bin/env bash
# One-command Step-1 demo: start the .bin player, wait, then open RViz.
# Ctrl-C (or closing RViz) stops both.
#
#   bash /mnt/e/githouse/RAP-SLAM/ros2/run_step1.sh
#   bash run_step1.sh --ros-args -p rate:=5.0 -p loop:=true   # extra args go to the player

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

python3 "$SCRIPT_DIR/pointcloud_player.py" "$@" &
PLAYER_PID=$!
trap 'kill "$PLAYER_PID" 2>/dev/null || true' EXIT

sleep 2
echo "Starting RViz (needs WSLg). Fixed frame: velodyne, topic: /points_raw"
rviz2 -d "$SCRIPT_DIR/rviz/rap_slam.rviz"
