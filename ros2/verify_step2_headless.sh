#!/usr/bin/env bash
# Headless Step-2 verification (no GUI): play a POSS segment through the
# player + odometry node, save the live TUM poses, and compare translations
# against the offline pipeline output (output_poss/odom_poses.txt).
#
#   bash /mnt/e/githouse/RAP-SLAM/ros2/verify_step2_headless.sh [n_frames]
#
# Expect: live frame count == n_frames and max translation diff ~ 0 (the node
# runs the same KISS-ICP config as the offline pipeline on the same scans).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
N="${1:-300}"
OUT_DIR="$REPO_ROOT/output_ros2"
OUT="$OUT_DIR/live_poses_step2.txt"
mkdir -p "$OUT_DIR"

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

# Last POSS frame index covered by the offline run: 3850 + 1419 - 1 = 5268.
END=$((3850 + N - 1))

rm -f "$OUT" "$OUT_DIR/odom_node.log" "$OUT_DIR/player.log"
python3 "$SCRIPT_DIR/odometry_node.py" --ros-args -p save_poses:="$OUT" \
  > "$OUT_DIR/odom_node.log" 2>&1 &
ODOM_PID=$!
# Wait until the node has actually created its subscription (imports take a
# few seconds; a fixed sleep races and drops the first scans).
for _ in $(seq 1 30); do
  grep -q "KISS-ICP odometry up" "$OUT_DIR/odom_node.log" 2>/dev/null && break
  sleep 0.5
done
python3 "$SCRIPT_DIR/pointcloud_player.py" --ros-args -p end:=$END \
  > "$OUT_DIR/player.log" 2>&1 &
PLAYER_PID=$!
trap 'kill "$PLAYER_PID" "$ODOM_PID" 2>/dev/null || true' EXIT

# 10 Hz playback + margin for startup/teardown.
sleep $((N / 10 + 10))
kill "$PLAYER_PID" "$ODOM_PID" 2>/dev/null || true
sleep 1

python3 - "$OUT" "$REPO_ROOT/output_poss/odom_poses.txt" "$N" <<'EOF'
import sys
import numpy as np

live_path, offline_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
live = np.loadtxt(live_path)[:, 1:4]                       # TUM: stamp tx ty tz ...
off = np.loadtxt(offline_path)[:n]                          # KITTI 3x4 rows
off_t = np.column_stack([off[:, 3], off[:, 7], off[:, 11]])
d = np.linalg.norm(live - off_t[: len(live)], axis=1)
print(f"live frames: {len(live)} / expected {n}")
print(f"translation diff vs offline (m): max={d.max():.6f} "
      f"mean={d.mean():.6f} final={d[-1]:.6f}")
if len(live) != n:
    sys.exit("FAIL: dropped scans (raise queue_depth or lower rate)")
if d.max() > 0.01:
    sys.exit("FAIL: live odometry diverges from offline pipeline")
print("PASS: live odometry matches the offline pipeline")
EOF
