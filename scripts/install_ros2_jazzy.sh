#!/usr/bin/env bash
# Install ROS 2 Jazzy on Ubuntu 24.04 (WSL2 or native), China-network friendly.
#
# - Probes ROS 2 apt mirrors (TUNA -> Aliyun -> official) and picks the first
#   reachable one. Override with: ROS2_APT_MIRROR=https://... bash install_ros2_jazzy.sh
# - Idempotent: safe to re-run; apt resumes partial downloads.
# - Works as root (WSL default here) or via sudo.
#
# Installs: ros-base, rviz2, ros2cli, demo nodes (Step-0 verification),
# sensor_msgs_py (PointCloud2 helpers), python3-numpy, colcon (for later).

set -euo pipefail

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

echo "==> [1/6] Base tools + locale"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y curl gnupg lsb-release software-properties-common locales python3-numpy
$SUDO locale-gen en_US en_US.UTF-8 || true
$SUDO update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 || true
$SUDO add-apt-repository universe -y || true

echo "==> [2/6] Probing ROS 2 apt mirrors"
CANDIDATES=()
if [ -n "${ROS2_APT_MIRROR:-}" ]; then CANDIDATES+=("$ROS2_APT_MIRROR"); fi
CANDIDATES+=(
  "https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu"
  "https://mirrors.aliyun.com/ros2/ubuntu"
  "http://packages.ros.org/ros2/ubuntu"
)
MIRROR=""
for m in "${CANDIDATES[@]}"; do
  printf '    %s -> ' "$m"
  if curl -fsS -o /dev/null --max-time 10 "$m/dists/noble/Release"; then
    echo "OK"
    MIRROR="$m"
    break
  else
    echo "unreachable"
  fi
done
[ -n "$MIRROR" ] || { echo "ERROR: no ROS 2 apt mirror reachable. Check network/proxy."; exit 1; }

echo "==> [3/6] ROS GPG key"
KEY_OK=0
for k in \
  "https://mirrors.tuna.tsinghua.edu.cn/rosdistro/ros.key" \
  "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key"; do
  printf '    %s -> ' "$k"
  if curl -fsSL --max-time 15 "$k" -o /tmp/ros-archive-keyring.gpg 2>/dev/null; then
    echo "OK"
    $SUDO install -m 0644 /tmp/ros-archive-keyring.gpg /usr/share/keyrings/ros-archive-keyring.gpg
    KEY_OK=1
    break
  else
    echo "failed"
  fi
done
[ "$KEY_OK" -eq 1 ] || { echo "ERROR: could not download the ROS apt key from any source."; exit 1; }

echo "==> [4/6] apt source: $MIRROR"
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] $MIRROR noble main" \
  | $SUDO tee /etc/apt/sources.list.d/ros2.list > /dev/null
$SUDO apt-get update -y

echo "==> [5/6] Installing ROS 2 Jazzy packages (this is the big download)"
$SUDO apt-get install -y \
  ros-jazzy-ros-base \
  ros-jazzy-ros2cli \
  ros-jazzy-rviz2 \
  ros-jazzy-demo-nodes-cpp \
  ros-jazzy-demo-nodes-py \
  ros-jazzy-sensor-msgs-py \
  python3-colcon-common-extensions

echo "==> [6/6] Verify"
# ROS setup.bash references unbound vars; drop nounset while sourcing
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u
echo "    ROS_DISTRO=$ROS_DISTRO"
ros2 --help > /dev/null && echo "    ros2 CLI: OK"
command -v rviz2 > /dev/null && echo "    rviz2: OK (GUI via WSLg)"

echo ""
echo "Done. Next:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  # Step 0 check: terminal A: ros2 run demo_nodes_cpp talker"
echo "  #               terminal B: ros2 run demo_nodes_py listener"
echo "  # Step 1 viz:   bash /mnt/e/githouse/RAP-SLAM/ros2/run_step1.sh"
