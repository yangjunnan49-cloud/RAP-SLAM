# RAP-SLAM — ROS 2 / RViz visualization (Phase 2)

Step 0 + Step 1 of the ROS 2 roadmap: Jazzy environment on WSL2 Ubuntu 24.04
and a standalone player node that streams KITTI/POSS `.bin` scans as
`sensor_msgs/PointCloud2` for live viewing in RViz.

## Layout

- `../scripts/install_ros2_jazzy.sh` — idempotent ROS 2 Jazzy installer.
  Auto-probes apt mirrors (TUNA → Aliyun → official) and the rosdistro GPG key
  sources, so it works behind a hostile network. Re-runnable; apt resumes.
- `pointcloud_player.py` — standalone rclpy node (no colcon build needed).
- `rviz/rap_slam.rviz` — RViz layout: grid + PointCloud2 on `/points_raw`,
  fixed frame `velodyne`, orbit view.
- `run_step1.sh` — one-command demo: starts the player, then RViz (WSLg GUI).

## Step 0 — install + verify DDS

```bash
bash /mnt/e/githouse/RAP-SLAM/scripts/install_ros2_jazzy.sh
source /opt/ros/jazzy/setup.bash
# terminal A
ros2 run demo_nodes_cpp talker
# terminal B
ros2 run demo_nodes_py listener   # expect "I heard: [Hello World: N]"
```

## Step 1 — watch POSS scans in RViz

```bash
bash /mnt/e/githouse/RAP-SLAM/ros2/run_step1.sh
# extra player args are forwarded, e.g.:
bash run_step1.sh --ros-args -p rate:=5.0 -p loop:=true -p start:=3850 -p end:=4100
```

Player parameters: `data_dir` (default: POSS seq 00 velodyne dir), `topic`
(`/points_raw`), `frame_id` (`velodyne`), `rate` (Hz, default 10), `start`,
`end` (frame numbers), `loop` (bool).

## Notes / gotchas

- **Effective rate tops out around 10 Hz.** The version-proof `.tolist()`
  conversion (~63k points/scan) plus DDS serialization is the bottleneck.
  POSS was recorded at 10 Hz, so this is the native rate anyway.
- **QoS is Reliable on purpose.** RViz's default PointCloud2 subscription is
  Reliable; on localhost there is no loss to protect against. Real sensor
  drivers and rosbag playback use BestEffort — then RViz must be switched to
  BestEffort too, or the display stays silently empty.
- **Never `source /opt/ros/jazzy/setup.bash` under `set -u`** — it references
  unbound variables. `install_ros2_jazzy.sh` handles this internally.
- **WSLg window quirks:** do not hand-write `Window Geometry` /
  `QMainWindow State` blobs into `.rviz` configs — a stale/invalid state can
  leave the RViz window off-screen or zero-sized (taskbar icon present, no
  visible window). If a window is lost: focus it via the taskbar, then
  `Win+←` / `Win+→` to snap it back (or `Alt+Space` → `M` → arrow key → move
  the mouse). The shipped `rap_slam.rviz` deliberately omits geometry state.
  If the window is invisible even on a fresh launch (WSLg + OGRE presentation
  bug), relaunch with software rendering:
  `RVIZ_SOFTWARE=1 bash run_step1.sh --ros-args -p loop:=true`
- Headless check (no GUI): run the player, then
  `ros2 topic hz /points_raw` / `ros2 topic echo --once --field width /points_raw`.

## Next

Step 2 — wrap KISS-ICP as an odometry node publishing `nav_msgs/Odometry` +
`odom -> base_link` TF, so the trajectory grows live in RViz.
