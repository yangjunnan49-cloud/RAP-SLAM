# RAP-SLAM — ROS 2 / RViz visualization (Phase 2)

Step 0–2 of the ROS 2 roadmap: Jazzy environment on WSL2 Ubuntu 24.04,
a standalone player node that streams KITTI/POSS `.bin` scans as
`sensor_msgs/PointCloud2`, and a KISS-ICP odometry node publishing
`nav_msgs/Odometry` + TF + `nav_msgs/Path` for live trajectory viewing.

## Layout

- `../scripts/install_ros2_jazzy.sh` — idempotent ROS 2 Jazzy installer.
  Auto-probes apt mirrors (TUNA → Aliyun → official) and the rosdistro GPG key
  sources, so it works behind a hostile network. Re-runnable; apt resumes.
- `pointcloud_player.py` — standalone rclpy node (no colcon build needed).
- `odometry_node.py` — KISS-ICP odometry node (Step 2). Reuses
  `rap_slam.odometry.KissOdometry` with the offline pipeline's parameters.
- `rviz/rap_slam.rviz` — RViz layout: grid + PointCloud2 on `/points_raw` +
  Path on `/odom_path`, fixed frame `odom`, orbit view.
- `run_step1.sh` — player + RViz only.
- `run_step2.sh` — player + odometry node + RViz (the full live demo).
- `verify_step2_headless.sh` — no-GUI regression check: plays a POSS segment
  through both nodes and diffs the live trajectory against
  `../output_poss/odom_poses.txt`.

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

## Step 2 — live KISS-ICP odometry

One-time dependency (system python3 already has rclpy via apt; KISS-ICP goes
into the same interpreter):

```bash
apt-get install -y python3-pip
pip install --break-system-packages --ignore-installed typing_extensions \
  -i https://pypi.tuna.tsinghua.edu.cn/simple 'kiss-icp==1.3.0' 'numpy<2'
```

Run the demo:

```bash
bash /mnt/e/githouse/RAP-SLAM/ros2/run_step2.sh
bash run_step2.sh --ros-args -p rate:=5.0 -p loop:=true      # args -> player
ODOM_ARGS="--ros-args -p queue_depth:=5" bash run_step2.sh   # args -> odometry
```

The node subscribes `/points_raw`, and publishes `/odom`
(`nav_msgs/Odometry`), `/odom_path` (`nav_msgs/Path`), TF `odom → base_link`
(dynamic) and `base_link → velodyne` (static identity). RViz fixed frame is
`odom`, so the green trajectory grows live during playback.

Odometry parameters: `topic` (`/points_raw`), `odom_frame` (`odom`),
`base_frame` (`base_link`), `lidar_frame` (`velodyne`), `max_range` (**80.0**),
`min_range` (1.5), `voxel_size` (0.8), `queue_depth` (1), `save_poses` (path
to a TUM pose log appended per scan; empty = off).

Headless verification (no GUI), plays N frames and diffs against the offline
pipeline:

```bash
bash /mnt/e/githouse/RAP-SLAM/ros2/verify_step2_headless.sh 300    # or 1419
# PASS example (full POSS segment): live frames 1419/1419,
# translation diff vs offline: max = 1e-6 m
```

Verified 2026-07-21 (WSL2): full 1419-frame POSS segment, live trajectory
numerically identical to the offline pipeline (max diff 1e-6 m); KISS-ICP
registration 18–45 ms/scan, real-time at the native 10 Hz playback.

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
  `RVIZ_SOFTWARE=1 bash run_step2.sh --ros-args -p loop:=true`
- **Step-2 dependencies live in the system python3** (`/usr/bin/python3`,
  where apt puts rclpy), NOT the conda env. pip needs
  `--break-system-packages` (PEP 668) and `--ignore-installed typing_extensions`
  (Debian ships 4.10 without a RECORD file, uninstall fails). Keep `numpy<2`
  so the apt-installed ROS python modules stay compatible.
- **`max_range` must be 80.0 to match the offline pipeline** —
  `run_pipeline.py` passes `--odom_max_range` default 80.0 while the
  `KissOdometry` wrapper itself defaults to 60.0. At 60.0 the live trajectory
  drifts ~0.7 m from the offline one over 300 frames.
- **Start the odometry node before the player** and wait for its
  "KISS-ICP odometry up" log line; otherwise the first few scans are missed
  and the trajectory shifts by that many frames (fixed sleep is not enough —
  import time varies). `verify_step2_headless.sh` shows the pattern.
- **Do not keep state in WSL `/tmp` across `wsl` invocations** — it can be
  wiped when the VM restarts. Use a repo dir (e.g. `output_ros2/`) or `/mnt/e`.
- Headless check (no GUI): run the player, then
  `ros2 topic hz /points_raw` / `ros2 topic echo --once --field width /points_raw`.

## Next

Step 3 — loop-closure + PGO nodes: Scan Context / proximity candidates, RAP
verification, GTSAM optimization, `map → odom` correction TF, and loop-edge
markers in RViz.
