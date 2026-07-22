# RAP-SLAM — ROS 2 / RViz visualization (Phase 2)

Step 0–3 of the ROS 2 roadmap: Jazzy environment on WSL2 Ubuntu 24.04,
a standalone player node that streams KITTI/POSS `.bin` scans as
`sensor_msgs/PointCloud2`, a KISS-ICP odometry node publishing
`nav_msgs/Odometry` + TF + `nav_msgs/Path`, and a loop-closure + PGO stack
(Scan Context / proximity candidates -> RAP verification -> GTSAM PGO)
publishing the `map -> odom` correction TF, a corrected `/map_path`, and
`/loop_edges` markers.

## Layout

- `../scripts/install_ros2_jazzy.sh` — idempotent ROS 2 Jazzy installer.
  Auto-probes apt mirrors (TUNA → Aliyun → official) and the rosdistro GPG key
  sources, so it works behind a hostile network. Re-runnable; apt resumes.
- `pointcloud_player.py` — standalone rclpy node (no colcon build needed).
- `odometry_node.py` — KISS-ICP odometry node (Step 2). Reuses
  `rap_slam.odometry.KissOdometry` with the offline pipeline's parameters.
- `rap_server.py` — GPU worker (Step 3): loads RAP + GTSAM once in the conda
  env and serves registration / PGO requests over a localhost TCP socket
  (length-prefixed pickle). NOT a ROS node — see the Step-3 section.
- `loop_closure_node.py` — loop-closure + PGO node (Step 3): streaming
  keyframe selection, Scan Context + proximity candidates, RAP verification
  via `rap_server.py`, GTSAM PGO, `map -> odom` correction TF.
- `rviz/rap_slam.rviz` — Step 1/2 layout (fixed frame `odom`).
- `rviz/rap_slam_step3.rviz` — Step 3 layout (fixed frame `map`, adds the
  `/map_path` and `/loop_edges` displays).
- `run_step1.sh` — player + RViz only.
- `run_step2.sh` — player + odometry node + RViz.
- `run_step3.sh` — rap_server + odometry + loop-closure + player + RViz
  (the full live SLAM demo).
- `verify_step2_headless.sh` — no-GUI regression: live odometry vs
  `../output_poss/odom_poses.txt`.
- `verify_step3_headless.sh` — no-GUI regression: full live loop-closure
  pipeline vs ground truth (APE before/after PGO).

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

## Step 3 — live loop closure + PGO

Architecture (two interpreters, one localhost socket):

- `rap_server.py` runs in the **conda env** (`rap-inference`, torch/CUDA):
  loads `RAPBackend` once and serves `register` (2-way RAP verification) and
  `pgo` (GTSAM optimization) requests over TCP `127.0.0.1:5557`,
  length-prefixed pickle. Not a ROS node.
- `loop_closure_node.py` runs in the **system python3** (rclpy): pairs
  `/points_raw` + `/odom` by stamp, streaming keyframe selection (same
  thresholds as the offline pipeline: 6 m / 12 deg), Scan Context descriptors
  per keyframe, proximity + SC candidates (same params as the offline POSS
  run: sep 200, dist 15 m, SC < 0.25), RAP verification in a worker thread
  (acceptance on overlap >= 0.25 only — never gate on odometry consistency),
  GTSAM PGO after each accepted loop.

Outputs: TF `map -> odom` correction (10 Hz, identity until the first loop),
`/map_path` (nav_msgs/Path, map frame, PGO-corrected trajectory),
`/loop_edges` (visualization_msgs/Marker LINE_LIST, cyan, map frame).
Path/marker publishers are TRANSIENT_LOCAL so RViz shows the latest state
whenever it (re)starts.

Run the demo:

```bash
bash /mnt/e/githouse/RAP-SLAM/ros2/run_step3.sh
bash run_step3.sh --ros-args -p rate:=5.0 -p loop:=true    # args -> player
LOOP_ARGS="--ros-args -p loop_sep:=100" bash run_step3.sh  # args -> loop node
SERVER_ARGS="--port 5558" bash run_step3.sh                # args -> rap_server
```

Headless verification (full POSS segment, ~4 min at 12 Hz playback):

```bash
bash /mnt/e/githouse/RAP-SLAM/ros2/verify_step3_headless.sh 1419 12
# PASS example (2026-07-22, WSL2): 1419/1419 frames,
# loops (0,1394) overlap 0.76 + (0,1415) overlap 0.86,
# live odometry APE 8.257 m (= offline), live PGO APE 1.757 m
# (offline batch: 1.821 m with 4 loops; live accepts the 2 streaming-
# reachable ones — j=1418 pairs need the batch-only "last scan" keyframe)
```

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
- **rclpy parameter typing is strict:** the player's `rate` is declared
  double, so `--ros-args -p rate:=15` fails with InvalidParameterTypeException
  (INTEGER vs DOUBLE). Always pass a decimal: `-p rate:=15.0`.
- **`pkill -f` self-match:** `pkill -f 'odometry_node.py|...'` inside
  `wsl bash -c "..."` kills the invoking shell itself (the pattern appears in
  its own command line), silently aborting the rest of the command. Use the
  bracket trick (`pkill -f 'odometry_[n]ode.py'`) or separate calls.
- **Playback headroom:** KISS-ICP registration grows to ~60 ms/scan late in
  the POSS segment; at 15 Hz playback (66 ms budget) the odometry node is one
  hiccup from dropping frames. Use 10–12 Hz for full-segment runs.
- Headless check (no GUI): run the player, then
  `ros2 topic hz /points_raw` / `ros2 topic echo --once --field width /points_raw`.
