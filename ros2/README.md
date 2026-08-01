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
- `run_step3_kitti.sh` / `verify_step3_kitti.sh` — KITTI odometry variants
  (GUI demo / headless verification) — see the "KITTI live runs" section.

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
`/loop_edges` (visualization_msgs/Marker LINE_LIST, cyan, map frame),
`/map_cloud` (sensor_msgs/PointCloud2, map frame — accumulated map rebuilt
from stride-downsampled scans at their latest corrected poses after every PGO
and every `map_period` scans; `map_voxel` 0.3 m, `map_stride` 5,
`map_max_points` 500000).
Path/marker/map publishers are TRANSIENT_LOCAL so RViz shows the latest state
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
# reachable ones — j=1418 pairs need the batch-only "last scan" keyframe),
# merged map 500k points rebuilt per PGO (~0.17 s), saved as
# output_ros2/step3/merged_map_live.ply
```

## KITTI live runs (Step 3 on KITTI odometry sequences)

The same 4-process Step-3 stack runs on the KITTI odometry velodyne tree at
`/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset/` (sequences 01–04,
scans + poses + calib verified complete). Nothing in the nodes is
dataset-specific; only the player's `data_dir` and the evaluation GT differ.
KITTI ground truth **must** use `apply_calib=True` (`T_velo = Tr^-1 @ T_cam @
Tr`; POSS used `False`).

- `verify_step3_kitti.sh <seq> [n_frames] [rate]` — headless verification,
  sibling of `verify_step3_headless.sh`. Products in
  `output_ros2/step3_kitti<seq>/`, logs `output_ros2/*_kitti<seq>.log` (POSS
  artifacts untouched). PASS criteria: all n odometry frames present,
  `/map_cloud` published, merged map >50k points, odometry APE < 10 m. Loop
  count is reported but never fails the run (KITTI 04 has no true loops);
  PGO APE > odometry APE with accepted loops prints a warning only.
  Env knobs: `LOOP_ARGS="--ros-args -p max_total:=120 ..."` injects extra
  loop-node params, `SAVE_SUFFIX=_tag` redirects products/logs to
  `step3_kitti<seq>_tag` / `*_kitti<seq>_tag.log` for side-by-side re-runs.
- `run_step3_kitti.sh <seq> [rate]` — GUI demo (same RViz step3 config),
  player pointed at the KITTI directory.

```bash
bash /mnt/e/githouse/RAP-SLAM/ros2/verify_step3_kitti.sh 04 271 12.0
bash /mnt/e/githouse/RAP-SLAM/ros2/verify_step3_kitti.sh 03        # full seq, 12 Hz
bash /mnt/e/githouse/RAP-SLAM/ros2/run_step3_kitti.sh 04 12.0      # GUI demo
```

Results (2026-07-23, WSL2, RTX 5070, rate 12 Hz, full sequences):

| Seq | Frames | Loops accepted | Odometry APE | PGO APE | Map points | Verdict |
|-----|--------|----------------|--------------|---------|------------|---------|
| 04  | 271/271  | 0 (no true loops) | **2.957 m** (= offline 2.96 m) | 2.957 m | 500000 | PASS |
| 03  | 801/801  | 0               | **4.428 m**  | 4.428 m | 500000 | PASS |
| 01  | 1101/1101 | 0              | **117.84 m** | 117.84 m | 500000 | FAIL (odometry drift, see below) |
| 02  | 4661/4661 | 0 (17 candidates, all correctly rejected) | **31.640 m** | 31.640 m | 500000 | FAIL (absolute criterion; see below) |

Seq 02 (long urban drive, 5.07 km path): the pipeline is healthy — all
4661 frames processed, 812 keyframes, and odometry drift is only 0.62% of
path length (31.6 m APE); the absolute <10 m PASS criterion simply does not
scale to a 5 km sequence. All 17 verified candidates were false positives
(GT distance 177–386 m) and RAP correctly rejected every one (overlap
0.03–0.11 < 0.25). Seq 02 **does contain true revisits** (27 GT pairs,
sep >= 200 scans, dist < 5 m: scans ~930–1030 ↔ ~4200–4270 and
~1800–1820 ↔ ~4400–4410) whose Scan Context distances are 0.087–0.172 —
well under the 0.25 threshold — yet they were never proposed, for two
independent reasons found by a tuned re-run
(`SAVE_SUFFIX=_bigbudget LOOP_ARGS="--ros-args -p max_total:=120 -p
sc_shortlist:=25" bash verify_step3_kitti.sh 02 4661 12.0`):

1. **Verification budget exhaustion.** `max_total=16` caps RAP checks per
   run; false SC hits burned all 16 by scan ~1832, long before the first
   revisit at scan ~4200. This default was tuned for the 1419-frame POSS
   segment and is too small for 4661-frame sequences.
2. **Ring-key shortlist misses the true match.** Even with the budget
   raised to 120 (121 proposals), the true pairs were never shortlisted —
   the top-N ring-key prefilter (N=10, even N=25) ranks repetitive urban
   geometry above the true revisit. SC retrieval recall needs revalidation
   off-POSS.

The tuned re-run also exposed a real hazard: with more verifications, RAP
**accepted 11 false loops** (overlap 0.29–0.50, all 98–547 m apart in GT;
odometry-consistency dT 92–572 m matched GT, so RAP's transforms were the
wrong ones), PGO APE blew up to 500 m, and CPU contention dropped 408
scans. Overlap-only acceptance (`min_overlap=0.25`), safe on POSS where
true loops scored 0.76–0.86, admits false loops on repetitive KITTI urban
streets scoring 0.35–0.49. Takeaway: seq-02-scale runs need (a) a larger or
adaptive verification budget, (b) SC shortlist/threshold revalidation
off-POSS, and (c) a second acceptance signal beyond raw overlap.

### Acceptance gate (fixed 2026-07-23)

Hazard (c) is fixed by a two-metric acceptance gate, driven by the
discriminator experiment `../scripts/eval_acceptance_gate.py` (8 GT-true
KITTI 02 revisit pairs + the 11 false pairs re-scored offline, plus the 2
hardest false hits from a live re-run):

| metric | TRUE loops (n=8 KITTI + 4 POSS) | FALSE loops (n=13) |
|--------|--------------------------------|---------------------|
| overlap | 0.75–0.96 | ≤ 0.64 |
| mean symmetric NN dist (m) | 0.39–1.45 | ≥ 1.26 (gross failures 3.7–22.1) |
| T_rel error vs GT | 0.15–4.3 m / ≤ 1.3° | 98–494 m |

Acceptance is now `overlap >= min_overlap AND mean_dist <= max_mean_dist`
with defaults **min_overlap 0.7 / max_mean_dist 2.5 m** (0 disables the
distance gate). Overlap carries the separation (gap 0.64 < t < 0.76);
mean_dist guards against catastrophic misalignments whose overlap spikes
(up to 0.50 observed live). Odometry consistency remains diagnostic-only.
`mean_dist` is computed alongside `overlap_ratio` in
`RAPBackend._alignment_metrics` and threaded through `rap_server.py`,
`loop_closure_node.py` (ROS params `min_overlap`, `max_mean_dist`), and
offline `rap_slam/loop_closure.py` + `run_pipeline.py --max-mean-dist`.

Regressions with the gate: POSS live re-run PASS (loops (0,1394)
0.76/1.00 m and (0,1415) 0.86/0.62 m accepted, PGO APE 1.757 m); KITTI 02
big-budget re-run (121 verifications): **0 false accepts** — including the
two clever urban misregistrations at overlap 0.62/0.64 — PGO APE stays at
the odometry value 31.640 m (the absolute <10 m script criterion still
FAILs on 5 km seq 02, expected); offline POSS pipeline reproduces the
baseline exactly (4/4 loops accepted, PGO APE 1.821 m). Seq 02's TRUE
loops remain unverified in the live system (retrieval shortlist issue
above); their measured overlaps 0.85–0.96 would pass the gate once
retrieval proposes them.

KITTI 01 is the highway sequence (~2.45 km path, fast and structureless).
The pipeline itself is healthy there: all 1101 frames processed, calib
correct (first-100-frame APE 0.42 m — a frame-conversion bug would break
immediately), RPE 0.037 m/step. The APE grows monotonically with distance
(0.42 m @100 fr → 5.9 m @300 → 35 m @600 → 117.8 m @1101, ~4.8% of path
length) — classic KISS-ICP drift on KITTI 01, not a live-pipeline defect.
No loop candidates fire (the trajectory never revisits a place), so PGO has
nothing to correct.

**Run these scripts as root** (`wsl -u root bash ...`): the conda env lives
under `/root/miniconda3`, and WSL sessions that default to a non-root user
fail with "Permission denied" on the rap_server launch.

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

## Streaming retrieval rework (2026-08-01, commit b39d116)

`loop_closure_node.py` no longer uses the ring-key shortlist. Retrieval is
now: full-scan vectorized SC matching (`scan_context.sc_distances_batch`)
against ALL earlier keyframes -> `rap_slam/cluster_gate.ClusterGate`
(strong SC < `sc_strong` 0.12 and proximity hits queue immediately and
uncapped; weaker SC hits need a `min_cluster`=2 (i,j)-cluster, capped at 2
promotions/cluster with 25-scan spacing) -> best-first `PriorityQueue` ->
RAP verification (overlap>=0.7 AND mean_dist<=2.5) -> GTSAM PGO (runs when
the queue empties OR `pgo_max_wait`=30 s after the first pending loop).
`max_total` now defaults to 0 (unlimited; the old hard budget was what
killed KITTI 02). Offline policy simulator: `scripts/eval_streaming_retrieval.py
--preset {kitti02,poss} [--keyframes-json auto] [--sweep]` (no GPU/ROS).

Live results: POSS regression PASS (2/2 loops, PGO APE 1.757 m = baseline);
KITTI 02 @10 Hz clean run: 182 accepted / 84 rejected, odom APE 31.640 m ->
PGO 27.387 m. Old policy on the same data proposed 0/37 true pairs.

New gotchas:
- **Odometry node drops frames under CPU contention** (`queue_depth=1` is
  deliberate). Dropped frames shift GT correspondence and make the verify
  script's APE meaningless (60 m artifact on KITTI 02). If a run shows
  `dropped>0` in the loop node log, re-run at 10 Hz; the drop-aware fallback
  is `scripts/eval_live_dropped.py` (coarse checks only — publish jitter
  defeats exact timestamp-to-frame recovery).
- **Verify scripts delete the previous run's products at startup**
  (`rm -f` accepted_loops.json / keyframes.json / map_poses.txt / PLY in
  `output_ros2/step3*/`). Copy anything you need before re-running.
- Drain wait: `DRAIN_SECS` env knob; the wait now counts `Loop worker idle`
  lines so a mid-playback idle cannot end it early. KITTI 02's backlog
  drains ~4 min after playback.
