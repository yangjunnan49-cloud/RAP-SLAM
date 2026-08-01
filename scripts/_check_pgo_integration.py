"""Quick end-to-end check of the integrated pose_graph defaults (relfloor+huber)
on the saved KITTI-02 and POSS live products. Not a shipped artifact — used
once on 2026-08-01 to validate the RAP-CORE PGO integration."""
import json
import sys

sys.path.insert(0, ".")
from rap_slam import evaluate, real_dataset, pose_graph  # noqa: E402
from scripts.eval_pgo_robust import load_tum  # noqa: E402


def check(odom_path, save_dir, gt_args, expect):
    odom = load_tum(odom_path)
    kfs = json.load(open(f"{save_dir}/keyframes.json"))
    loops = json.load(open(f"{save_dir}/accepted_loops.json"))
    n = len(odom)
    gt = real_dataset.load_gt_poses(*gt_args, start=0 if "kitti" in save_dir else 3850,
                                    end=(n - 1) if "kitti" in save_dir else (3850 + n - 1),
                                    apply_calib="kitti" in save_dir)
    gt_rel = evaluate.align_to_first_pose(list(gt))
    opt = pose_graph.optimize_pose_graph(odom, kfs, loops)  # all defaults
    full = pose_graph.propagate_to_full_trajectory(odom, kfs, opt)
    m = evaluate.compute_ape_rpe(gt_rel[:n], evaluate.align_to_first_pose(full))
    tag = "KITTI-02" if "kitti" in save_dir else "POSS"
    print(f"{tag}: integrated-defaults APE rmse = {m['ape_rmse']:.3f} m "
          f"(expect {expect})")


KR = "/mnt/e/githouse/Datasets/data_odometry_velodyne/dataset"
check("output_ros2/live_poses_kitti02.txt", "output_ros2/step3_kitti02",
      (f"{KR}/poses/02.txt", f"{KR}/sequences/02/calib.txt"), "26.423")
check("output_ros2/live_poses_step3.txt", "output_ros2/step3",
      ("data/poss/dataset/poses/00.txt", None), "1.757")
