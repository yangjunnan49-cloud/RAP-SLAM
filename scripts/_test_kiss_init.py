import logging, sys
import numpy as np
sys.path.insert(0, "/mnt/e/githouse/RAP-SLAM")
logging.basicConfig(level=logging.WARNING)
import glob
import open3d as o3d
from rap_slam import evaluate, sequence
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

scans = [np.asarray(o3d.io.read_point_cloud(p).points, dtype=np.float32)
         for p in sorted(glob.glob("/mnt/e/githouse/RAP-SLAM/output/sequence/scans/scan_*.ply"))]
gt = sequence.load_kitti_poses("/mnt/e/githouse/RAP-SLAM/output/sequence/gt_poses.txt")

for th in [2.0, 3.0, 4.0]:
    cfg = KISSConfig()
    cfg.data.max_range = 60.0; cfg.data.min_range = 1.5; cfg.data.deskew = False
    cfg.mapping.voxel_size = 0.8
    cfg.adaptive_threshold.initial_threshold = th
    o = KissICP(cfg)
    for s in scans:
        o.register_frame(s.astype(np.float64), np.zeros(len(s)))
        poses.append if False else None
    poses = [np.asarray(o.last_pose)]  # placeholder
    # rerun properly collecting poses
    o = KissICP(cfg)
    ps = []
    for s in scans:
        o.register_frame(s.astype(np.float64), np.zeros(len(s)))
        ps.append(np.asarray(o.last_pose).copy())
    gt_rel = evaluate.align_to_first_pose(list(gt)); od_rel = evaluate.align_to_first_pose(ps)
    m = evaluate.compute_ape_rpe(gt_rel, od_rel)
    e01 = np.linalg.inv(np.linalg.inv(gt_rel[0]) @ gt_rel[1]) @ (np.linalg.inv(od_rel[0]) @ od_rel[1])
    print(f"threshold={th}: APE rmse={m['ape_rmse']:.3f} max={m['ape_max']:.3f} RPE rmse={m['rpe_rmse']:.4f} "
          f"frame0-1 err={np.linalg.norm(e01[:3,3]):.3f} m", flush=True)
