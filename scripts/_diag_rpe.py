import numpy as np, sys
sys.path.insert(0, "/mnt/e/githouse/RAP-SLAM")
from rap_slam import sequence, evaluate
gt = sequence.load_kitti_poses("/mnt/e/githouse/RAP-SLAM/output/sequence/gt_poses.txt")
od = sequence.load_kitti_poses("/mnt/e/githouse/RAP-SLAM/output/odom_poses.txt")
gt_rel = evaluate.align_to_first_pose(list(gt)); od_rel = evaluate.align_to_first_pose(list(od))
errs = []
for i in range(len(gt)-1):
    dT_gt = np.linalg.inv(gt_rel[i]) @ gt_rel[i+1]
    dT_od = np.linalg.inv(od_rel[i]) @ od_rel[i+1]
    e = np.linalg.inv(dT_gt) @ dT_od
    errs.append(np.linalg.norm(e[:3,3]))
errs = np.array(errs)
worst = np.argsort(errs)[::-1][:10]
print("worst RPE frames:", [(int(i), round(float(errs[i]),3)) for i in worst])
print("median RPE:", round(float(np.median(errs)),4))
pos_err = [np.linalg.norm(gt_rel[i][:3,3]-od_rel[i][:3,3]) for i in range(len(gt))]
print("APE by frame (every 10):", [round(float(pos_err[i]),2) for i in range(0,len(gt),10)])
