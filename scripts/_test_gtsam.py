import numpy as np
print("numpy", np.__version__)
import gtsam
print("gtsam imported")
from gtsam import Pose3, Rot3, Point3, Values, NonlinearFactorGraph, noiseModel
T = np.eye(4)
T[0,3] = 1.0
print("making Pose3 from numpy...")
p = Pose3(T)
print("Pose3 ok:", p.translation())
graph = NonlinearFactorGraph()
initial = Values()
prior = noiseModel.Diagonal.Sigmas(np.array([0.01]*6))
from gtsam import PriorFactorPose3, BetweenFactorPose3
graph.add(PriorFactorPose3(0, Pose3(np.eye(4)), prior))
print("prior added")
initial.insert(0, Pose3(np.eye(4)))
T1 = np.eye(4); T1[0,3]=1.0
initial.insert(1, Pose3(T1))
odom_noise = noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
graph.add(BetweenFactorPose3(0, 1, Pose3(T1), odom_noise))
print("between added")
from gtsam import LevenbergMarquardtOptimizer, LevenbergMarquardtParams
params = LevenbergMarquardtParams()
params.setMaxIterations(100)
opt = LevenbergMarquardtOptimizer(graph, initial, params)
print("optimizing...")
result = opt.optimize()
print("done, err:", graph.error(result))
print("pose1:", result.atPose3(1).matrix())
