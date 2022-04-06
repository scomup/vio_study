from __future__ import print_function

import numpy as np

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import math

theta = np.arange(0, np.pi*2, np.pi/6)
pose = np.stack([np.cos(theta)*np.exp(-theta*0.1),np.sin(theta)*np.exp(-theta*0.1),theta+np.pi/2]).T
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.02, 0.02, 0.01))
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.03, 0.03, 0.01))
initial_estimate = gtsam.Values()
graph = gtsam.NonlinearFactorGraph()

for i, p in enumerate(pose):
    initial_estimate.insert(i, gtsam.Pose2(*p))

graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(*pose[0]), PRIOR_NOISE))
for i in range(initial_estimate.size()-1):
    j = (i+1)
    dij = initial_estimate.atPose2(i).inverse() * initial_estimate.atPose2(j)
    graph.add(gtsam.BetweenFactorPose2(i, j, dij,ODOMETRY_NOISE))
graph.add(gtsam.BetweenFactorPose2(initial_estimate.size()-1, 0, gtsam.Pose2(0,0,0),ODOMETRY_NOISE))

parameters = gtsam.GaussNewtonParams()

    # Stop iterating once the change in error between steps is less than this value
parameters.setRelativeErrorTol(1e-5)
    # Do not perform more than N iteration steps
parameters.setMaxIterations(100)
    # Create the optimizer ...
optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)
    # ... and optimize
result = optimizer.optimize()
marginals = gtsam.Marginals(graph, result)

for i, p in enumerate(pose):
    gtsam_plot.plot_pose2(0, result.atPose2(i), 0.1,marginals.marginalCovariance(i))

plt.axis('equal')
plt.show()

