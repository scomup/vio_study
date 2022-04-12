import numpy as np
import matplotlib.pyplot as plt
import gtsam
from typing import List, Optional
from functools import partial

I = np.eye(1)
def error_match2d(measurement: np.ndarray, this: gtsam.CustomFactor,
                values: gtsam.Values,
                jacobians: Optional[List[np.ndarray]]) -> float:
    """Odometry Factor error function
    :param measurement: Odometry measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """
    x = values.atVector(this.keys()[0])
    a = measurement[0:2]
    b = measurement[2:4]
    error = transform2d(x,a).reshape(-1) - b
    if jacobians is not None:
        jacobians[0] = np.array( [[1, 0,-a[1]],[0,1,a[0]]])
        #jacobians[1] = np.array([0,1,a[0]])

    return error


def transform2d(x,p):
    t = x[0:2]
    R = np.array([[np.cos(x[2]),-np.sin(x[2])], [np.sin(x[2]),np.cos(x[2])]])
    element = int(p.size/2)
    tp = np.dot(R,p).reshape(2, -1) + np.array([t,]*(element)).transpose()
    return tp

if __name__ == '__main__':
    x = np.array([-0.3,0.2,np.pi/3])

    elements = 100
    a = (np.random.rand(elements,2)-0.5)*2
    b = transform2d(x, a.T).T
    b += np.random.normal(0, 0.03, (elements, 2))
    # Create a Factor Graph and Values to hold the new data

    # We now can use nonlinear factor graphs
    factor_graph = gtsam.NonlinearFactorGraph()

    # New Values container
    v = gtsam.Values()

    v.insert(0, np.array([0,0,0]))

    # Add factors for GPS measurements
    gps_model = gtsam.noiseModel.Isotropic.Sigma(2, 0.01)

    # Add the GPS factors
    for k in range(elements):
        gf = gtsam.CustomFactor(gps_model, [0], partial(error_match2d, np.array([a[k,0],a[k,1],b[k,0],b[k,1]])))
        factor_graph.add(gf)

    # Initialize optimizer
    params = gtsam.GaussNewtonParams()
    optimizer = gtsam.GaussNewtonOptimizer(factor_graph, v, params)

    # Optimize the factor graph
    result = optimizer.optimize()
    print(result.atVector(0))
    x = result.atVector(0)
    print(optimizer.error())
    a2 = transform2d(x,a.T).T
    #exit(0)
    plt.cla()
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.scatter(a2[:,0], a2[:,1], c= 'r')
    plt.scatter(b[:,0], b[:,1], c= 'b')
    plt.show()