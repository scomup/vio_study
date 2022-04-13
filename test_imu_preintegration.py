
import numpy as np

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import B, V, X

imu = np.load('/home/liu/bag/imu0.npy')


from scipy.spatial.transform import Rotation

epsilon = 1e-5


# 3d Rotation Matrix to so3
def log(mat):
    rot = Rotation.from_dcm(mat)
    q = rot.as_quat()
    squared_n = np.dot(q[0:3], q[0:3])
    if (squared_n < epsilon * epsilon):
        squared_w = q[3] * q[3]
        two_atan_nbyw_by_n = 2. / q[3] - (2.0/3.0) * (squared_n) / (q[3] * squared_w)
    else:
        n = np.sqrt(squared_n)
        if (np.abs(q[3]) < epsilon):
            if (q[3] > 0.):
              two_atan_nbyw_by_n = np.pi / n
            else:
              two_atan_nbyw_by_n = -np.pi / n
        else:
            two_atan_nbyw_by_n = 2. * np.arctan(n / q[3]) / n
    return two_atan_nbyw_by_n * q[0:3]
    
# so3 to 3d Rotation Matrix
def exp(v):
    theta_sq = np.dot(v, v)
    imag_factor = 0.
    real_factor = 0.
    if (theta_sq < epsilon * epsilon):
        theta_po4 = theta_sq * theta_sq
        imag_factor = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4
        real_factor = 1. - (1.0 / 8.0) * theta_sq +   (1.0 / 384.0) * theta_po4
    else:
        theta = np.sqrt(theta_sq)
        half_theta = 0.5 * theta
        sin_half_theta = np.sin(half_theta)
        imag_factor = sin_half_theta / theta
        real_factor = np.cos(half_theta)
    quat = np.array([imag_factor*v[0], imag_factor*v[1], imag_factor*v[2], real_factor])
    rot = Rotation.from_quat(quat)
    return rot.as_dcm()


def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])

#check imuFactor.pdf: Derivative of The Local Coordinate Mapping
def H(theta):
    h = np.eye(3)
    theta_shew = skew(theta)
    theta_shew_k = np.eye(3)
    m = 1
    n = 1
    for k in range(1,5):
        m *= (k+1)
        n *= -1
        theta_shew_k = theta_shew_k.dot(theta_shew)
        h += (n/m)*theta_shew_k
    return h

"""
va = np.array([1,1,1])
vb = np.array([0.2,0.3,0.4])
a = gtsam.SO3(va)
b = gtsam.SO3(vb)
R0 = gtsam.SO3.Expmap(va+vb)
R1 = gtsam.SO3.Expmap(va) * gtsam.SO3.Expmap(vb)
R2 = gtsam.SO3.Expmap(va) * gtsam.SO3.Expmap(H(va).dot(vb))
"""

class imuIntegration():

    def __init__(self):
        self.theta = np.array([0,0,0])
        self.position = np.array([0,0,0])
        self.velocity = np.array([0,0,0])
    def update(self, acc, omega, dt):
        #check imuFactor.pdf: (11~13)
        H_inv = np.linalg.inv( H(self.theta))
        Rka = exp(self.theta).dot(acc)
        self.theta = self.theta + H_inv.dot(omega) * dt
        self.position = self.position + self.velocity * dt + Rka*dt*dt
        self.velocity = self.velocity + Rka * dt 
imu_integration = imuIntegration()

lastImuT_opt = -1

for i in imu:
    # imu data is between two pose
    """array([ 0.00000000e+00,  1.45013384e-03, -1.07148661e+00,  9.80438765e+00,
        1.13141796e-03,  9.92803351e-04, -5.22291678e-01])
    """
    imuTime = i[0]
    dt = 0
    if(lastImuT_opt < 0):
        dt = 0.01
    else:
        dt = imuTime - lastImuT_opt
    if dt <= 0:
        continue
    imu_integration.update(i[1:4], i[4:7], dt)
    """
        deltaRij.ypr = (-0.00522291673 9.95753481e-06 1.12882015e-05)
    deltaPij =   7.2506692e-08 -5.35743306e-05  0.000490219382
    deltaVij = 1.45013384e-05  -0.0107148661   0.0980438765
    """

    print(imu_integration.theta)
    print(imu_integration.position)
    print(imu_integration.velocity)
