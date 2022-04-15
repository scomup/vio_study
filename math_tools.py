from scipy.spatial.transform import Rotation
import numpy as np

epsilon = 1e-5
# 3d Rotation Matrix to so3
def logmapSO3(mat):
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
def expmapSO3(v):
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
    return rot.as_matrix()


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


# dM33(x)*v/dx
def funD(x, v, func):
    A = func(x)
    D = np.zeros([3,3])
    delta = 0.0000001
    for i in range(3):
        for j in range(3):
            da = x.copy()
            da[j] += delta
            d = (func(da)[i].dot(v) - A[i].dot(v))/delta
            D[i,j] = d
    return D

#M = expmapSO3(np.array([1,0.3,0.7]))
x = np.array([0.,0.,0.])
v = np.array([1,0.3,0.7])
D = funD(x,v,expmapSO3)
D2 = expmapSO3(x).dot(skew(v))
print(D)
print(D2)