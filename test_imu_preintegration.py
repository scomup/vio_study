
import numpy as np

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import B, V, X
from math_tools import *

imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
imuIntegratorOpt = gtsam.PreintegratedImuMeasurements(imu_params, gtsam.imuBias.ConstantBias())
prevBias = gtsam.imuBias.ConstantBias()

imu = np.load('/home/liu/bag/imu0.npy')



class imuIntegration():

    def __init__(self):
        self.theta = np.array([0,0,0])
        self.position = np.array([0,0,0])
        self.velocity = np.array([0,0,0])
    def update(self, acc, omega, dt):
        #check imuFactor.pdf: (11~13)
        H_inv = np.linalg.inv( H(self.theta))
        Rka = SO3exp(self.theta).dot(acc)
        self.theta = self.theta + H_inv.dot(omega) * dt
        self.position = self.position + self.velocity * dt + Rka*dt*dt/2
        self.velocity = self.velocity + Rka * dt 

    def predict(self, acc, omega, dt):
            #check imuFactor.pdf: (11~13)

        theta = self.theta


        H_inv = np.linalg.inv( H(self.theta))
        Rka = SO3exp(self.theta).dot(acc)
        self.theta = self.theta + H_inv.dot(omega) * dt
        self.position = self.position + self.velocity * dt + Rka*dt*dt/2
        self.velocity = self.velocity + Rka * dt 

imu_integration = imuIntegration()

lastImuT_opt = -1

prevPose = gtsam.Pose3(gtsam.Rot3.Quaternion(1,0,0,0), gtsam.Point3(0,0,0)) 
prevVel = np.array([2,0,0])
prevState = gtsam.NavState(prevPose, prevVel)
trj = []
for i in imu:
    imuTime = i[0]
    dt = 0
    if(lastImuT_opt < 0):
        dt = 0.01
    else:
        dt = imuTime - lastImuT_opt
    if dt <= 0:
        continue
    imu_integration.update(i[1:4], i[4:7], dt)
    imuIntegratorOpt.integrateMeasurement(i[1:4], i[4:7], dt)
    currState = imuIntegratorOpt.predict(prevState, prevBias)
    #trj.append([currState.pose().translation()[0],currState.pose().translation()[1],currState.pose().translation()[2]])
    #print(prevState.pose().translation())

    #print(imu_integration.theta)
    #print(gtsam.Rot3.Logmap(imuIntegratorOpt.deltaRij()))


    pass
#import matplotlib.pyplot as plt
##trj = np.array(trj)
#plt.plot(trj[:,0],trj[:,2])
#  plt.show()