
import numpy as np

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import B, V, X

gps = np.load('/home/liu/bag/gps0.npy') 
imu = np.load('/home/liu/bag/imu0.npy')

optimizer = None
graphFactors = None
graphValues = None
lidar2Imu = gtsam.Pose3(gtsam.Rot3.Quaternion(1,0,0,0), gtsam.Point3(0,0,0)) 
priorPoseNoise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]) ) # rad,rad,rad,m, m, m
priorVelNoise = gtsam.noiseModel.Isotropic.Sigma(3, 100) 
priorBiasNoise = gtsam.noiseModel.Isotropic.Sigma(6, 10) 
correctionNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([ 0.05, 0.05, 0.05, 0.1, 0.1, 0.1]))
imuAccBiasN = 6.4356659353532566e-05
imuGyrBiasN = 3.5640318696367613e-05

noiseModelBetweenBias = np.array([imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN])


# Set IMU preintegration parameters
measured_acc_cov = np.eye(3) * np.power( 0.01, 2)
measured_omega_cov = np.eye(3) * np.power(  0.00175, 2)
# error committed in integrating position from velocities
integration_error_cov = np.eye(3) * np.power( 0, 2)

imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    # acc white noise in continuous
imu_params.setAccelerometerCovariance(measured_acc_cov)
    # integration uncertainty continuous
imu_params.setIntegrationCovariance(integration_error_cov)
    # gyro white noise in continuous
imu_params.setGyroscopeCovariance(measured_omega_cov)
imu_params.setOmegaCoriolis(np.zeros(3))
imuIntegratorOpt = gtsam.PreintegratedImuMeasurements(imu_params, gtsam.imuBias.ConstantBias())

def resetOptimization():
    global optimizer
    global graphFactors
    global graphValues
    optParameters = gtsam.ISAM2Params()
    optParameters.setRelinearizeThreshold(0.1)
    #optParameters.setRelinearizeSkip(1)
    optimizer = gtsam.ISAM2(optParameters)
    graphFactors = gtsam.NonlinearFactorGraph()
    graphValues = gtsam.Values()
    return 

trj0 = []
trj1 = []
velocity = []
bias = []
first = True
prevPose = None
prevVel = None
prevBias = None
key = None
begin_time = 0
end_time = 0
begin_idx = 0
lastImuT_opt = -1

for p in gps:
    lidarPose = gtsam.Pose3(gtsam.Rot3.Quaternion(*p[4:8]), gtsam.Point3(*p[1:4]))
    end_time = p[0]
    if(first):
        resetOptimization()
        begin_time = p[0]
        prevPose = lidarPose.compose(lidar2Imu)
        priorPose = gtsam.PriorFactorPose3(X(0), prevPose, priorPoseNoise)
        #.add(priorPose)
        prevVel = gtsam.Point3(0,0,0)
        priorVel = gtsam.PriorFactorPoint3(V(0), prevVel, priorVelNoise)
        graphFactors.add(priorVel)
        prevBias = gtsam.imuBias.ConstantBias()
        priorBias = gtsam.PriorFactorConstantBias(B(0), prevBias, priorBiasNoise)
        graphFactors.add(priorBias)
        # add values
        graphValues.insert(X(0), prevPose)
        graphValues.insert(V(0), prevVel)
        graphValues.insert(B(0), prevBias)
        # optimize once
        optimizer.update(graphFactors, graphValues)
        graphFactors.resize(0)
        graphValues.clear()
        imuIntegratorOpt.resetIntegrationAndSetBias(prevBias)
        prevState_ = gtsam.NavState(prevPose, prevVel)

        key = 1
        first = False
        continue
    if (key == 100):
        updatedPoseNoise = gtsam.noiseModel.Gaussian.Covariance(optimizer.marginalCovariance(X(key-1)))
        updatedVelNoise  = gtsam.noiseModel.Gaussian.Covariance(optimizer.marginalCovariance(V(key-1)))
        updatedBiasNoise = gtsam.noiseModel.Gaussian.Covariance(optimizer.marginalCovariance(B(key-1)))
        resetOptimization()
        priorPose = gtsam.PriorFactorPose3(X(0), prevPose, updatedPoseNoise)
        graphFactors.add(priorPose)
        priorVel = gtsam.PriorFactorPoint3(V(0), prevVel, updatedVelNoise)
        graphFactors.add(priorVel)
        priorBias = gtsam.PriorFactorConstantBias(B(0), prevBias, updatedBiasNoise)
        graphFactors.add(priorBias)
        # add values
        graphValues.insert(X(0), prevPose)
        graphValues.insert(V(0), prevVel)
        graphValues.insert(B(0), prevBias)
        # optimize once
        optimizer.update(graphFactors, graphValues)
        graphFactors.resize(0)
        graphValues.clear()
        key = 1
    # 1. integrate imu data and optimize
    for i in imu[begin_idx:]:
        # imu data is between two pose
        imuTime = i[0]
        begin_idx += 1
        if(imuTime< begin_time ):
            continue
        if(imuTime > end_time):
            break
        dt = 0
        if(lastImuT_opt < 0):
            dt = 0.01
        else:
            dt = imuTime - lastImuT_opt
        if dt <= 0:
            continue
        imuIntegratorOpt.integrateMeasurement(i[1:4], i[4:7], dt)
        imuIntegratorOpt.print()
        lastImuT_opt = imuTime
        state = imuIntegratorOpt.predict(prevState_, prevBias)
        #state.propState_.pose()
        trj0.append([state.pose().translation()[0],state.pose().translation()[1],state.pose().translation()[2]])
    #if(key>=1):
    #    break
    #imuQueOpt.pop_front()
    # add imu factor to graph
    imu_factor = gtsam.ImuFactor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), imuIntegratorOpt)
    bias_factor = gtsam.BetweenFactorConstantBias( B(key - 1), B(key), gtsam.imuBias.ConstantBias(), \
        gtsam.noiseModel.Diagonal.Sigmas( np.sqrt(imuIntegratorOpt.deltaTij())* noiseModelBetweenBias ))
    #v_factor = gtsam.BetweenFactorPoint3( V(key - 1), V(key), gtsam.Point3(0,0,0), \
    #    gtsam.noiseModel.Diagonal.Sigmas( np.sqrt(imuIntegratorOpt.deltaTij())* np.array([0.01,0.01,0.01]) ))

    graphFactors.add(imu_factor)
    graphFactors.add(bias_factor)
    #graphFactors.add(v_factor)
    # add pose factor
    curPose = lidarPose.compose(lidar2Imu)
    pose_factor = gtsam.PriorFactorPose3(X(key), curPose, correctionNoise)
    graphFactors.add(pose_factor)
    # insert predicted values
    propState_ = imuIntegratorOpt.predict(prevState_, prevBias)
    graphValues.insert(X(key), propState_.pose())
    graphValues.insert(V(key), propState_.velocity())
    graphValues.insert(B(key), prevBias)
    # optimize
    optimizer.update(graphFactors, graphValues)
    optimizer.update()
    graphFactors.resize(0)
    graphValues.clear()
    #Overwrite the beginning of the preintegration for the next step.
    result = optimizer.calculateEstimate()
    prevPose  = result.atPose3(X(key))
    prevVel   = result.atPoint3(V(key))
    prevBias  = result.atConstantBias(B(key))
    #Reset the optimization preintegration object.
    prevState_ = gtsam.NavState(prevPose, prevVel)
    imuIntegratorOpt.resetIntegrationAndSetBias(prevBias)
    key+=1
    trj0.append([prevPose.translation()[0],prevPose.translation()[1],prevPose.translation()[2]])
    trj1.append([curPose.translation()[0],curPose.translation()[1],curPose.translation()[2]])
    velocity.append([prevVel[0],prevVel[1],prevVel[2]])
    print(prevBias.accelerometer())
    bias.append([prevBias.accelerometer()[0],prevBias.accelerometer()[1],prevBias.accelerometer()[2]])
    #if(key == 100):
    #    break
trj0 = np.array(trj0)
trj1 = np.array(trj1)
velocity = np.array(velocity)
bias = np.array(bias)

#v0 = trj0[1:] - np.roll(trj0, 1,axis=0)[1:]
#v1 = trj1[1:] - np.roll(trj1, 1,axis=0)[1:]

def smooth(input, win):
    b=np.ones(win)/win
    return np.convolve(input, b, mode='same')

plt.grid()

if(True):
    #plt.scatter(trj0[:,0], trj0[:,1],color='b',s=3, label='imu preintegration')
    #plt.scatter(trj1[:,0], trj1[:,1],color='r',s=3, label='prue ndt matching')
    #plt.plot(trj0[:,2], label='imu preintegration')
    plt.plot(trj0[:,2], label='pose z')
    #plt.plot(velocity[:,2], label='velocity z')
    #plt.plot(trj0[:,0], label='imu preintegration')
else:
    #plt.plot(bias[:,0],color='r', label='bias x')
    #plt.plot(bias[:,1],color='g', label='bias y')
    #plt.plot(bias[:,2],color='b', label='bias z')
    plt.plot(velocity[:,0],color='r', label='bias x')
    plt.plot(velocity[:,1],color='g', label='bias y')
    plt.plot(velocity[:,2],color='b', label='bias z')

plt.legend() 
plt.show()


