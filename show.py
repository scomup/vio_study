import matplotlib.pyplot as plt
import numpy as np
import quaternion


def ToEulerAngles(q):
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    pitch = 0
    sinp = 2 * (q.w * q.y - q.z * q.x)
    if (np.abs(sinp) >= 1):
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll,pitch,yaw])


class transform:
    def __init__(self,translation, rotation):
        self.translation = translation
        self.rotation = rotation

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            translation = quaternion.as_rotation_matrix(self.rotation).dot(other.translation) + self.translation
            rotation = (self.rotation * other.rotation).normalized()
            return transform(translation, rotation)
        else:
            raise NotImplementedError()
    def inv(self):
        rotation = self.rotation.conjugate()
        translation = -quaternion.as_rotation_matrix(rotation).dot(self.translation)
        return transform(translation, rotation)

position = []
def getVs(data):
    ref = None
    ref_pose = None
    Vs = []
    for cur in data:
        if ref is None:
            ref = cur
            ref_pose = transform(ref[1:4],np.quaternion(*ref[4:8]))
            continue
        cur_pose = transform(cur[1:4],np.quaternion(*cur[4:8]))
        dt = cur[0] - ref[0]
        delta = ref_pose.inv() * cur_pose
        velocity = delta.translation/dt
        #va = quaternion.as_euler_angles(delta.rotation)/dt
        va = ToEulerAngles(delta.rotation)/dt
        Vs.append([cur[0],velocity[0],velocity[1],velocity[2],va[0],va[1],va[2]])
        ref_pose = cur_pose
        ref = cur
    return np.array(Vs)
def getVs2(data):
    ref = None
    ref_pose = None
    Vs = []
    for cur in data:
        if ref is None:
            ref = cur
            ref_pose = ref[1:4]
            continue
        cur_pose = cur[1:4]
        dt = cur[0] - ref[0]
        delta =  cur_pose - ref_pose
        velocity = delta/dt
        Vs.append([cur[0],np.linalg.norm(delta),0,0])
        ref_pose = cur_pose
        ref = cur
    return np.array(Vs)

def getA(data):
    ref = None
    As = []
    #c = 0
    for cur in data:
        if ref is None:
            ref = cur
            continue

        #c+=1
        #if(c%3):
        #    continue
        dt = cur[0] - ref[0]
        a = (cur[1:4] - ref[1:4])/dt
        As.append([cur[0],a[0],a[1],a[2]])
        ref = cur
    return np.array(As)


def loadImuData(imu_data_file):
    imu = []
    print("-- Reading IMU measurements from file")
    with open(imu_data_file, encoding='UTF-8') as imu_data:
        data = imu_data.readlines()
        for i in range(3, len(data)):  # ignore the first line
            time, dt, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = map( float, data[i].split(' '))
            imu.append([time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
    return np.array(imu)

def loadGpsData(gps_data_file):
    gps=[]
    with open(gps_data_file, encoding='UTF-8') as gps_data:
        data = gps_data.readlines()
        for i in range(1, len(data)):
            time, x, y, z = map(float, data[i].split(','))
            gps.append([time, x, y, z])

    return np.array(gps)

#gps = np.load('/home/liu/bag/gps.npy') 
#imu = np.load('/home/liu/bag/imu.npy')
gps = loadGpsData('/home/liu/workspace/gtsam/examples/Data/KittiGps_converted.txt') 
imu = loadImuData('/home/liu/workspace/gtsam/examples/Data/KittiEquivBiasedImu.txt')

v = getVs2(gps)
#a = getA(v)

def smooth(input, win):
    b=np.ones(win)/win
    return np.convolve(input, b, mode='same')

#plt.plot(imu[:,0],smooth(imu[:,1], 5000),label='x')
#plt.plot(imu[:,0],smooth(imu[:,2], 5000),label='y')
#plt.plot(v[:,0],smooth(v[:,1], 100),label='vx')
#plt.plot(v[:,0],smooth(v[:,2], 100),label='vy')
#plt.plot(v[:,0],smooth(v[:,4], 1000),label='p')
#plt.plot(v[:,0],smooth(v[:,5], 1000),label='r')
#plt.plot(imu[:,0],smooth(imu[:,6], 5000),label='imu yaw')
#plt.plot(v[:,0],smooth(v[:,6], 100),label='real yaw')
f, axarr = plt.subplots(2)

axarr[0].plot(imu[:,0],smooth(imu[:,1], 5000),label='imu acc x')
axarr[0].plot(v[:,0],smooth(v[:,1]*0.01, 100),label='real speed x')
axarr[1].plot(imu[:,0],smooth(imu[:,2], 5000),label='imu acc y')
axarr[1].plot(v[:,0],smooth(v[:,2], 100),label='real speed y')


axarr[0].legend()
axarr[1].legend()

plt.show()
