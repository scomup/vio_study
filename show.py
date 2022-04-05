import matplotlib.pyplot as plt
import numpy as np
import quaternion

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
        Vs.append([cur[0],velocity[0],velocity[1],velocity[2]])
    
    return np.array(Vs)
    

gps = np.load('/home/liu/bag/gps.npy') 
imu = np.load('/home/liu/bag/imu.npy')

v = getVs(gps)
num=5000
b=np.ones(num)/num


plt.plot(imu[:,0],np.convolve(imu[:,1], b, mode='same'),label='x')
plt.plot(imu[:,0],np.convolve(imu[:,2], b, mode='same'),label='y')
plt.plot(v[:,0],v[:,1],label='vx')
plt.plot(v[:,0],v[:,2],label='vy')
#plt.plot(imu[0],np.convolve(imu[:,1], b, mode='same'),label='y')
#plt.plot(np.convolve(imu[:,2]-9.81, b, mode='same'),label='z')
#plt.plot(gps[:,0],label='xx')

plt.legend()
plt.show()
