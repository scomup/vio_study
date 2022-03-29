import numpy as np
import matplotlib.pyplot as plt

def get_points():
    theta = np.linspace( 0 , 2 * np.pi , 10 )
    radius = 1
    a = radius * np.cos( theta )[:-1]
    b = radius * np.sin( theta )[:-1]
    points = np.vstack([a,b]).T
    return points
points = get_points()

def calcHb(edges,points):
    n = points.shape[0]*2
    H = np.zeros([n,n])
    b = np.zeros(n)
    e = 0
    #J = np.zeros([len(edges)*2,n])
    #r = np.zeros(len(edges)*2)
    row = 0
    for (i,j,rx,ry) in edges:
        #r[2*row] = rx
        #r[2*row+1] = ry
        
        if(i!=j):
            #J[2*row:2*row+2,i*2:i*2+2] = np.eye(2)
            #J[2*row:2*row+2,j*2:j*2+2] = -np.eye(2)
            dx = points[i,0] - points[j,0]
            dy = points[i,1] - points[j,1]

            H[i*2:i*2+2,j*2:j*2+2] += -np.eye(2)
            H[j*2:j*2+2,i*2:i*2+2] += -np.eye(2)
            H[i*2:i*2+2,i*2:i*2+2] += np.eye(2)
            H[j*2:j*2+2,j*2:j*2+2] += np.eye(2)
            b[i*2:i*2+2] += np.array([rx-dx,ry-dy])
            b[j*2:j*2+2] += -np.array([rx-dx,ry-dy])
            e += np.linalg.norm(np.array([rx-dx,ry-dy]))
        else:
            #J[2*row:2*row+2,i*2:i*2+2] = np.eye(2)
            H[i*2:i*2+2,j*2:j*2+2] += np.eye(2)
            b[i*2:i*2+2] += np.array([rx-points[i,0],ry-points[i,1]])
            e += np.linalg.norm(np.array([rx-points[i,0],ry-points[i,1]]))
        row+=1
    
    #H2 = J.T.dot(J)
    #b2 = J.T.dot(r)
    return H,b,e

        
xx = []
yy = []
edges = []

for step in range(1,6):
    for i in range(points.shape[0]):
        j = (i + step)%points.shape[0]
        xx.append(points[i,0])
        xx.append(points[j,0])
        yy.append(points[i,1])
        yy.append(points[j,1])
        rx = points[i,0] - points[j,0] + np.random.normal(0,0.1)
        ry = points[i,1] - points[j,1] + np.random.normal(0,0.1)
        edges.append([i,j, rx, ry])
        print('i:%d,j:%d'%(i,j))
    print('-----')
edges.append([0,0,points[0,0],points[0,1]])

points_est = np.zeros([points.shape[0],2])
while(True):
    H, b, e = calcHb(edges,points_est)
    print(e)
    dx = np.linalg.solve(H, b)
    if(np.linalg.norm(dx) < 0.001):
        break
    points_est += dx.reshape([points.shape[0],2])

plt.plot(xx,yy, color='b')
plt.scatter(points[:,0],points[:,1], color='b',label='real')
plt.scatter(points_est[:,0],points_est[:,1], color='r',label='est')

points_est2 = np.zeros([points.shape[0],2])
while(True):
    H, b, e = calcHb(edges,points_est2)
    H = H[2:,2:]
    b = b[2:]
    dx = np.linalg.solve(H, b)
    if(np.linalg.norm(dx) < 0.001):
        break
    points_est2[1:] += dx.reshape([points.shape[0]-1,2])

plt.scatter(points_est2[:,0],points_est2[:,1], color='g',label='remove')

plt.legend()
plt.show()


"""
m = 4
points = np.array([[0,0.],[-1,1],[1,1],[-1,-1],[1,-1],[2,-2]])
points += m
#pointsN = points + np.random.normal(0,0.3,points.shape)
#plt.scatter(pointsN[:,0],pointsN[:,1], color='r')
n = 2
J = np.zeros([6*n,6*n])
e = np.zeros([6*n])

J[0 : 2, 0 : 2] = np.eye(2)
J[0 : 2, 1*n : 1*n+2] = -np.eye(2)

J[2 : 4, 0 : 2] = np.eye(2)
J[2 : 4, 2*n : 2*n+2] = -np.eye(2)

J[4 : 6, 0 : 2] = np.eye(2)
J[4 : 6, 3*n : 3*n+2] = -np.eye(2)

J[6 : 8, 0 : 2] = np.eye(2)
J[6 : 8, 4*n : 4*n+2] = -np.eye(2)

J[8 : 10, 4*n : 4*n+2] = np.eye(2)
J[8 : 10, 5*n : 5*n+2] = -np.eye(2)

J[10 : 12, 0 : 2] = np.eye(2)



e[0 : 2] = points[0] - points[1]
e[2 : 4] = points[0] - points[2]
e[4 : 6] = points[0] - points[3]
e[6 : 8] = points[0] - points[4]
e[8 :10] = points[4] - points[5]
e[10 : 12] = points[0]


b = J.T.dot(e)

H = J.T.dot(J)
x = np.linalg.solve(H, b)
points_0 = x.reshape(points.shape)

#-------------------------------
H_1 = H[2:12,2:12]
b_1 = b[2:12]
x_1 = np.linalg.solve(H_1, b_1)
points_1 = x_1.reshape([5,2])

#-------------------------------
Haa = H[2:12,2:12]
Hbb_inv = np.linalg.inv(H[0:2,0:2])
Hab = H[2:12,0:2]
Hba = H[0:2,2:12]

ba = b[2:12]
bb = b[0:2]

H_2 = Haa - Hab.dot(Hbb_inv.dot(Hba))
b_2 = ba - Hab.dot(Hbb_inv.dot(bb))
x_2 = np.linalg.solve(H_2, b_2)
points_2 = x_2.reshape([5,2])




plt.scatter(points[:,0],points[:,1], color='black',label='true')
#plt.scatter(pointsN[:,0],pointsN[:,1], color='yellow',label='noise')

plt.scatter(points_0[:,0],points_0[:,1], color='b',label='origin')
plt.scatter(points_1[:,0],points_1[:,1], color='r',label='remove')
plt.scatter(points_2[:,0],points_2[:,1], color='g',label='margin')
plt.legend()
plt.show()


"""
