import numpy as np
import matplotlib.pyplot as plt

def fun(x,y):
    return 2*(x*y-1)**2

def calcH(x,y):
    xx = 4*y*y
    yy = 4*x*x
    xy = 8*x*y - 4
    return np.array([[xx,xy],[xy,yy]])
      
def calcJ(x,y):
    dx = 4*x*y*y - 4*y
    dy = 4*y*x*x - 4*x
    return np.array([dx,dy])

def solve(p):
    step = 0.2
    H = calcH(p[0],p[1])
    J = calcJ(p[0],p[1])

    while(True):
        e = fun(p[0],p[1])
        #ax1.scatter(p[0]/0.02,p[1]/0.02,c='black')
        #print(e)
        if(e < 0.0001):
            break
        H = calcH(p[0],p[1])
        J = calcJ(p[0],p[1])
        dp = -np.linalg.solve(H,J)
        if(np.linalg.norm(dp) > step):
           dp = dp /  np.linalg.norm(dp) * step
        p = p + dp
    return p,H,J

def taylor(p,H,J):
    x0 = p[0]
    y0 = p[1]
    return fun(x0, y0) + (xx - x0) * J[0] + (yy - y0) * J[1] + \
         (xx - x0)*(xx - x0)*H[0,0]/2 + (xx - x0)*(yy - y0)*H[0,1] + (yy - y0)*(yy - y0)*H[1,1]/2


X = np.linspace(0, 2, 100)
Y = np.linspace(0, 2, 100)
xx, yy = np.meshgrid(X, Y)
xy = np.dstack([xx, yy])

zz = fun(xx,yy)
fig = plt.figure(figsize=(20,4))
ax0 = fig.add_subplot(1,4,1)

p1 = np.array([1.4, 0.5])
p2 = np.array([0.5, 1.2])

ax0.pcolormesh(xx, yy, zz,vmax=4,cmap="jet")
ax0.scatter(p1[0],p1[1],marker='x', color='w')
ax0.scatter(p2[0],p2[1],marker='o', facecolors='none',edgecolors="w")

p1n,H1,J1=solve(p1)
zz1 = taylor(p1n,H1,J1)
ax1 = fig.add_subplot(1,4,2)
ax1.pcolormesh(xx, yy, zz1,vmax=4,cmap="jet")
ax1.scatter(p1[0],p1[1],marker='x', color='w')

p2n,H2,J2=solve(p2)
zz2 = taylor(p2n,H2,J2)
ax2 = fig.add_subplot(1,4,3)
ax2.pcolormesh(xx, yy, zz2,vmax=4,cmap="jet")
ax2.scatter(p2[0],p2[1],marker='o', facecolors='none',edgecolors="w")

zz3 = (zz1 + zz2)/2

ax3 = fig.add_subplot(1,4,4)
ax3.pcolormesh(xx, yy, zz3,vmax=4,cmap="jet")


plt.show()
