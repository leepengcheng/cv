#coding:utf-8
#求直线上各点
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

xsize=1
ysize=1
p0=np.array([0,0])
p1=np.array([8,11])
vec=p1-p0
A=vec[1]*1.0/vec[0]
B=-1
C=p0[1]-A*p0[0]
D=np.linalg.norm([A,B])
x_pts=[]
y_pts=[]
MAX=10

def drawLine(start,end):
    global MAX
    while True:
        if (start==end).all() or MAX<=0:
            break
        MAX=MAX-1
        x,y=start
        p1_x,p1_y=x+xsize,y
        p2_x,p2_y=x+xsize,y+ysize
        d_p1=np.abs((A*p1_x+B*p1_y+C)/D)
        d_p2=np.abs((A*p2_x+B*p2_y+C)/D)
        if d_p1<d_p2:
            start=np.array([p1_x,p1_y])
        else:
            start=np.array([p2_x,p2_y])
        x_pts.append(start[0])
        y_pts.append(start[1])


drawLine(p0,p1)
plt.scatter(x_pts, x_pts, c='red',s=5)
plt.plot([p0[0],p1[0]],[p0[1],p1[1]])
plt.grid(color='blue', linewidth='0.3', linestyle='--')
plt.show()