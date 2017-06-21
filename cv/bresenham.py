#coding:utf-8
#求直线上各点
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

xsize=0.2
ysize=0.2
p0=np.array([0,0])
p1=np.array([8,10])
vec=p1-p0
A=vec[1]*1.0/vec[0]
B=-1
C=p0[1]-A*p0[0]
D=np.linalg.norm([A,B])
x_pts=[]
y_pts=[]

def drawLine(start,end):
    while True:
        if start[0]==end[0] and start[1]==end[1]:
            break
        x,y=start
        p1_x,p1_y=x+xsize,y
        p2_x,p2_y=x,y+ysize
        d_p1=np.abs((A*p1_x+B*p1_y+C)/D)
        d_p2=np.abs((A*p2_x+B*p2_y+C)/D)
        if d_p1<d_p2:
            start=np.array([p1_x,p1_y])
        else:
            start=np.array([p2_x,p2_y])
        x_pts.append(start[0])
        y_pts.append(start[1])


drawLine(p0,p1)
plt.plot(x_pts,y_pts)
plt.grid(color='blue', linewidth='0.3', linestyle='--')
plt.show()