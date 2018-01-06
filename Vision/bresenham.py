#coding:utf-8
#Bresenham算法:求栅格上离直线最近的各点
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

xsize=1
ysize=1
p0=np.array([0.1,0])
p1=np.array([8,11])
vec=p1-p0
#直线方程Y=AX+B
A=vec[1]*1.0/vec[0] #斜率
B=p0[1]-A*p0[0]    #常数项
x_pts=[]
y_pts=[]

def drawLine(start,end):
    x,x_end=[np.floor(x[0]) for x in start,end]
    while True:
        x=x+1
        if x==x_end:
            break
        y=np.round(A*x+B)
        x_pts.append(x)
        y_pts.append(y)

drawLine(p0,p1)
xticks,yticks=map(lambda x:[int(min(x)-3),int(max(x)+3)],(x_pts,y_pts))
plt.figure("Bresenham")
plt.axes().set_xticks(range(*xticks))
plt.axes().set_yticks(range(*yticks))
plt.axes().set_xlim(*xticks)
plt.axes().set_ylim(*yticks)
plt.scatter(x_pts, y_pts, c='red',s=20)
plt.plot([p0[0],p1[0]],[p0[1],p1[1]])
plt.grid(color='blue', linewidth='0.3', linestyle='--')
plt.show()