#coding:utf-8
'''
凸包算法
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=np.random.randint(0,100,10).reshape([-1,2])
# data=np.array([[0,0],[0,10],[15,10],[10,1]])
data=data[data[:,1].argsort()]
p0=data[0]
data1=data[1:,]
vecs=[(p-p0).dot(np.array([1,0]))/np.linalg.norm(p-p0) for p in data1]#矢量集合
orders=np.argsort(-np.array(vecs)) #矢量排序序号
data1=data1[orders]
data=np.vstack((p0,data1,p0))
plt.scatter(data[:-1,0],data[:-1,1]) #散点图
plt.plot(data[:,0],data[:,1]) 
plt.show()
