# #coding:utf-8
# # 公式1： X(k|k-1) = AX(k-1 | k-1) + BU(k) + W(k)
# # 公式2： P(k|k-1)  = AP(k-1|k-1)A' + Q(k)
# # 公式3： X(k|k)  = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)
# # 公式4：   Kg(k)  = P(k|k-1)H'/{HP(k|k-1)H' + R}        //卡尔曼增益
# # 公式5 ： P(k|k)   = (1- Kg(k) H) P(k|k-1)
# # Z(k) = HX(k) + V，Z是测量值，X是系统值，W是过程噪声，V是测量噪声，H是测量矩阵，A是转移矩阵，
# # Q是W的协方差，R是V的协方差，X(k|k-1)是估计值；X(k|k)是X(k|k-1)的最优估计值，即滤波估计值；
# # P(k|k-1)是估计值误差方差矩阵，P(k|k)是滤波误差方差矩阵
# # 这里设置A=1，H=1，BU=0,W=0
# #coding:utf-8
# import numpy
# import matplotlib.pyplot as plt
# #这里是假设A=1，H=1的情况

# # intial parameters
# n_iter = 50
# sz = (n_iter,) # size of array
# x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
# z = numpy.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)Z(k) = HX(k) + V

# Q = 1e-5 # process variance

# # allocate space for arrays
# Xpost=numpy.zeros(sz)      # a posteri estimate of x
# Ppost=numpy.zeros(sz)      # a posteri error estimate
# Xprio=numpy.zeros(sz)    # a priori estimate of x
# Pprio=numpy.zeros(sz)    # a priori error estimate
# K=numpy.zeros(sz)         # gain or blending factor

# R = 0.1**2 # estimate of measurement variance, change to see effect

# # intial guesses
# Xpost[0] = 0.0
# Ppost[0] = 1.0

# for k in range(1,n_iter):
#     # time update
#     Xprio[k] = Xpost[k-1] #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
#     Pprio[k] = Ppost[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

#     # measurement update
#     K[k] = Pprio[k]/( Pprio[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
#     Xpost[k] = Xprio[k]+K[k]*(z[k]-Xprio[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
#     Ppost[k] = (1-K[k])*Pprio[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

# plt.subplot(211)
# plt.plot(z,'k+',label='noisy measurements')     #测量值
# plt.plot(Xpost,'b-',label='a posteri estimate')  #过滤后的值
# plt.axhline(x,color='g',label='truth value')    #系统值
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Voltage')

# plt.subplot(212)
# valid_iter = range(1,n_iter) # Pprio not valid at step 0
# plt.plot(valid_iter,Pprio[valid_iter],label='a priori error estimate')
# plt.xlabel('Iteration')
# plt.ylabel('$(Voltage)^2$')
# plt.setp(plt.gca(),'ylim',[0,.01])
# plt.show()

import numpy as np
from numpy.random import random
def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)#例如:输入[1,2,3] 输出[1,3,6]
    cumulative_sum[-1] = 1. # avoid round-off error避免圆整错误，保证所有的权重之和等于1而不是0.9999
    indexes = np.searchsorted(cumulative_sum, random(N)) #生成N个随机数，并找出每个随机数在cumulative_sum中的位置(random(N):随机均布)
    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N) #将权重都设为一样


a=np.array([1,2,3,4,5])
b=[0.2,0.3,0.4,0.05,0.05]
simple_resample(a,b)