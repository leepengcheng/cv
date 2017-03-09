#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
已知方程 Wx+b=y
给出100个x输入和对应的输出y
求W和b的拟合值
即W->[0.100, 0.200] b->0.300的偏离程度
'''

# 生成随机数据 总共 100 个点的输入值和输出值
x_data = np.float32(np.random.rand(100, 2)) # shape=(100,2)
y_data = np.dot(x_data,[[0.1],[0.2]]) + 0.300# shape=(1,2)*(2,100)=(1,100)


W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))
# 构造线性模型 x*W+b=y
y = tf.matmul(x_data,W) + B

# 建立损失函数->均方差函数
loss = tf.reduce_mean(tf.square(y - y_data))
# loss=tf.reduce_mean(-y_data*tf.log(y))
# 设置训练优化器的学习率为0.5
train  = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 启动会话/交互会话(用于Ipython)
sess = tf.Session()
# sess =tf.InteractiveSession()
sess.run(init)
# 201迭代
for step in range(100):
    l,_,w,b=sess.run([loss,train,W,B])
    print("step: %4d  loss:%s   w:%s b:%s\n"%(step,l,w,b))
sess.close()