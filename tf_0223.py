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
x_data = np.float32(np.random.rand(2, 100)) # shape=(2,100)
y_data = np.dot([0.100, 0.200], x_data) + 0.300# shape=(1,2)*(2,100)=(1,100)

#创建2个变量
# tf.zeros([1])->tensor初始值为0,维度(1,1)
b = tf.Variable(tf.zeros([1]))
# tf.random_uniform->tensor,(-1,1)之间的均匀分布随机值
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# 构造线性模型 W*x+b=y
y = tf.matmul(W, x_data) + b

# 建立损失函数->均方差函数
loss = tf.reduce_mean(tf.square(y - y_data))
# 设置训练优化器的学习率为0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 设置训练目标为损失函数最小化
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动会话/交互会话(用于Ipython)
sess = tf.Session()
# sess =tf.InteractiveSession()
sess.run(init)
# 201迭代
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        # 当前epoch的W 和 B值
        print(step, sess.run(W), sess.run(b))
