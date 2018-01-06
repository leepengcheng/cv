#coding:utf-8
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# 生成样本
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel() #转换为1维
y[::5] += 3 * (0.5 - np.random.rand(8)) #添加噪声
# 拟合回归模型
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) #高斯核函数
svr_lin = SVR(kernel='linear', C=1e3) #线性核函数
svr_poly = SVR(kernel='poly', C=1e3, degree=3) #多项式函数
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# 线宽
lw = 2
plt.scatter(X, y, color='darkorange', label='data') #绘制原始数据散点图

plt.plot(X, y_rbf, color='navy', linewidth=lw, label='RBF model')#高斯核
plt.plot(X, y_lin, color='c', linewidth=lw, label='Linear model') #线性核
plt.plot(X, y_poly, color='r', linewidth=lw, label='Polynomial model')#多项式核
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()