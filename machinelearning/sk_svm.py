#coding:utf-8
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) #高斯核函数
svr_lin = SVR(kernel='linear', C=1e3) #线性核函数
svr_poly = SVR(kernel='poly', C=1e3, degree=2) #多项式函数
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)
# look at the results
lw = 2
# index1=np.where(y>0) #大于0的index
# index2=np.setdiff1d(np.arange(len(y)),index1) #小于等于0的index
plt.scatter(X, y, color='darkorange', label='data')

plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='r', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()