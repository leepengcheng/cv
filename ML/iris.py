#coding:utf-8
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb

#线性回归
lr = linear_model.LinearRegression()
# #加载波士顿数据
boston = datasets.load_boston()
y = boston.target

#交叉验证(Cross Validation)的好处是可以从有限的数据中获得尽可能多的有效信
#息，从而可以从多个角度去学习样本，避免陷入局部的极值。在这个过程中，无论是训练样本还
#是测试样本都得到了尽可能多的学习。下例中将数据分为10份,每次迭代使用9个作为训练集，1个作为测试集
#cross_val_predic用线性回归建立模型，并返回预测值
predicted = cross_val_predict(lr, boston.data, y, cv=10)
fig, ax = plt.subplots()
ax.scatter(y, predicted) #散点图,X为房价的实际值,Y为线性回归的预测房价
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4) #创建中间的分割线，离该线越近预测越准确
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

from sklearn import datasets
from sklearn import linear_model
import numpy as np
import pylab as pl

iris = datasets.load_iris()
x_train,y_train=iris.data,iris.target
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
print(model.coef_)


