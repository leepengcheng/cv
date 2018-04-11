#coding:utf-8
'''
线性回归和线性逻辑
'''
from sklearn import linear_model
from sklearn import datasets #数据库
from sklearn.model_selection import cross_val_predict #交叉验证
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb





#########线性回归############
x_train=np.random.randn(10,2)
y_train=x_train.dot(np.array([[0.2],[0.3]]))
x_test=np.random.randn(100,2)

linear = linear_model.LinearRegression() #创建线性回归模型
linear.fit(x_train, y_train)    #训练模型
linear.score(x_train, y_train)  #check score
print('Coefficient: \n', linear.coef_) #拟合求出的回归系数
print('Intercept: \n', linear.intercept_) #截距
predicted= linear.predict(x_test) ##验证测试值
##########逻辑回归############
x_train=np.random.randn(100,2)
y_train=np.random.randint(0,2,100)
x_test=np.random.randn(100,2)
logistic=linear_model.LogisticRegression() #创建逻辑回归模型
logistic.fit(x_train, y_train) 
logistic.score(x_train, y_train) 
print('Coefficient: \n', logistic.coef_) #求出的逻辑回归系数
print('Intercept: \n', logistic.intercept_) #截距
predicted= logistic.predict(x_test) ##验证测试值



#线性回归
lr = linear_model.LinearRegression()
# #加载波士顿数据
boston = datasets.load_boston()
x_boston,y_boston=boston.data,boston.target
ymin_boston,ymax_boston=y_boston.min(),y_boston.max()
#交叉验证(Cross Validation)的好处是可以从有限的数据中获得尽可能多的有效信
#息，从而可以从多个角度去学习样本，避免陷入局部的极值。在这个过程中，无论是训练样本还
#是测试样本都得到了尽可能多的学习。下例中将数据分为10份,每次迭代使用9个作为训练集，1个作为测试集
#cross_val_predic用线性回归建立模型，并返回预测值
predicted = cross_val_predict(lr, x_boston, y_boston, cv=10)
print('Coefficient: \n', lr.coef_) #拟合求出的回归系数
fig, ax = plt.subplots()
#散点图,X为房价的实际值,Y为线性回归的预测房价
ax.scatter(y_boston, predicted) 
#创建中间的分割线，离该线越近预测越准确
ax.plot([ymin_boston, ymax_boston], [ymin_boston, ymax_boston], 'k--', lw=4) 
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()