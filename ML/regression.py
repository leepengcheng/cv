#coding:utf-8
from sklearn import linear_model
import numpy as np

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
print(predicted)
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
print(predicted)

