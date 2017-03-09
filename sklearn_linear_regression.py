from sklearn import linear_model
import numpy as np
x_train=np.random.randn(10,2)
y_train=x_train.dot(np.array([[0.2],[0.3]]))
x_test=np.random.randn(100,2)
# #Create linear regression object
linear = linear_model.LinearRegression()
# #Train the model using the training sets and
# #check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
# #Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
# #Predict Output
predicted= linear.predict(x_test)
print(predicted)