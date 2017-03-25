#Import Library
from sklearn.linear_model import LogisticRegression
import numpy as np
#Assumed you have, X (predictor) and Y (target)
#for training data set and x_test(predictor)
#of test_dataset
#Create logistic regression object
X=np.random.randn(10,2)
y=X.dot(np.array([[0.2],[0.3]]))
# x_test=np.random.randn(100,2)
model = LogisticRegression()
#Train the model using the training sets
#and check score
model.fit(X, y)
model.score(X, y)
#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
# predicted= model.predict(x_test)