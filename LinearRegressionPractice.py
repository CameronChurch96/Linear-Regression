#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#creating the datasets and creating the x and y variables
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#creating subsets of data to traing to computer and test it in the next step
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#process of training and testing the computer
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#the computer will test the linear regression and we can compare it to the data
y_pred = regressor.predict(x_test)

#visualizing the training data on a graph
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title('Salary vs. Years of Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the test data on a graph
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs. Years of Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(regressor.coef_)
print(regressor.intercept_)