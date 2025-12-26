# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
Data Preparation: Load the dataset and separate the independent features ($X$) from the dependent target variable ($y$).

### Step2
Feature Scaling: Normalize the features to bring all variables to a similar scale, ensuring faster and more stable convergence.

### Step3
Matrix Initialization: Add a column of ones to the feature matrix for the intercept and initialize the weight vector ($\theta$) with zeros.

### Step4
Cost Computation: Calculate the Mean Squared Error (MSE) to quantify the difference between the model's predictions and actual values.

### Step5
Gradient Descent Optimization: Iteratively update the weights by moving in the opposite direction of the gradient to minimize the cost function.

### Step6
Model Validation: Plot the cost history to confirm convergence and use the optimized weights to perform final predictions.

## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

# load the boston dataset
from sklearn.datasets import fetch_openml
boston = fetch_openml(name="boston", version=1, as_frame=True)

# defining feature matrix(X) and response vector(y)
X= boston.data y=boston.target X=X.astype(float) y=y.astype(float)

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.4,random_state=1)

# create linear regression object
reg=linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print("Coefficients: ", reg.coef_)

# variance score: 1 means perfect prediction
print("Variance score: {}".format(reg.score(X_test, y_test)))

# plot for residual error
## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color = "green", s= 10, label = 
'Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color ="blue", s= 10, label ='Test 
data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth= 2)

## plotting legend
plt.legend(loc ="upper right")

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()

```
## Output:

<img width="2339" height="3309" alt="Untitled7 - Jupyter Notebook" src="https://github.com/user-attachments/assets/dd791973-bd96-4d36-84df-13ad26748f0d" />

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
