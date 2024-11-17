# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:50:18 2023

@author: mehri
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest

# Importing the Toyota Corolla dataset
toyota_df = pd.read_csv("C:/Users/mehri/Downloads/ToyotaCorolla.csv")


# Pre-processing
# Task 1: 
## Dropping 'Id' and 'Model' columns
toyota_df = toyota_df.drop(columns=['Id', 'Model'], axis = 1)

## Setting 'Price' as the target variable & the rest of variables as predictors
X = toyota_df.drop(columns=['Price'])
y = toyota_df['Price']


## Splitting the data into training (70%) & test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# Task 2: Ridge Regression Model 1
## Alpha = 1

# Run ridge regression with penalty equals to 1
ridge = Ridge(alpha=1) #equivalent to lambda, when lambda = 0 it's similar to linear regression
ridge_model = ridge.fit(X_train,y_train) # forcing the coefficient estimatesto be reduced

# Print the coefficients
ridge_model.coef_

# Generate the prediction value from the test data
y_test_pred = ridge_model.predict(X_test)

# Calculate the MSE
ridge_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using ridge with penalty of 1 = "+str(ridge_mse))

## Performance of Ridge Model 1 - MSE: 1843924.5887054754


 
# Task 3: Removing anomalies/outliers using Isolation Forest Model
## Contamination = 0.05
iforest = IsolationForest(n_estimators=100, contamination=.05)

pred = iforest.fit_predict(toyota_df)
score = iforest.decision_function(toyota_df)

from numpy import where
anomaly_index = where(pred==-1)
anomaly_values = toyota_df.iloc[anomaly_index]



# Task 4: Removing observations that are flagged as anomalies from the training/test data sets
## Filtering out anomalies from the training set
X_train_no_anomalies = X_train[pred[X_train.index] != -1]
y_train_no_anomalies = y_train[pred[X_train.index] != -1]

## Filtering out anomalies from the test set
X_test_no_anomalies = X_test[pred[X_test.index] != -1]
y_test_no_anomalies = y_test[pred[X_test.index] != -1]



# Task 5: Ridge Regression Model 2
## Alpha = 1 - Retraining Ridge Regression Model without anomalies
ridge_model_no_anomalies = ridge.fit(X_train_no_anomalies, y_train_no_anomalies)

# Generate the prediction value from the modified test data
y_test_pred_no_anomalies = ridge_model_no_anomalies.predict(X_test_no_anomalies)

# Calculate the MSE without anomalies
ridge_mse_no_anomalies = mean_squared_error(y_test_no_anomalies, y_test_pred_no_anomalies)
print("Test MSE using ridge without anomalies = " + str(ridge_mse_no_anomalies))

## Performance of Ridge Model 2 - MSE: 1599979.4098647092

