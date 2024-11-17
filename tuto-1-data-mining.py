# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:12:27 2023

@author: mehri
"""
# Import data
import pandas as pd
df = pd.read_csv("C:/Users/mehri/Downloads/ToyotaCorolla.csv")

# Construct variables (Age_08_04, KM, HP, Automatic, CC, Doors, Cylinders, Gears, Weight)
X = df.iloc[:,3:12] #iloc is specifically for slicing the dataframe
y = df['Price']


## Detecting multicollinearity
# Create VIF dataframe
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a VIF dataframe
X1 = add_constant(X)
vif_data = pd.DataFrame()
vif_data["feature"] = X1.columns

# Calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]

print(vif_data)

# High VIF values indicate that a variable can be predicted linearly 
# from the other independent variables in the dataset, which suggests multicollinearity


# Removing "Cylinders" variable

X = X.drop(columns=["Cylinders"])
print(X)


## Standardize the predictors
X.describe()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, columns=X.columns) # transform into a dataframe and add column names


################# Model building ######################################

## 2. Linear regression (sklearn)
# Load libraries
from sklearn.linear_model import LinearRegression

# Run linear regression
lm1 = LinearRegression() # create a blank model object and instantiate the model
model1 = lm1.fit(scaled_X, y) # fit the regression model using the predictors X and the target y

# View results
model1.intercept_
model1.coef_


## 3. Cross-Validation 1 (Validation set approach)
# Separate the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.35, random_state = 662)

# Run linear regression based on the training data
lm2 = LinearRegression()
model2 = lm2.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred = model2.predict(X_test)
# y_test_preddf=pd.DataFrame(y_test_pred, columns=['Predicted rating']) if need to convert to dataframe

# Calculate the MSE for the test set
from sklearn.metrics import mean_squared_error
lm2_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using validation set approach = "+str(lm2_mse))


## 4. Ridge regression, Alpha = 1
from sklearn.linear_model import Ridge

# Run ridge regression with penalty equals to 1
ridge = Ridge(alpha=1)
ridge_model = ridge.fit(X_train,y_train)

# Print the coefficients
ridge_model.coef_

# Generate the prediction value from the test data
y_test_pred = ridge_model.predict(X_test)

# Calculate the MSE
ridge_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using ridge with penalty of 1 = "+str(ridge_mse))


## 5. LASSO, Alpha = 1
from sklearn.linear_model import Lasso

# Run LASSO with penalty = 1
lasso = Lasso(alpha=1)
lasso_model = lasso.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred = lasso_model.predict(X_test)


# Calculate the MSE
lasso_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using lasso with penalty of 1 = "+str(lasso_mse))

# Print the coefficients
lasso_model.coef_


# Among these options, MSE_Ridge (1,723,936) has the lowest MSE, so it is 
# the best model based on the criteria of minimizing the MSE. 
# Lower MSE values indicate better model performance in terms of fitting the data.


################# Ridge Regressions ######################################

## 6. Ridge regressions, Changing Alphas

## 6.1. Ridge regression, Alpha = 10
from sklearn.linear_model import Ridge

# Run ridge regression with penalty equals to 10
ridge = Ridge(alpha=10)
ridge_model = ridge.fit(X_train,y_train)

# Print the coefficients
ridge_model.coef_

# Generate the prediction value from the test data
y_test_pred = ridge_model.predict(X_test)

# Calculate the MSE
ridge_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using ridge with penalty of 10 = "+str(ridge_mse))


## 6.2. Ridge regression, Alpha = 100
from sklearn.linear_model import Ridge

# Run ridge regression with penalty equals to 100
ridge = Ridge(alpha=100)
ridge_model = ridge.fit(X_train,y_train)

# Print the coefficients
ridge_model.coef_

# Generate the prediction value from the test data
y_test_pred = ridge_model.predict(X_test)

# Calculate the MSE
ridge_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using ridge with penalty of 100 = "+str(ridge_mse))


## 6.3. Ridge regression, Alpha = 1000
from sklearn.linear_model import Ridge

# Run ridge regression with penalty equals to 1000
ridge = Ridge(alpha=1000)
ridge_model = ridge.fit(X_train,y_train)

# Print the coefficients
ridge_model.coef_

# Generate the prediction value from the test data
y_test_pred = ridge_model.predict(X_test)

# Calculate the MSE
ridge_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using ridge with penalty of 1000 = "+str(ridge_mse))


## 6.4. Ridge regression, Alpha = 10000
from sklearn.linear_model import Ridge

# Run ridge regression with penalty equals to 10000
ridge = Ridge(alpha=10000)
ridge_model = ridge.fit(X_train,y_train)

# Print the coefficients
ridge_model.coef_

# Generate the prediction value from the test data
y_test_pred = ridge_model.predict(X_test)

# Calculate the MSE
ridge_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using ridge with penalty of 10000 = "+str(ridge_mse))



################# LASSO Regressions ######################################

## 7. LASSO regressions, Changing Alphas

# 7.1. LASSO, Alpha = 10

# Run LASSO with penalty = 10
lasso = Lasso(alpha=10)
lasso_model = lasso.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred = lasso_model.predict(X_test)

# Calculate the MSE
lasso_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using lasso with penalty of 10 = "+str(lasso_mse))

# Print the coefficients
lasso_model.coef_


# 7.2. LASSO, Alpha = 100

# Run LASSO with penalty = 100
lasso = Lasso(alpha=100)
lasso_model = lasso.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred = lasso_model.predict(X_test)

# Calculate the MSE
lasso_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using lasso with penalty of 100 = "+str(lasso_mse))

# Print the coefficients
lasso_model.coef_


# 7.3. LASSO, Alpha = 1000

# Run LASSO with penalty = 1000
lasso = Lasso(alpha=1000)
lasso_model = lasso.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred = lasso_model.predict(X_test)

# Calculate the MSE
lasso_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using lasso with penalty of 1000 = "+str(lasso_mse))

# Print the coefficients
lasso_model.coef_


# 7.4. LASSO, Alpha = 10000

# Run LASSO with penalty = 10000
lasso = Lasso(alpha=10000)
lasso_model = lasso.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred = lasso_model.predict(X_test)

# Calculate the MSE
lasso_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using lasso with penalty of 10000 = "+str(lasso_mse))

# Print the coefficients
lasso_model.coef_


