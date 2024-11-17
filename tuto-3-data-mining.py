# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:06:20 2023

@author: mehri
"""

# Import Sheet4.csv data
import pandas as pd

cereals_df = pd.read_csv("C:/Users/mehri/OneDrive/Desktop/Fall 2023/Data Mining & Visualization - INSY 662/Individual Assignments/Assignment 3/Sheet4.csv")

# Pre-processing
# Checking missing values
print(cereals_df.isnull().sum())

# Drop rows with missing values
cereals_df = cereals_df.dropna()

# Check missing values again after dropping
print(cereals_df.isnull().sum())

# Check the shape of the DataFrame after dropping rows
print(cereals_df.shape)


# Dropping irrelevant predictors
cereals_df = cereals_df.drop(columns=['Name'], axis = 1)

# Dummify all categorical predictors
# Using get_dummies - 1st approach
cereals_df = pd.get_dummies(cereals_df, columns=['Manuf', 'Type'], drop_first=True)

# Construct predictor & target variables
X = cereals_df.iloc[:, 1:12].join(cereals_df.iloc[:, 14:21])
y = cereals_df.iloc[:, 12]


# Split the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)



## Random Forest
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': [3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Create a Random Forest Classifier with verbose=True
randomforest = RandomForestClassifier(random_state=0, verbose=True)

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=randomforest, param_grid=param_grid, cv=5)

# Fit the model to your data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_test_pred = best_model.predict(X_test)

# Evaluate model accuracy
from sklearn.metrics import accuracy_score
accuracy_rf = accuracy_score(y_test, y_test_pred)      
print(accuracy_rf)



## Gradient Boosting
# Import GradientBoostingClassifier package
from sklearn.ensemble import GradientBoostingClassifier

# Define the hyperparameter grid for Gradient Boosting
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': [3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Create a Gradient Boosting Classifier with verbose=True
gradient_boosting = GradientBoostingClassifier(random_state=0, verbose=True)

# Create a GridSearchCV object
grid_search_gb = GridSearchCV(estimator=gradient_boosting, param_grid=param_grid, cv=5)

# Fit the model to your data
grid_search_gb.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(grid_search_gb.best_params_)

# Get the best model
best_model_gb = grid_search_gb.best_estimator_

# Make predictions
y_test_pred_gb = best_model_gb.predict(X_test)

# Calculate accuracy score
accuracy_gb = accuracy_score(y_test, y_test_pred_gb)
print(accuracy_gb)
