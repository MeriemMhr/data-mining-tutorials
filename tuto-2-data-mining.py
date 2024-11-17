# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:30:39 2023

"""

# Loading libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

# Creating a dataframe
data = np.array([['Black', 1, 1], ['Blue', 0, 0], ['Blue', -1, -1]])
column_names = ['Color', 'x1', 'x2']  # Changed 'y' to 'Color' to match the column name in data
row_names = ['A', 'B', 'C']
df = pd.DataFrame(data, columns=column_names, index=row_names)

# Constructing variables
X = df.iloc[:, 1:3]
y = df['Color']

# Building a KNN model with k = 2
knn = KNeighborsClassifier(n_neighbors=2).fit(X, y)

# Making prediction for a new observation (x1 = 0.1, x2 = 0.1)
new_obs = [[0.1, 0.1]]
prediction = knn.predict(new_obs)
print("Predicted class:", prediction)

# Predicting the target variable with predict_proba method
proba = knn.predict_proba(new_obs)
print("Predicted probabilities:", proba)

# Building a model with k = 2 and using the Euclidean distance function
knn1 = KNeighborsClassifier(n_neighbors=2, metric="euclidean").fit(X, y)



