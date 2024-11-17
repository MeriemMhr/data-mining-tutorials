# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:49:31 2023

@author: mehri
"""

# Import libraries
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# Import cereals dataset
cereals_df = pd.read_csv("C:/Users/mehri/Downloads/cereals.CSV")

# Pre-processing
# Task 0: Cleaning the data & dropping irrelevant/problematic variables

# Checking missing values
print(cereals_df.isnull().sum())

# Drop missing values
cereals_df = cereals_df.dropna()
print(cereals_df)
         

# Agglomerative Clustering
# Task 1: Clustering with complete linkage

# Using sklearn
X = cereals_df[["Calories", "Protein", "Fat", "Fiber", "Carbo", "Sodium", "Sugars", "Potass", "Vitamins"]]
cluster = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='euclidean')
cluster_labels = cluster.fit_predict(X)

# Print cluster labels
print(cluster_labels)

# Plot - Facultative
from matplotlib import pyplot
# 'cereals_df['Calories']' represents the x-coordinates ;'cereals_df['Sodium']' represents the y-coordinates
plot = pyplot.scatter(cereals_df['Calories'], cereals_df['Sodium'], c=cluster_labels, cmap='rainbow')

# Add a legend to the scatter plot:
pyplot.legend(*plot.legend_elements(), title='clusters')
pyplot.show()

# Reporting the number of cereals in each cluster (1 and 2)
num_cereals_cluster1 = (cluster_labels == 0).sum()
num_cereals_cluster2 = (cluster_labels == 1).sum()

print(f"Number of cereals in cluster #1: {num_cereals_cluster1}")
print(f"Number of cereals in cluster #2: {num_cereals_cluster2}")


# K-Mean Clustering
# Task 2: K-Mean Clustering with k = 2
X = cereals_df[["Calories", "Protein", "Fat", "Fiber", "Carbo", "Sodium", "Sugars", "Potass", "Vitamins"]]


# Standardizing the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2)
model = kmeans.fit(X_std)
labels = model.predict(X_std)

# Reporting the number of cereals in each cluster (1 and 2)
num_cereals_cluster1 = (labels == 0).sum()
num_cereals_cluster2 = (labels == 1).sum()

print(f"Number of cereals in cluster #1: {num_cereals_cluster1}")
print(f"Number of cereals in cluster #2: {num_cereals_cluster2}")


# Task 3: Explanation of each cluster's characteristics
## Cf. Written Response in Word Doc.

