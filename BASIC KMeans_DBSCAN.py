import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons

# Creating the two dimensional 'Moons' dataset
X,Y = make_moons(n_samples = 200, random_state = 100) # X = two dimensions, Y = label
print(X[:5,])

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Implementing K-Means CLustering
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X_scaled)
plt.figure(figsize = (8,6))
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='plasma')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Implementing DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=12)
dbscan.fit(X_scaled)
plt.figure(figsize = (8,6))
plt.scatter(X[:,0], X[:,1], c=dbscan.labels_, cmap='plasma')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
