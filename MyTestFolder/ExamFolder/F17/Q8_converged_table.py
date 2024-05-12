import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data
heights = np.array([5.7, 6.0, 6.2, 6.3, 6.4, 6.6, 6.7, 6.9, 7.0, 7.4]).reshape(-1, 1)
player_labels = np.array([1, 1, 1, 1, 2, 2, 2, 1, 1, 2])  # For visualization purposes

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(heights) # TODO MAKE SURE TO CHECK THE CLUSTER NUMBER
clusters = kmeans.labels_

# Print the cluster assignments
print("Cluster Assignments:", clusters)
