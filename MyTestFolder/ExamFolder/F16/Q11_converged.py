import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data
heights = np.array([-2.1, -1.7, -1.5, -0.4, 0.0, 0.6, 0.8, 1.0, 1.1]).reshape(-1, 1)
player_labels = np.array([2, 2, 2, 1, 1, 1, 3, 3, 3])  # For visualization purposes

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(heights) # TODO MAKE SURE TO CHECK THE CLUSTER NUMBER
clusters = kmeans.labels_

# Print the cluster assignments
print("Cluster Assignments:", clusters)
