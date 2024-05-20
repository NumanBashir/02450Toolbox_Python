import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data
heights = np.array([42.0, 38.3, 40.1, 34.2, 50.9, 30.3, 68.6, 19.4]).reshape(-1, 1)
player_labels = np.array([1, 1, 1, 1, 2, 2, 2, 1])  # For visualization purposes

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(heights) # TODO MAKE SURE TO CHECK THE CLUSTER NUMBER
clusters = kmeans.labels_

# Print the cluster assignments
print("Cluster Assignments:", clusters)

# THE NUMBER IN THE PRINT INDICATES THE CLUSTER ASSIGNMENTS FOR EACH DATA POINT
