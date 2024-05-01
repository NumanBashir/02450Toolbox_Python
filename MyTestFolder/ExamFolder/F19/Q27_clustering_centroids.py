import numpy as np
from sklearn.cluster import KMeans

# Data points
x = np.array([0.4, 0.5, 1.1, 2.2, 2.6, 3.0, 3.6, 3.7, 4.9, 5.0])

# Initial cluster centers
initial_centers = np.array([2.4, 3.3, 3.5]).reshape(-1, 1)

# K-means clustering
kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, random_state=42)
kmeans.fit(x.reshape(-1, 1))

# Cluster centers after fitting
final_centers = kmeans.cluster_centers_

# Assignments of points to clusters
assignments = kmeans.labels_

# Organize points into their respective clusters
clusters = {i: [] for i in range(3)}
for point, label in zip(x, assignments):
    clusters[label].append(point)

# Sort clusters for easier reading
sorted_clusters = {key: sorted(value) for key, value in clusters.items()}

print("Clusters:", sorted_clusters)
print("Final Cluster Centers:", final_centers)
