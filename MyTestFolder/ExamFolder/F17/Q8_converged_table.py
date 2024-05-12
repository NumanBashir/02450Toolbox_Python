import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data
heights = np.array([5.7, 6.0, 6.2, 6.3, 6.4, 6.6, 6.7, 6.9, 7.0, 7.4]).reshape(-1, 1)
player_labels = np.array([1, 1, 1, 1, 2, 2, 2, 1, 1, 2])  # For visualization purposes

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(heights)
clusters = kmeans.labels_

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(heights, np.zeros_like(heights), c=clusters, s=100, cmap='viridis')
plt.title('K-Means Clustering of Basketball Player Heights')
plt.xlabel('Height')
plt.yticks([])
plt.grid(True)

# Adding player labels for clarity
for i, txt in enumerate(player_labels):
    plt.annotate(f'O{txt}', (heights[i], 0))

plt.show()

# Print the cluster assignments
print("Cluster Assignments:", clusters)
