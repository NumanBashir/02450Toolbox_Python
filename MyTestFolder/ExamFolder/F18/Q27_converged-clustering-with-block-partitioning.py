import numpy as np # type: ignore

# Initial partitions
#TODO: Update based on the split of the cluster
clusters = {
    0: [0.1, 0.3],
    1: [0.5, 1.0],
    2: [2.2, 3.0, 4.1],
    3: [4.4, 4.7]
}

# Convert clusters to numpy arrays for easier manipulation
for key in clusters:
    clusters[key] = np.array(clusters[key])

# Calculate the new centroids
new_centroids = np.array([np.mean(cluster) for cluster in clusters.values()])

# Data points
#TODO: update based on dataset
data = np.array([0.1, 0.3, 0.5, 1.0, 2.2, 3.0, 4.1, 4.4, 4.7])

# Function to assign clusters based on closest centroid
def assign_clusters(data, centroids):
    clusters = {}
    for point in data:
        # Find the closest centroid
        closest_centroid = np.argmin(np.abs(centroids - point))
        if closest_centroid not in clusters:
            clusters[closest_centroid] = []
        clusters[closest_centroid].append(point)
    return clusters

# Assign data points to new clusters based on new centroids
new_clusters = assign_clusters(data, new_centroids)

# Sort and display the clusters for easier comparison
sorted_clusters = {key: sorted(value) for key, value in new_clusters.items()}
for key in sorted(sorted_clusters):
    print(f"Cluster {key + 1}: {sorted_clusters[key]}")
