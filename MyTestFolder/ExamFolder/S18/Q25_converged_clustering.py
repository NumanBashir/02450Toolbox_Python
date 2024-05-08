import numpy as np

# TODO Data points
data = np.array([1.0, 1.2, 1.5, 2.0, 2.2, 2.5, 3.0, 3.2])

# TODO Initial centroids based on the first three observations
centroids = np.array([1.0, 1.2, 1.5])

# Function to assign points to the nearest centroid
def assign_points_to_centroids(data, centroids):
    clusters = {}
    for point in data:
        # Calculate distance from the point to each centroid
        distances = np.abs(point - centroids)
        # Find the nearest centroid (index of the minimum distance)
        nearest = np.argmin(distances)
        if nearest not in clusters:
            clusters[nearest] = []
        clusters[nearest].append(point)
    return clusters

# Function to recalculate centroids
def recalculate_centroids(clusters):
    new_centroids = np.array([np.mean(clusters[key]) for key in sorted(clusters.keys())])
    return new_centroids

# Perform the k-means clustering algorithm
previous_centroids = None
iteration = 0
while not np.array_equal(centroids, previous_centroids):
    print(f"Iteration {iteration}, Centroids: {centroids}")
    clusters = assign_points_to_centroids(data, centroids)
    previous_centroids = centroids
    centroids = recalculate_centroids(clusters)
    iteration += 1

# Final clusters
final_clusters = {key: clusters[key] for key in sorted(clusters.keys())}
print("Final Clusters:", final_clusters)
