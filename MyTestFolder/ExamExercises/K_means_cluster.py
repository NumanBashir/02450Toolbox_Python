# Define the dataset
X = [42, 60, 17, 48, 12]

# Initial centroids
mu1 = 17
mu2 = 12

# Function to assign clusters
def assign_clusters(X, centroids):
    clusters = {c: [] for c in centroids}
    for x in X:
        closest_centroid = min(centroids, key=lambda c: abs(x - c))
        clusters[closest_centroid].append(x)
    return clusters

# Function to recompute centroids
def recompute_centroids(clusters):
    new_centroids = []
    for cluster in clusters.values():
        if cluster:  # to avoid division by zero
            new_centroids.append(sum(cluster) / len(cluster))
    return new_centroids

# Initialize centroids list
centroids = [mu1, mu2]

# Iteratively apply K-means algorithm
for i in range(10):  # limit iterations to prevent infinite loop in case of no convergence
    clusters = assign_clusters(X, centroids)
    new_centroids = recompute_centroids(clusters)
    
    # Check for convergence (if centroids do not change)
    if set(new_centroids) == set(centroids):
        break
    else:
        centroids = new_centroids

# Return the final clusters and centroids
print(clusters)
print(centroids)
