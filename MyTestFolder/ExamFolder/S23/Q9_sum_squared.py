# Define the dataset
X = [2, 5, 8, 12, 13]

# Initial centroids
mu1 = 4
mu2 = 10

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

# Function to calculate total cost (sum of squared distances)
def calculate_total_cost(clusters, centroids):
    total_cost = 0
    for centroid, points in clusters.items():
        for point in points:
            total_cost += (point - centroid) ** 2
    return total_cost

# Initialize centroids list
centroids = [mu1, mu2]

# Iteratively apply K-means algorithm
for i in range(10):  # limit iterations to prevent infinite loop in case of no convergence
    clusters = assign_clusters(X, centroids)
    new_centroids = recompute_centroids(clusters)
    total_cost = calculate_total_cost(clusters, centroids)
    
    # Check for convergence (if centroids do not change)
    if set(new_centroids) == set(centroids):
        break
    else:
        centroids = new_centroids



# Return the final clusters and centroids
print("Clusters after first iteration:", clusters)
print("Centroids after first iteration:", new_centroids)
print("Total cost after first iteration:", total_cost)
