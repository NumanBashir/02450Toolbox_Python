import numpy as np
import matplotlib.pyplot as plt

# Initial positions of observations and centroids
observations = np.array([0, 2, 4, 5, 6, 7, 14])
centroids = np.array([0, 5, 8])

# Function to assign observations to the nearest centroid
def assign_observations(observations, centroids):
    assignments = []
    for obs in observations:
        distances = np.abs(centroids - obs)
        nearest_centroid = np.argmin(distances)
        assignments.append(nearest_centroid)
    return np.array(assignments)

# Function to compute new centroids
def compute_new_centroids(observations, assignments, num_centroids):
    new_centroids = []
    for i in range(num_centroids):
        cluster_points = observations[assignments == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points)
        else:
            new_centroid = centroids[i]  # Keep the old centroid if no points are assigned to it
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# Assign observations to nearest centroid
assignments = assign_observations(observations, centroids)

# Compute new centroids
new_centroids = compute_new_centroids(observations, assignments, len(centroids))

# Print results
print("Initial centroids:", centroids)
print("Assignments:", assignments)
print("New centroids:", new_centroids)

# Plot the results
plt.scatter(observations, np.zeros_like(observations), color='purple', label='Observations')
plt.scatter(centroids, np.zeros_like(centroids), color='black', marker='x', label='Initial Centroids')
plt.scatter(new_centroids, np.zeros_like(new_centroids), color='red', marker='x', label='New Centroids')
plt.legend()
plt.xlabel('x')
plt.title('K-means: Initial and New Centroid Locations')
plt.show()
