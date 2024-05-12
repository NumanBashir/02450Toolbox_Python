import numpy as np

# Heights data
heights = np.array([5.7, 6.0, 6.2, 6.3, 6.4, 6.6, 6.7, 6.9, 7.0, 7.4]).reshape(-1, 1)

# Function to calculate the Euclidean distance matrix
def calculate_distance_matrix(heights):
    return np.sqrt(((heights - heights.T)**2))

# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    sorted_distances = np.sort(distances)
    return 1 / np.mean(sorted_distances[1:K+1])  # Exclude the zero distance (self)

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(heights)

# Calculate densities for each observation
K = 3
densities = np.array([calculate_density(distance_matrix[i], K) for i in range(len(heights))])

# Calculate ARD for O10
index_o10 = 9  # Zero-indexed position of O10
nearest_indices = np.argsort(distance_matrix[index_o10])[1:K+1]  # Get indices of nearest K neighbors
average_neighbor_density = np.mean(densities[nearest_indices])
ard_o10 = densities[index_o10] / average_neighbor_density

# Print ARD for O10
print(f"Average Relative Density (ARD) for O10: {ard_o10:.3f}")
