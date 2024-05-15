import numpy as np

def compute_density(distances, K):
    """
    Compute the density of a point based on the distances to its K-nearest neighbors.
    
    Parameters:
    distances (array): Array of distances between the point and each other point.
    K (int): Number of nearest neighbors to consider.
    
    Returns:
    float: The density of the point.
    """
    # Find the K-nearest distances
    nearest_distances = np.sort(distances)[:K]
    
    # Calculate the density
    density = 1 / (np.mean(nearest_distances) + 1e-10)  # Added small value to avoid division by zero
    
    return density

def compute_ard(distance_matrix, observation_index, K):
    """
    Compute the average relative KNN density (a.r.d.) of a given observation.
    
    Parameters:
    distance_matrix (2D array): Matrix of distances between all observations.
    observation_index (int): Index of the observation for which to compute the a.r.d.
    K (int): Number of nearest neighbors to consider.
    
    Returns:
    float: The average relative density of the observation.
    """
    # Get distances for the target observation
    distances = distance_matrix[observation_index]
    
    # Find the indices of the K-nearest neighbors
    nearest_indices = np.argsort(distances)[:K]
    
    # Compute density of the target observation
    density_x = compute_density(distances, K)
    
    # Compute densities of the K-nearest neighbors
    densities_neighbors = [compute_density(distance_matrix[i], K) for i in nearest_indices]
    
    # Compute average relative density
    ard = density_x / (np.mean(densities_neighbors) + 1e-10)  # Added small value to avoid division by zero
    
    return ard

# Example usage:
distance_matrix = np.array([
    [0.00, 4.84, 0.50, 4.11, 1.07, 4.10, 4.71, 4.70, 4.93],
    [4.84, 0.00, 4.00, 5.96, 4.12, 2.01, 5.36, 3.59, 3.02],
    [0.50, 4.00, 0.00, 4.07, 0.72, 3.75, 4.66, 4.48, 4.64],
    [4.11, 5.96, 4.07, 0.00, 4.48, 4.69, 2.44, 3.68, 4.15],
    [1.07, 4.12, 0.72, 4.48, 0.00, 3.54, 4.96, 4.62, 4.71],
    [4.10, 2.01, 3.75, 4.69, 3.54, 0.00, 3.72, 2.23, 1.95],
    [4.71, 5.36, 4.66, 2.44, 4.96, 3.72, 0.00, 2.03, 2.73],
    [4.70, 3.59, 4.48, 3.68, 4.62, 2.23, 2.03, 0.00, 0.73],
    [4.93, 3.02, 4.64, 4.15, 4.71, 1.95, 2.73, 0.73, 0.00]
])

observation_index = 8  # Index of o9
K = 2

ard = compute_ard(distance_matrix, observation_index, K)
print(f"a.r.d. of observation o{observation_index + 1} using K = {K} is approximately {ard:.3f}")
