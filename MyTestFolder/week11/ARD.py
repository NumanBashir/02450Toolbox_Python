# Given data from the table (Pairwise Euclidean distance between the 10 first observations)
import numpy as np


distances_to_O1 = np.array([0, 393.5, 68.1, 165.4, 271.8, 200.6, 210.9, 206.1, 166.3, 365.0])

# Since we are calculating for O1, we should exclude the distance to itself and find the two nearest neighbors
# Exclude the first distance (0, as it is the distance to itself)
distances_to_O1 = distances_to_O1[1:]

# Find the indices of the two smallest distances (K=2 nearest neighbors)
K = 2
nearest_neighbors_indices = np.argsort(distances_to_O1)[:K]

# Calculate the density for O1
density_O1 = (1 / K * sum(distances_to_O1[nearest_neighbors_indices]))**-1

# Now, calculate the density for the two nearest neighbors (excluding O1)
densities_neighbors = []
for neighbor_index in nearest_neighbors_indices:
    # We calculate the distance from each neighbor to all others, except for itself and O1
    distances = np.concatenate((distances_to_O1[:neighbor_index], distances_to_O1[neighbor_index + 1:]))
    nearest_neighbor_distances = np.partition(distances, K)[:K]
    density_neighbor = (1 / K * sum(nearest_neighbor_distances))**-1
    densities_neighbors.append(density_neighbor)

# Calculate the ARD for O1
ard_O1 = density_O1 / np.mean(densities_neighbors)
print(ard_O1)

### Ikke noget rigtigt svar
