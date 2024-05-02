# Given the correct pairwise distances from the provided matrix
import numpy as np

# Taken from F19_Q6

# Numbers have been changed in o1, o2, o6, o8,

correct_distance_matrix = np.array([
    [0.0, 1.3, 4.1, 3.8, 4.5, 2.4, 3.2, 2.7, 3.0, 3.9],
    [1.3, 0.0, 3.2, 3.1, 4.7, 2.3, 2.6, 2.2, 2.7, 4.2],
    [4.1, 3.2, 0.0, 0.4, 4.9, 2.7, 1.1, 1.6, 2.4, 4.8],
    [3.8, 3.1, 0.4, 0.0, 4.6, 2.5, 0.9, 1.3, 2.1, 4.5],
    [4.5, 4.7, 4.9, 4.6, 0.0, 3.1, 4.4, 3.7, 2.8, 2.3],
    [2.4, 2.3, 2.7, 2.5, 3.1, 0.0, 1.8, 1.2, 0.9, 2.8],
    [3.2, 2.6, 1.1, 0.9, 4.4, 1.8, 0.0, 1.0, 1.7, 4.1],
    [2.7, 2.2, 1.6, 1.3, 3.7, 1.2, 1.0, 0.0, 1.1, 3.6],
    [3.0, 2.7, 2.4, 2.1, 2.8, 0.9, 1.7, 1.1, 0.0, 2.9],
    [3.9, 4.2, 4.8, 4.5, 2.3, 2.8, 4.1, 3.6, 2.9, 0.0]
])

# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
density_o1 = calculate_density(correct_distance_matrix[0], K=3)
# density_o2 = calculate_density(correct_distance_matrix[1], K=3)
# density_o6 = calculate_density(correct_distance_matrix[5], K=3)
# density_o8 = calculate_density(correct_distance_matrix[7], K=3)

print("Density of o1")
print(density_o1)