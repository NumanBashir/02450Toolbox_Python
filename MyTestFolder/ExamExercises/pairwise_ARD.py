# Given the correct pairwise distances from the provided matrix
import numpy as np


correct_distance_matrix = np.array([
    [0.0, 4.2, 8.3, 3.9, 3.8, 4.6, 6.3, 4.8, 7.1, 4.9],
    [4.2, 0.0, 7.4, 2.6, 3.0, 3.2, 5.3, 3.1, 6.6, 4.6],
    [8.3, 7.4, 0.0, 6.3, 7.1, 5.5, 2.8, 5.4, 2.4, 5.3],
    [3.9, 2.6, 6.3, 0.0, 1.5, 1.6, 4.1, 1.8, 5.3, 2.4],
    [3.8, 3.0, 7.1, 1.5, 0.0, 2.4, 4.9, 2.8, 5.8, 3.2],
    [4.6, 3.2, 5.5, 1.6, 2.4, 0.0, 3.7, 1.7, 4.8, 2.3],
    [6.3, 5.3, 2.8, 4.1, 4.9, 3.7, 0.0, 3.8, 1.9, 3.6],
    [4.8, 3.1, 5.4, 1.8, 2.8, 1.7, 3.8, 0.0, 4.9, 2.1],
    [7.1, 6.6, 2.4, 5.3, 5.8, 4.8, 1.9, 4.9, 0.0, 4.4],
    [4.9, 4.6, 5.3, 2.4, 3.2, 2.3, 3.6, 2.1, 4.4, 0.0]
])

# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
density_o5 = calculate_density(correct_distance_matrix[4], K=2)
density_o4 = calculate_density(correct_distance_matrix[3], K=2)
density_o6 = calculate_density(correct_distance_matrix[5], K=2)

print("Densities of o5, o4 and o6:")
print(density_o5, density_o4, density_o6)

ard_o5_solution = density_o5 / ((density_o4 + density_o6) / 2)

print("ARD: " + str(ard_o5_solution))

# To make the script more readable, let's define a function that creates the proximity matrix and prints it in a readable format.

def create_and_print_proximity_matrix(distance_matrix):
    # Invert the distances to create proximity values, avoid division by zero
    proximity_matrix = 1 / np.where(distance_matrix > 0, distance_matrix, np.inf)
    # Set the diagonal to zeros
    np.fill_diagonal(proximity_matrix, 0)
    
    # Convert the proximity matrix to a string with rounded values for better readability
    proximity_matrix_str = np.array2string(proximity_matrix, precision=3)
    
    print("Proximity Matrix:\n" + proximity_matrix_str)

# Call the function with the corrected distance matrix
print(create_and_print_proximity_matrix(correct_distance_matrix))



