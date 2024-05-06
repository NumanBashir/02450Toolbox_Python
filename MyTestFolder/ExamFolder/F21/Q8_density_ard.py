# Given the correct pairwise distances from the provided matrix
import numpy as np


correct_distance_matrix = np.array([
    [0.0, 53.8, 87.0, 67.4, 67.5, 71.2, 65.2, 117.9, 56.1, 90.3, 109.8],
    [53.8, 0.0, 69.9, 75.5, 62.9, 58.0, 63.0, 135.0, 84.1, 107.9, 131.5],
    [87.0, 69.9, 0.0, 49.7, 38.5, 19.3, 35.5, 91.8, 76.9, 78.7, 89.1],
    [67.4, 75.5, 49.7, 0.0, 24.2, 47.2, 47.0, 62.3, 33.4, 37.2, 60.0],
    [67.5, 62.9, 38.5, 24.2, 0.0, 37.7, 41.7, 79.5, 52.4, 60.2, 78.9],
    [71.2, 58.0, 19.3, 47.2, 37.7, 0.0, 21.5, 95.6, 68.3, 78.4, 91.0],
    [65.2, 63.0, 35.5, 47.0, 41.7, 21.5, 0.0, 96.0, 64.3, 75.5, 89.4],
    [117.9, 135.0, 91.8, 62.3, 79.5, 95.6, 96.0, 0.0, 66.9, 44.3, 24.2],
    [56.1, 84.1, 76.9, 33.4, 52.4, 68.3, 64.3, 66.9, 0.0, 39.2, 60.7],
    [90.3, 107.9, 78.7, 37.2, 60.2, 78.4, 75.5, 44.3, 39.2, 0.0, 39.4],
    [109.8, 131.5, 89.1, 60.0, 78.9, 91.0, 89.4, 24.2, 60.7, 39.4, 0.0]
])

# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
# TODO YOU NEED TO WRITE DOWN THE K NUMBER OF NEAREST NEIGHBORS, IN THIS QUESTION IT WAS K=3, WHY WE WRITE DOWN THE THREE NEAREST NEIGHBORS

density_o5 = calculate_density(correct_distance_matrix[4], K=3) # THE OBSERVATION WE ARE LOOKING AT 

# ITS NEAREST NEIGHBORS
density_o3 = calculate_density(correct_distance_matrix[2], K=3)
density_o4 = calculate_density(correct_distance_matrix[3], K=3)
density_o6 = calculate_density(correct_distance_matrix[5], K=3)

print("Densities of o5, o3, 04 and o6:")
print(density_o5, density_o3, density_o4, density_o6)

ard_o5_solution = density_o5 / ((density_o4 + density_o6 + density_o3) / 2)

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



