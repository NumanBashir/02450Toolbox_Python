import numpy as np

correct_distance_matrix = np.array([
    [0.0, 5.0, 7.7, 6.1, 4.2, 11.0, 7.3, 9.0, 11.3, 1.4],
    [5.0, 0.0, 5.4, 4.0, 7.5, 7.9, 5.3, 6.8, 11.9, 3.5],
    [7.7, 5.4, 0.0, 5.2, 7.2, 6.1, 7.8, 6.7, 12.9, 6.4],
    [6.1, 4.0, 5.2, 0.0, 5.1, 5.4, 8.4, 3.3, 8.1, 4.8],
    [4.2, 7.5, 7.2, 5.1, 0.0, 8.7, 8.8, 6.6, 7.7, 4.1],
    [11.0, 7.9, 6.1, 5.4, 8.7, 0.0, 12.0, 4.2, 9.3, 9.8],
    [7.3, 5.3, 7.8, 8.4, 8.8, 12.0, 0.0, 11.0, 16.3, 6.7],
    [9.0, 6.8, 6.7, 3.3, 6.6, 4.2, 11.0, 0.0, 6.2, 7.8],
    [11.3, 11.9, 12.9, 8.1, 7.7, 9.3, 16.3, 6.2, 0.0, 10.4],
    [1.4, 3.5, 6.4, 4.8, 4.1, 9.8, 6.7, 7.8, 10.4, 0.0]
])


# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
density_o2 = calculate_density(correct_distance_matrix[2], K=2) # THE ONE WE ARE LOOKING AT

# THE TWO NEAREST NEIGHBORS
density_o3 = calculate_density(correct_distance_matrix[1], K=2) 
density_o10 = calculate_density(correct_distance_matrix[3], K=2)

print("Densities of o2, o5 and o10:")
print(density_o2, density_o3, density_o10)

ard_o2_solution = density_o2 / ((density_o3 + density_o10) / 2)

print("ARD: " + str(ard_o2_solution))

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