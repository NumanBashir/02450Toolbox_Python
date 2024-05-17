import numpy as np

correct_distance_matrix = np.array([
    [0.0, 1.7, 1.4, 0.4, 2.2, 3.7, 5.2, 0.2, 4.3, 6.8, 6.0],
    [1.7, 0.0, 1.0, 2.0, 1.3, 2.6, 4.5, 1.8, 3.2, 5.9, 5.2],
    [1.4, 1.0, 0.0, 1.7, 0.9, 2.4, 4.1, 1.5, 3.0, 5.5, 4.8],
    [0.4, 2.0, 1.7, 0.0, 2.6, 4.0, 5.5, 0.3, 4.6, 7.1, 6.3],
    [2.2, 1.3, 0.9, 2.6, 0.0, 1.7, 3.4, 2.4, 2.1, 4.8, 4.1],
    [3.7, 2.6, 2.4, 4.0, 1.7, 0.0, 2.0, 3.8, 1.6, 3.3, 2.7],
    [5.2, 4.5, 4.1, 5.5, 3.4, 2.0, 0.0, 5.4, 2.5, 1.6, 0.9],
    [0.2, 1.8, 1.5, 0.3, 2.4, 3.8, 5.4, 0.0, 4.4, 6.9, 6.1],
    [4.3, 3.2, 3.0, 4.6, 2.1, 1.6, 2.5, 4.4, 0.0, 3.4, 2.9],
    [6.8, 5.9, 5.5, 7.1, 4.8, 3.3, 1.6, 6.9, 3.4, 0.0, 1.0],
    [6.0, 5.2, 4.8, 6.3, 4.1, 2.7, 0.9, 6.1, 2.9, 1.0, 0.0]
])


# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
density_o2 = calculate_density(correct_distance_matrix[2], K=2)
density_o3 = calculate_density(correct_distance_matrix[4], K=2)
density_o10 = calculate_density(correct_distance_matrix[1], K=2)

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