import numpy as np

correct_distance_matrix = np.array([
    [0, 8.55, 0.43, 1.25, 1.14, 3.73, 2.72, 1.63, 1.68, 1.28],
    [8.55, 0, 8.23, 8.13, 8.49, 6.84, 8.23, 8.28, 8.13, 7.66],
    [0.43, 8.23, 0, 1.09, 1.10, 3.55, 2.68, 1.50, 1.52, 1.05],
    [1.25, 8.13, 1.09, 0, 1.23, 3.21, 2.17, 1.29, 1.33, 0.56],
    [1.14, 8.49, 1.10, 1.23, 0, 3.20, 2.68, 1.56, 1.50, 1.28],
    [3.73, 6.84, 3.55, 3.21, 3.20, 0, 2.98, 2.66, 2.50, 3.00],
    [2.72, 8.23, 2.68, 2.17, 2.68, 2.98, 0, 2.28, 2.30, 2.31],
    [1.63, 8.28, 1.50, 1.29, 1.56, 2.66, 2.28, 0, 0.25, 1.46],
    [1.68, 8.13, 1.52, 1.33, 1.50, 2.50, 2.30, 0.25, 0, 1.44],
    [1.28, 7.66, 1.05, 0.56, 1.28, 3.00, 2.31, 1.46, 1.44, 0]
])


# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
density_o2 = calculate_density(correct_distance_matrix[1], K=2)
density_o6 = calculate_density(correct_distance_matrix[5], K=2)
density_o10 = calculate_density(correct_distance_matrix[9], K=2)

print("Densities of o2, o5 and o10:")
print(density_o2, density_o6, density_o10)

ard_o2_solution = density_o2 / ((density_o6 + density_o10) / 2)

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