import numpy as np

correct_distance_matrix = np.array([
    [0.0, 2.91, 0.63, 1.88, 1.02, 1.82, 1.92, 1.58, 1.08, 1.43],
    [2.91, 0.0, 3.23, 3.9, 2.88, 3.27, 3.48, 4.02, 3.08, 3.47],
    [0.63, 3.23, 0.0, 2.03, 1.06, 2.15, 2.11, 1.15, 1.09, 1.65],
    [1.88, 3.9, 2.03, 0.0, 2.52, 1.04, 2.25, 2.42, 2.18, 2.17],
    [1.02, 2.88, 1.06, 2.52, 0.0, 2.44, 2.38, 1.53, 1.71, 1.94],
    [1.82, 3.27, 2.15, 1.04, 2.44, 0.0, 1.93, 2.72, 1.98, 1.8],
    [1.92, 3.48, 2.11, 2.25, 2.38, 1.93, 0.0, 2.53, 2.09, 1.66],
    [1.58, 4.02, 1.15, 2.42, 1.53, 2.72, 2.53, 0.0, 1.68, 2.06],
    [1.08, 3.08, 1.09, 2.18, 1.71, 1.98, 2.09, 1.68, 0.0, 1.48],
    [1.43, 3.47, 1.65, 2.17, 1.94, 1.8, 1.66, 2.06, 1.48, 0.0]
])


# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
density_o4 = calculate_density(correct_distance_matrix[3], K=2)
density_o1 = calculate_density(correct_distance_matrix[0], K=2)
density_o6 = calculate_density(correct_distance_matrix[5], K=2)

print("Densities of o4, o1 and o6:")
print(density_o4, density_o4, density_o6)

ard_o5_solution = density_o4 / ((density_o1 + density_o6) / 2)

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