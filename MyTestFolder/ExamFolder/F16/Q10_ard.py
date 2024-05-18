import numpy as np

correct_distance_matrix = np.array([
    [0, 0.534, 1.257, 1.671, 1.090, 1.315, 1.484, 1.253, 1.418],
    [0.534, 0, 0.727, 2.119, 1.526, 1.689, 1.214, 0.997, 1.056],
    [1.257, 0.727, 0, 2.809, 2.220, 2.342, 1.088, 0.965, 0.807],
    [1.671, 2.119, 2.809, 0, 0.601, 0.540, 3.135, 2.908, 3.087],
    [1.090, 1.526, 2.220, 0.601, 0, 0.331, 2.563, 2.338, 2.500],
    [1.315, 1.689, 2.342, 0.540, 0.331, 0, 2.797, 2.567, 2.708],
    [1.484, 1.214, 1.088, 3.135, 2.563, 2.797, 0, 0.275, 0.298],
    [1.253, 0.997, 0.965, 2.908, 2.338, 2.567, 0.275, 0, 0.343],
    [1.418, 1.056, 0.807, 3.087, 2.500, 2.708, 0.298, 0.343, 0]
])


# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
density_o2 = calculate_density(correct_distance_matrix[3], K=1)
density_o3 = calculate_density(correct_distance_matrix[5], K=1)

print("Densities of o2, o5 and o10:")
print(density_o2, density_o3)

ard_o2_solution = density_o2 / ((density_o3) / 1)

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