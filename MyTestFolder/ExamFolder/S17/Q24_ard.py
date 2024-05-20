import numpy as np

correct_distance_matrix = np.array([
    [0, 0.2606, 1.1873, 2.4946, 2.9510, 2.5682, 3.4535, 2.4698],
    [0.2606, 0, 1.2796, 2.4442, 2.8878, 2.4932, 3.3895, 2.4216],
    [1.1873, 1.2796, 0, 2.8294, 3.6892, 2.9147, 4.1733, 2.2386],
    [2.4946, 2.4442, 2.8294, 0, 1.4852, 0.2608, 2.2941, 1.8926],
    [2.9510, 2.8878, 3.6892, 1.4852, 0, 1.5155, 1.0296, 3.1040],
    [2.5682, 2.4932, 2.9147, 0.2608, 1.5155, 0, 2.3316, 1.8870],
    [3.4535, 3.3895, 4.1733, 2.2941, 1.0296, 2.3316, 0, 3.7588],
    [2.4698, 2.4216, 2.2386, 1.8926, 3.1040, 1.8870, 3.7588, 0]
])


# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
density_o2 = calculate_density(correct_distance_matrix[7], K=2) # THE ONE WE ARE LOOKING AT

# THE TWO NEAREST NEIGHBORS
density_o3 = calculate_density(correct_distance_matrix[3], K=2) 
density_o10 = calculate_density(correct_distance_matrix[5], K=2)

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