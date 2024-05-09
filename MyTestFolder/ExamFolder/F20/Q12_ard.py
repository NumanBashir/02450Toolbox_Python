import numpy as np

correct_distance_matrix = np.array([
    [0, 725, 800, 150, 1000, 525, 600, 500, 400, 850],
    [725, 0, 75, 575, 275, 1250, 1325, 226, 325, 125],
    [800, 75, 0, 650, 200, 1325, 1400, 300, 400, 51],
    [150, 575, 650, 0, 850, 675, 750, 350, 250, 700],
    [1000, 275, 200, 850, 0, 1525, 1600, 500, 600, 150],
    [525, 1250, 1325, 675, 1525, 0, 75, 1025, 925, 1375],
    [600, 1325, 1400, 750, 1600, 75, 0, 1100, 1000, 1450],
    [500, 226, 300, 350, 500, 1025, 1100, 0, 100, 350],
    [400, 325, 400, 250, 600, 925, 1000, 100, 0, 450],
    [850, 125, 51, 700, 150, 1375, 1450, 350, 450, 0]
])


# Function to calculate the density based on the provided formula
def calculate_density(distances, K):
    return 1 / np.mean(np.sort(distances)[1:K+1])  # Exclude the zero distance (self)

# Calculate the densities for o5 and its two nearest neighbors (o4 and o6)
# o5 index is 4, o4 index is 3, and o6 index is 5 in zero-indexed Python
density_o2 = calculate_density(correct_distance_matrix[1], K=2)
density_o3 = calculate_density(correct_distance_matrix[2], K=2)
density_o10 = calculate_density(correct_distance_matrix[9], K=2)

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