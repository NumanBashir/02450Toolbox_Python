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

# Distances for observation o11 from the distance matrix
distances_o11 = correct_distance_matrix[10]  # Index 10 corresponds to o11

# Identify the two smallest distances (excluding the distance to itself, which is 0)
sorted_indices = np.argsort(distances_o11)
closest_two_indices = sorted_indices[1:3]  # Skip the first one as it is the distance to itself

# Distances of the two closest observations to o11
closest_distances = distances_o11[closest_two_indices]

# Number of features M
M = 8

# Bandwidth lambda
lambda_ = 20

# Gaussian kernel density estimation
kde_estimate = (1 / (2 * np.sqrt((2 * np.pi * lambda_**2)**M))) * np.sum(np.exp(-closest_distances**2 / (2 * lambda_**2)))

print(kde_estimate, closest_distances)


# TODO CHECK THESE OPTIONS FROM THE ANSWERS 
lambda_ = 20
M = 8
two_pi_lambda_squared = (2 * np.pi * lambda_**2)**M

# Calculate the denominator for the density estimations
denominator = np.sqrt(two_pi_lambda_squared)

# Compute the estimates for each option
option_A = 0.5 / denominator * 0.6246
option_B = 0.5 / denominator * 1.922
option_C = 1 / denominator * 0.6246
option_D = 1 / denominator * 1.922

print("A: ", option_A)
print("B: ", option_B)
print("C: ", option_C)
print("D: ", option_D)

