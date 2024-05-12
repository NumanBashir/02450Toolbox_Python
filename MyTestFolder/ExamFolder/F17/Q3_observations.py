import numpy as np

# Define the observation vector and projection matrix
vector = np.array([0.68, 0.66, -0.67, -1.47])
projection_matrix = np.array([
    [-0.60, 0.02],
    [-0.61, 0],
    [-0.46, 0.46],
    [0.25, 0.89]
])

# Calculate the projection
projected_vector = projection_matrix.T.dot(vector)

# Output the result
print("Projected vector:", projected_vector)
