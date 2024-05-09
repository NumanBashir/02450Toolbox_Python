import numpy as np


V = np.array([
    [0.45, -0.60, -0.64, 0.15],
    [-0.40, -0.80, 0.43, -0.16],
    [0.58, -0.01, 0.24, -0.78],
    [0.55, -0.08, 0.59, 0.58]
])

# Calculate Euclidean norms of the first and second column vectors
norm_first = np.linalg.norm(V[:, 0])  # First column vector
norm_second = np.linalg.norm(V[:, 1])  # Second column vector

# Output the norms to compare them
print("Norm of the first column vector:", norm_first)
print("Norm of the second column vector:", norm_second)

# Interpret the loadings visually
print("\nMatrix V (Loading vectors):")
print(V)
