import numpy as np

# Given data
mu = np.array([12.9, 58.2, 1.7, 1436.8, 4.1])
x = np.array([15.5, 59.2, 1.4, 1438.0, 5.3])
V = np.array([
    [0.11, -0.8, 0.3, -0.17, -0.48],
    [-0.58, -0.31, 0.01, -0.5, 0.56],
    [0.49, 0.08, -0.49, -0.72, -0.07],
    [0.6, -0.36, 0.04, 0.27, 0.66],
    [-0.23, -0.36, -0.82, 0.37, -0.09]
])

# Center the observation
x_centered = x - mu

# Project onto the principal components
b = V.T @ x_centered

# Round the b vector to 2 decimal places
b_rounded = np.round(b, 2)

print(b_rounded)
