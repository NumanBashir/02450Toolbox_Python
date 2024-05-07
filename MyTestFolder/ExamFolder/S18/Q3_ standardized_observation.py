import numpy as np

V = np.array([
    [0.38, -0.51, 0.23, 0.47, -0.55, 0.11],
    [0.41, 0.41, -0.53, 0.24, 0.00, 0.58],  
    [0.50, 0.34, -0.13, 0.15, -0.05, -0.77],  
    [0.29, 0.48, 0.78, -0.17, 0.00, 0.23],  
    [0.45, -0.42, 0.09, 0.03, 0.78, 0.04],  
    [0.39, -0.23, -0.20, -0.82, -0.30, 0.04],
])

# Standardized observation z*
z_star = np.array([-0.1, 0.2, 0.1, -0.3, 1, 0.5])

# First two principal components from V
V_reduced = V[:, :2]

# Project z* onto the space of the first two principal components
projection = np.dot(z_star, V_reduced)

print(projection)
