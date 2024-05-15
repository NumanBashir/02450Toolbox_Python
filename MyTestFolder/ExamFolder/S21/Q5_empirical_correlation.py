import numpy as np

# Given empirical covariance matrix
covariance_matrix = np.array([
    [143.0, 39.0, -0.0, 253.0, 142.0],
    [39.0, 415.0, -7.0, -6727.0, 143.0],
    [-0.0, -7.0, 1.0, 108.0, -2.0],
    [253.0, -6727.0, 108.0, 370027.0, -1403.0],
    [142.0, 143.0, -2.0, -1403.0, 171.0]
])

# Extract the covariance between MONTH (x1) and PM2.5 (x2) --> Change this according to the question
cov_x2_x3 = covariance_matrix[1, 2]

# Variance of MONTH (x1) and PM2.5 (x2)
var_x2 = covariance_matrix[1, 1]
var_x3 = covariance_matrix[2, 2]

# Calculate the empirical correlation coefficient
correlation_x2_x3 = cov_x2_x3 / np.sqrt(var_x2 * var_x3)
print(correlation_x2_x3)
