import numpy as np

# Given empirical covariance matrix
covariance_matrix = np.array([
    [12, -29, -21, -12, -317],
    [-29, 6104, 6026, 1557, 67964],
    [-21, 6026, 7263, 1701, 70892],
    [-12, 1557, 1701, 1012, 25415],
    [-317, 67964, 70892, 25415, 1212707]
])

# Extract the covariance between MONTH (x1) and PM2.5 (x2) --> Change this according to the question
cov_x1_x2 = covariance_matrix[0, 1]

# Variance of MONTH (x1) and PM2.5 (x2)
var_x1 = covariance_matrix[0, 0]
var_x2 = covariance_matrix[1, 1]

# Calculate the empirical correlation coefficient
correlation_x1_x2 = cov_x1_x2 / np.sqrt(var_x1 * var_x2)
print(correlation_x1_x2)
