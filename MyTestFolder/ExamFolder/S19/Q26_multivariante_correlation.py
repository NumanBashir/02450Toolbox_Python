import numpy as np

# Define the covariance matrices
Sigma1 = np.array([
    [0.5, 0.56, 0.0],
    [0.56, 1.5, -1.12],
    [0.0, -1.12, 2.0]
])

Sigma2 = np.array([
    [2.0, -1.12, 0.0],
    [-1.12, 1.5, 0.56],
    [0.0, 0.56, 0.5]
])

# Function to calculate correlation from covariance matrix
def calculate_correlation(Sigma):
    correlation = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
    return correlation

# Calculate the correlations
correlation1 = calculate_correlation(Sigma1)
correlation2 = calculate_correlation(Sigma2)

print(f"Correlation between x1 and x2 for Sigma1: {correlation1:.3f}")
print(f"Correlation between x1 and x2 for Sigma2: {correlation2:.3f}")
