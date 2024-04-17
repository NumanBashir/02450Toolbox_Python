import numpy as np

# Singular values from S matrix in the PCA decomposition
singular_values = np.array([43.67, 33.47, 31.15, 30.36, 27.77, 13.86])

# Total variance is the sum of squares of the singular values
total_variance = np.sum(singular_values**2)

# Calculate variance explained by each component
variance_explained = singular_values**2 / total_variance

# Single variance explained
print(variance_explained)

# Calculate cumulative variance explained
cumulative_variance_explained = np.cumsum(variance_explained)

# Cumulative variance explained
print(cumulative_variance_explained)
