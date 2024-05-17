import numpy as np

# Singular values from S matrix in the PCA decomposition
singular_values = np.array([19.64, 6.87, 3.26, 2.3, 1.12])

# TODO: If there is a statement about last 2 or similar, just cut the array to only have the ones asked about, ot just each of the explained variances

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
