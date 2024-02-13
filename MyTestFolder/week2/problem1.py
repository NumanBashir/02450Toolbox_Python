import numpy as np

# Given singular values
singular_values = np.array([17.0, 15.2, 13.1, 13.0, 11.8, 11.3])

# Calculate the variance explained by each component
variance_explained = (singular_values ** 2) / np.sum(singular_values ** 2)

# Calculate cumulative variance explained
cumulative_variance_explained = np.cumsum(variance_explained)

# Print the variance explained by the first three components
print(f"Variance explained by the first three components: {cumulative_variance_explained[2] * 100:.2f}%")

# Determine if Statement A is correct
if cumulative_variance_explained[2] > 0.9:
    print("Statement A is correct: The first three principal components account for more than 90% of the variation.")
else:
    print("Statement A is incorrect.")

# Interpretation for Statement B and C is based on the signs and magnitude of v1 and v2 components.
