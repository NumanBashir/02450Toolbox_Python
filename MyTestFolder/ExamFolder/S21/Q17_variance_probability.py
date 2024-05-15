import numpy as np

# Probabilities
p = np.array([0.5, 0.36, 0.14])

# Values of k
k = np.array([1, 2, 3])

# Mean
mu = 1.6

# Calculate the variance
variance = np.sum(p * (k - mu)**2)
print(f"Variance: {variance}")
