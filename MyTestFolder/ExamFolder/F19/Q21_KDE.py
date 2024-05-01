import numpy as np
from scipy.stats import norm

# Data points
x_train = np.array([4.5, -0.5])
x_test = np.array([1.2])

# Kernel width
sigma = 2

# KDE function to calculate the density at a point
def kde(x, data, bandwidth):
    n = len(data)
    density = sum(norm(loc=xi, scale=bandwidth).pdf(x) for xi in data) / n
    return density

# Calculate the density at the test point using the training data
density_at_test = kde(x_test, x_train, sigma)

# Calculate the negative log-likelihood
negative_log_likelihood = -np.log(density_at_test)

print("Density at test point:", density_at_test)
print("Negative log-likelihood:", negative_log_likelihood)
