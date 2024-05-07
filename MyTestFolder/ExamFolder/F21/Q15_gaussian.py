import numpy as np


data = np.array([
    [38.0, 15.1, 27.4, 77.9, 18.1, 33.3, 48.5, 50.0],
    [26.8, 12.8, 52.0, 77.0, 22.5, 68.1, 66.0, 75.0],  
    [64.5, 39.6, 74.4, 37.1, 45.7, 66.7, 66.0, 64.3],  
    [63.2, 45.7, 29.1, 41.4, 49.1, 56.9, 59.2, 50.0],  
    [66.3, 34.3, 37.7, 43.1, 40.9, 63.9, 70.9, 60.7],  
    [56.7, 34.7, 72.2, 47.3, 38.4, 61.1, 62.1, 55.4],
    [63.4, 30.6, 66.4, 49.8, 30.2, 62.5, 50.5, 42.9],
    [87.1, 85.3, 19.3, 19.2, 68.6, 34.7, 64.1, 33.9],
    [51.3, 46.8, 14.8, 53.4, 49.3, 37.5, 52.4, 35.7],
    [67.5, 62.3, 13.0, 33.2, 66.7, 51.4, 41.7, 39.3],
    [86.0, 71.3, 25.1, 20.5, 71.9, 25.0, 48.5, 32.1],
])

import numpy as np

# Means for x1 and x2 in class C1
mu_x1_c1 = (38.0 + 26.8) / 2
mu_x2_c1 = (15.1 + 12.8) / 2

# Class probability for C1
p_C1 = 2 / 11

# Variance (sigma squared) for Gaussian distributions
sigma_squared = 400

# Gaussian density function
def gaussian_density(x, mu, sigma_squared):
    return (1 / np.sqrt(2 * np.pi * sigma_squared)) * np.exp(-((x - mu)**2) / (2 * sigma_squared))

# Calculate densities for x1 = 32.0 and x2 = 14.0 for class C1
p_x1_given_c1 = gaussian_density(32.0, mu_x1_c1, sigma_squared)
p_x2_given_c1 = gaussian_density(14.0, mu_x2_c1, sigma_squared)

# Given denominator value for the Naive Bayes classification
p_NB_denominator = 0.00010141

# Calculate numerator of the Bayes rule
numerator = p_x1_given_c1 * p_x2_given_c1 * p_C1

# Calculate posterior probability for C1
posterior_c1 = numerator / p_NB_denominator

print(posterior_c1)
