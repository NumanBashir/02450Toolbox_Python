import numpy as np
from scipy.stats import norm

# Given parameters
weights = np.array([0.19, 0.34, 0.48])
means = np.array([3.177, 3.181, 3.184])
std_devs = np.array([0.0062, 0.0076, 0.0075])
x0 = 3.19

# Calculate the normal PDF for each component at x0
pdf_values = norm.pdf(x0, means, std_devs)

# Calculate the numerator for k=2
numerator = weights[1] * pdf_values[1]

# Calculate the denominator (total mixture density at x0)
denominator = np.sum(weights * pdf_values)

# Posterior probability for cluster k=2
posterior_prob = numerator / denominator

print(f"The probability that the observation x0 = {x0} is assigned to cluster k=2 is {posterior_prob:.3f}")
