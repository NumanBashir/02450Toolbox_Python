import numpy as np
from scipy.stats import norm

# Given parameters
weights = np.array([0.13, 0.55, 0.32])
means = np.array([18.347, 14.997, 18.421])
std_devs = np.array([1.2193, 0.986, 1.1354])
x0 = 15.38

# Calculate the normal PDF for each component at x0
pdf_values = norm.pdf(x0, means, std_devs)

# Calculate the numerator for k=2
numerator = weights[1] * pdf_values[1]

# Calculate the denominator (total mixture density at x0)
denominator = np.sum(weights * pdf_values)

# Posterior probability for cluster k=2
posterior_prob = numerator / denominator

print(f"The probability that the observation x0 = {x0} is assigned to cluster k=2 is {posterior_prob:.3f}")
