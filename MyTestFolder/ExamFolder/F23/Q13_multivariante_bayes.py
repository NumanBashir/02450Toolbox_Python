import numpy as np
from scipy.stats import norm

# Given data
mu_1 = np.array([0.77, -0.41])
sigma_1 = np.array([0.29, 0.55])
mu_2 = np.array([-0.91, 0.5])
sigma_2 = np.array([0.32, 1.12])

prior_1 = 0.53
prior_2 = 0.47

x_test = np.array([0, 0.7])

# Calculate the likelihoods using the diagonal elements of the covariance matrices
likelihood_1 = norm.pdf(x_test[0], mu_1[0], np.sqrt(sigma_1[0])) * norm.pdf(x_test[1], mu_1[1], np.sqrt(sigma_1[1]))
likelihood_2 = norm.pdf(x_test[0], mu_2[0], np.sqrt(sigma_2[0])) * norm.pdf(x_test[1], mu_2[1], np.sqrt(sigma_2[1]))

# Calculate the posteriors
posterior_1 = likelihood_1 * prior_1
posterior_2 = likelihood_2 * prior_2

# Calculate the ratio of the posteriors
posterior_ratio = posterior_1 / posterior_2

# Print results
print(f"Likelihood for class 1: {likelihood_1}")
print(f"Likelihood for class 2: {likelihood_2}")
print(f"Posterior for class 1: {posterior_1}")
print(f"Posterior for class 2: {posterior_2}")
print(f"Ratio of posteriors (p(y=1 | x_test) / p(y=2 | x_test)): {posterior_ratio}")

# Determine the class
if posterior_ratio > 1:
    print("Classify as class 1 (Low)")
else:
    print("Classify as class 2 (High)")
