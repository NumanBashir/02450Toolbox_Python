import numpy as np

# Given error rate
error_rate = 1 / 16

# Calculate alpha_t
alpha_t = 0.5 * np.log((1 - error_rate) / error_rate)

# Initial weight
initial_weight = 1 / 32

# Updated weights for correctly classified and misclassified observations
w_c_prime = initial_weight * np.exp(-alpha_t)
w_m_prime = initial_weight * np.exp(alpha_t)

# Normalization factor
Z = 30 * w_c_prime + 2 * w_m_prime

# Normalized weight for correctly classified observations
w_c = w_c_prime / Z

print(f"Updated weight for correctly classified observations: {w_c:.4f}")
