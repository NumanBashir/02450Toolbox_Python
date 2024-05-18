import numpy as np

# Number of observations
N = 25

# Number of misclassified observations
misclassified = 5

# Initial weights (uniform distribution)
w = np.ones(N) / N

# Calculate error rate
epsilon_t = misclassified / N

# Calculate alpha_t
alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)

# Update weights
# For misclassified observations
updated_weight_misclassified = w[0] * np.exp(alpha_t)

# Normalize weights
sum_weights = (N - misclassified) * (w[0] * np.exp(-alpha_t)) + misclassified * updated_weight_misclassified
updated_weight_misclassified /= sum_weights

# Print the updated weight for misclassified observations
print(f"Updated weight for misclassified observations: {updated_weight_misclassified:.3f}")

# Expected output: 0.1
