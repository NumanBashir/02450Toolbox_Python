# Initialize the weights
import numpy as np


N = 6  # number of observations
initial_weights = np.full(N, 1/N)

# Given that three positive points are misclassified, the indices of misclassified points are:
misclassified_indices = [1, 2]  # zero-indexed

# Calculate the weighted error (epsilon)
epsilon = np.sum(initial_weights[misclassified_indices])

# Calculate the classifier importance (alpha)
alpha = 0.5 * np.log((1 - epsilon) / epsilon)

# Update the weights
# For correctly classified, weights are decreased
# For misclassified, weights are increased
new_weights = initial_weights.copy()
for i in range(N):
    if i in misclassified_indices:
        new_weights[i] = initial_weights[i] * np.exp(alpha)
    else:
        new_weights[i] = initial_weights[i] * np.exp(-alpha)

# Normalize the new weights so they sum to 1
new_weights /= np.sum(new_weights)

epsilon, alpha, new_weights

print(new_weights)
