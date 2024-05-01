import numpy as np

# Initial data setup
y_true = np.array([1, 1, 1, 2, 2, 2, 2])  # True class labels
t_1 = np.array([1, 2, 1, 1, 1, 2, 2])     # Predicted outputs for the first round of AdaBoost

# Number of observations
N = len(y_true)

# Initial weights
w = np.ones(N) / N

# Calculate the error of the classifier at t=1
epsilon_1 = np.sum(w[y_true != t_1])

# Calculate alpha_1 (the weight of the classifier at t=1)
alpha_1 = 0.5 * np.log((1 - epsilon_1) / epsilon_1)

# Update weights
w[y_true != t_1] *= np.exp(alpha_1)   # Increase weights for misclassified
w[y_true == t_1] *= np.exp(-alpha_1)  # Decrease weights for correctly classified

# Normalize the weights
w /= np.sum(w)

print("Updated weights:", w)
print("Error epsilon_1:", epsilon_1)
print("Alpha_1:", alpha_1)