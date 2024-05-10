import numpy as np

def sigmoid(z):
    """Calculate the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def logistic_regression_prediction(w, x2):
    """Compute the logistic regression probability p(x5 = 1 | x2) using weights w and input x2."""
    # Construct the augmented feature vector
    x_tilde = np.array([1, x2])
    # Compute the linear combination using matrix multiplication
    linear_combination = np.dot(w, x_tilde)
    # Apply the sigmoid function to get the probability
    return sigmoid(linear_combination)

# Define weights for each option
weights = {
    'A': np.array([423.49, 48.16]),
    'B': np.array([0.0, -46.21]),
    'C': np.array([0.0, -27.89]),
    'D': np.array([418.94, -26.12])
}

# Select x2 = 16 for the prediction
x2_selected = 16

# Compute and print probabilities for each option
for key, w in weights.items():
    probability = logistic_regression_prediction(w, x2_selected)
    print(f"Option {key}: p(x5 = 1 | x2={x2_selected}) = {probability:.4f}")
