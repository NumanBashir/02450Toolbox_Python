import numpy as np

# Input data and corresponding targets
X = np.array([[0, 0], [1, 0], [1, 1], [1, 2]])
y = np.array([1, 3, 5, 7])

# Fixed weights for the first layer and activation function is ReLU
w1 = np.array([-2, 4, 2])  # Last element is the bias
X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))  # Augment X with a column of ones for the bias term

# Compute the activations from the first layer
z1_input = X_augmented.dot(w1)
z1 = np.maximum(0, z1_input)  # Apply ReLU activation function

# Define candidate weights for the second layer, assuming the question intends only the scalar to vary
w2_options = {
    'A': np.array([1]),
    'B': np.array([2]),
    'C': np.array([3]),
    'D': np.array([4])
}

# Calculate MSE for each option
mse_results = {}
for key, w2 in w2_options.items():
    f_x = z1 * w2  # Output of the network
    mse = np.mean((f_x - y) ** 2)
    mse_results[key] = mse

# REMEMBER TO CHECK IF WE ARE LOOKING AT MINIMIZE OR MAXIMIZE
print(mse_results)
