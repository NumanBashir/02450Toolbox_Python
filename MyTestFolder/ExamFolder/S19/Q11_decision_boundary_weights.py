import numpy as np

# Define the point b
b = np.array([0, -1])

# Define the weights for each option
weights = {
    "A": ([-0.77, -5.54, 0.01], [0.26, -2.09, -0.03]),
    "B": ([0.51, 1.65, 0.01], [0.1, 3.8, 0.04]),
    "C": ([-0.9, -4.39, -0.0], [-0.09, -2.45, -0.04]),
    "D": ([-1.22, -9.88, -0.01], [-0.28, -2.9, -0.01])
}

# Calculate the values of y_hat for each option
results = {}
for key, (w1, w2) in weights.items():
    w1 = np.array(w1)
    w2 = np.array(w2)
    
    y_hat_1 = w1[0] + w1[1] * b[0] + w1[2] * b[1]
    y_hat_2 = w2[0] + w2[1] * b[0] + w2[2] * b[1]
    y_hat_3 = 0  # Since we only need two decision boundaries, the third class is determined by the absence of the first two
    
    results[key] = np.array([y_hat_1, y_hat_2, y_hat_3])

# Print the results
for key, y_hats in results.items():
    max_class = np.argmax(y_hats) + 1  # Classes are 1-indexed
    print(f"Option {key}: y_hat_1 = {y_hats[0]:.2f}, y_hat_2 = {y_hats[1]:.2f}, y_hat_3 = {y_hats[2]:.2f} => Class {max_class}")

# Check which option gives the correct class
correct_class = 2  # As per the solution
for key, y_hats in results.items():
    max_class = np.argmax(y_hats) + 1
    if max_class == correct_class:
        print(f"Correct option is {key}")
