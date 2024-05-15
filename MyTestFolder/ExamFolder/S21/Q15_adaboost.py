import numpy as np

# Given error rate epsilon
epsilon = 0.3  # Example value; you can change this to the actual epsilon

# Calculate alpha
alpha = 0.5 * np.log((1 - epsilon) / epsilon)

# Calculate the ratio of weights
weight_ratio = np.exp(2 * alpha)

# Print the results
print(f"Alpha (α): {alpha}")
print(f"Ratio of weights w1/w5: {weight_ratio}")

# Verify the correct answer
expected_ratio = (1 - epsilon) / epsilon
print(f"Expected ratio (exp(2α)): {expected_ratio}")

# Checking if calculated ratio matches expected ratio
if np.isclose(weight_ratio, expected_ratio):
    print("The calculated ratio matches the expected ratio.")
else:
    print("There is a discrepancy in the ratio calculation.")
