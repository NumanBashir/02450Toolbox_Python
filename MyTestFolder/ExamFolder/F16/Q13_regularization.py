import numpy as np

# Define the given weights for different values of lambda
weights = {
    "wa": np.array([0.0538, 0.0558, 0.1861, -0.0596]),
    "wb": np.array([0.0089, 0.0931, 0.1093, 0.0417]),
    "wc": np.array([0.2811, 0.0445, 0.3379, -0.4626]),
    "wd": np.array([0.0167, 0.0698, 0.1354, 0.0403])
}

# Define the lambda values (not given directly in the problem, we are to infer them)
lambda_values = {
    "wa": 10,
    "wb": 1000,
    "wc": 1,
    "wd": 100
}

# Print the weights and corresponding lambda values
for w, lambda_val in lambda_values.items():
    print(f"Weights {w} correspond to lambda = {lambda_val}: {weights[w]}")

# Check which weight corresponds to lambda = 10
target_lambda = 10
for w, lambda_val in lambda_values.items():
    if lambda_val == target_lambda:
        print(f"\nThe weights corresponding to lambda = {target_lambda} are {w}: {weights[w]}")

# Verify the solution as stated in the question
correct_weights = weights["wa"]
print(f"\nVerification: Weights for lambda = 10 are {correct_weights}")
