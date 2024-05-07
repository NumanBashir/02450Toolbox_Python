# Define constants
K = 5  # Number of models at each stage
M = 8  # Total number of features

# Calculate the number of models for forward selection
# From having 1 feature up to M-1 features
N_forward = sum(K * (M - i) for i in range(1, M+1))

# Calculate the number of models for backward selection
# From having M features down to 2 features
N_backward = sum(K * (M - i) for i in range(3, M-1))

# Calculate the difference
difference = N_forward - N_backward

print(f"N_forward: {N_forward}")
print(f"N_backward: {N_backward}")
print(f"Difference (N_forward - N_backward): {difference}")
