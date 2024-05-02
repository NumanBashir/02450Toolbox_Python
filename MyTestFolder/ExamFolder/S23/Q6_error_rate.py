import numpy as np

# Simulated error data from Tables 3 and 4 (Example data)
errors_outer_fold_1 = np.array([
    [0, 0, 1, 0, 0],  # Errors for K=1
    [1, 1, 1, 1, 1],  # Errors for K=3
    [1, 1, 1, 0, 0]   # Errors for K=4
])

errors_outer_fold_2 = np.array([
    [0, 0, 1, 0, 1],  # Errors for K=1
    [0, 0, 1, 0, 0],  # Errors for K=3
    [0, 0, 1, 0, 0]   # Errors for K=4
])

# Function to calculate the best K based on minimum average error
def get_best_k(errors):
    mean_errors = errors.mean(axis=1)
    best_k_index = np.argmin(mean_errors)
    return best_k_index + 1, mean_errors[best_k_index]  # +1 to adjust index to K value

# Get best K for each outer fold and corresponding error rate
best_k_fold_1, error_rate_fold_1 = get_best_k(errors_outer_fold_1)
best_k_fold_2, error_rate_fold_2 = get_best_k(errors_outer_fold_2)

print("Outer Fold 1: Best K =", best_k_fold_1, "with Error Rate =", error_rate_fold_1)
print("Outer Fold 2: Best K =", best_k_fold_2, "with Error Rate =", error_rate_fold_2)
