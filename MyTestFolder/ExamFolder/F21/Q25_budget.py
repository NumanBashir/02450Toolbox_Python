import numpy as np

# Constants
N = 1000
D_test = int(0.2 * N)  # 20% for testing
D_train = N - D_test  # Remaining for training
total_budget = 200000  # Total time budget

def compute_time(K, L=3, N_train=800, N_test=200):
    # Time for training and testing on the main training and testing sets
    t_o = N_train * np.log2(N_train) + N_test

    # Initialize the total time for cross-validation
    t_i = 0
    # Compute time for each fold
    for _ in range(K):
        n_i = N_train / K  # Number of testing observations per fold
        n_ti = N_train - n_i  # Number of training observations per fold
        # Time for training and testing in one fold, multiplied by L for regularization strengths
        t_i += L * (n_ti * np.log2(n_ti) + n_i)

    # Sum the time for the outer loop and the inner cross-validation loop
    total_time = t_o + t_i
    return total_time

# Find the maximum K within the budget
K = 1
while True:
    if compute_time(K + 1) > total_budget:
        break
    K += 1

print("Maximum K within the budget:", K)
