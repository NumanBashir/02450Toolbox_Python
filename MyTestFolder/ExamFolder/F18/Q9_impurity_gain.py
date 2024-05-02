import numpy as np

# R is the matrix of the number of observations in each split belonging to each class
R = np.array([
    [108, 112, 56],
    [58, 75, 116]
])

# Total number of observations in each split
N_v = R.sum(axis=1)

# Total number of observations
N = R.sum()

# Function to calculate classification error impurity
def classification_error_impurity(R_k):
    total = R_k.sum()
    probabilities = R_k / total
    return 1 - np.max(probabilities)

# Impurity of each split
I_v1 = classification_error_impurity(R[0])
I_v2 = classification_error_impurity(R[1])

# Impurity of the overall dataset
probabilities = R.sum(axis=0) / N
I_0 = 1 - np.max(probabilities)

# Calculate the purity gain
Delta = I_0 - (N_v[0] / N * I_v1 + N_v[1] / N * I_v2)
print(f"Purity Gain (Δ): {Delta}")
