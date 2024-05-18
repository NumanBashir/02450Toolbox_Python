import numpy as np
from itertools import combinations

# Cluster indices after thresholding the dendrogram
cluster_indices = np.array([1, 1, 3, 2, 2, 2, 3, 3, 3])

# True class labels
true_labels = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])

# Number of observations
N = len(cluster_indices)

# Total number of object pairs
K = N * (N - 1) // 2

# Initialize f00 and f11
f00 = 0
f11 = 0

# Iterate over all possible pairs
for (i, j) in combinations(range(N), 2):
    if (cluster_indices[i] == cluster_indices[j]) and (true_labels[i] == true_labels[j]):
        f11 += 1
    elif (cluster_indices[i] != cluster_indices[j]) and (true_labels[i] != true_labels[j]):
        f00 += 1

# Calculate SMC
SMC = (f00 + f11) / K

# Print the result
print(f"f00: {f00}, f11: {f11}, SMC: {SMC:.4f}")

# Expected output: f00: 24, f11: 7, SMC: 0.8611
