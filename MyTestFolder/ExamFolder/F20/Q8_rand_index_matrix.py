import numpy as np
from scipy.special import comb

# Confusion matrix values (rows: actual classes, columns: predicted clusters)
conf_matrix = np.array([
    [114, 0, 32],   # Adelie
    [0, 119, 0],    # Gentoo
    [8, 0, 60]      # Chinstrap
])

# Total number of observations
N = np.sum(conf_matrix)

# Calculate S (sum of combinations of counts within the same class and cluster)
S = np.sum([comb(n, 2) for n in conf_matrix.flatten() if n > 1])

# Calculate total possible pairs
total_pairs = comb(N, 2)

# Sum of all intra-cluster (predicted cluster) combinations
sum_nZk = np.sum([comb(np.sum(conf_matrix[:, i]), 2) for i in range(conf_matrix.shape[1]) if np.sum(conf_matrix[:, i]) > 1])

# Sum of all inter-cluster (actual class) combinations
sum_nQm = np.sum([comb(np.sum(conf_matrix[i, :]), 2) for i in range(conf_matrix.shape[0]) if np.sum(conf_matrix[i, :]) > 1])

# Calculate D
D = total_pairs - sum_nZk - sum_nQm + S

# Calculate Rand Index
rand_index = (S + D) / total_pairs

print(f"Rand Index: {rand_index:.2f}")
