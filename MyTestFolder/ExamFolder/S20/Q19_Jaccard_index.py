import numpy as np
from itertools import combinations

# Provided data and class labels
data = np.array([
    [1, 1, 1, 0, 0],  # o1
    [1, 1, 1, 0, 0],  # o2
    [1, 1, 1, 0, 0],  # o3
    [1, 1, 1, 0, 0],  # o4
    [1, 1, 1, 0, 0],  # o5
    [0, 1, 1, 0, 0],  # o6
    [0, 1, 0, 1, 1],  # o7
    [1, 1, 1, 0, 0],  # o8
    [1, 0, 1, 0, 0],  # o9
    [0, 0, 0, 1, 1],  # o10
    [0, 1, 0, 1, 1],  # o11
])

class_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

# Cluster data based on f2
cluster_labels = data[:, 1]

# Calculate S and D
S = 0
D = 0
for (i, j) in combinations(range(len(class_labels)), 2):
    same_class = (class_labels[i] == class_labels[j])
    same_cluster = (cluster_labels[i] == cluster_labels[j])
    if same_class and same_cluster:
        S += 1
    elif not same_class and not same_cluster:
        D += 1

# Total number of pairs
N = len(class_labels)
total_pairs = (N * (N - 1)) // 2

# Calculate Jaccard Index
J = S / (total_pairs - D)

print("Number of pairs (total):", total_pairs)
print("Number of pairs (S):", S)
print("Number of pairs (D):", D)
print("Jaccard Index J:", J)
