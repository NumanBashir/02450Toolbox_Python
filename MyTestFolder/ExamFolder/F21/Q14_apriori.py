from itertools import combinations
import numpy as np

data = np.array([
    [0, 0, 0, 1, 0, 0, 0, 1],  # o1 -> C1
    [0, 0, 1, 0, 0, 1, 0, 1],  # o2 -> C1
    [0, 0, 1, 0, 0, 1, 0, 1],  # o3 -> C2
    [0, 1, 0, 0, 0, 1, 0, 1],  # o4 -> C2
    [0, 0, 0, 0, 0, 1, 0, 1],  # o5 -> C2
    [0, 0, 1, 0, 1, 1, 0, 1],  # o6 -> C2
    [0, 0, 1, 0, 0, 1, 0, 1],  # o7 -> C2
    [1, 1, 0, 0, 0, 0, 1, 1],  # o8 -> C3
    [0, 1, 0, 0, 0, 0, 0, 1],  # o9 -> C3
    [0, 1, 0, 0, 0, 1, 0, 1],  # o10 -> C3
    [1, 1, 0, 0, 0, 0, 0, 0],  # o11 -> C3
])

# Generate all combinations of three features
feature_combinations = list(combinations(range(8), 3))  # 8 features from f1 to f8

# Calculate support for each three-item set
supports = {}
for combination in feature_combinations:
    # Count how many rows have 1s in all columns of the current combination
    support_count = np.sum(np.all(data[:, combination] == 1, axis=1))
    supports[combination] = support_count / len(data)

# Filter itemsets with support greater than or equal to 0.3
L3 = {combination: support for combination, support in supports.items() if support >= 0.3}
print(L3) # REMEMBER THESE ARE 0-INDEXED MEANING YOU HAVE TO ADD ONE
