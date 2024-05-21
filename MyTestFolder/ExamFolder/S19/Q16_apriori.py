import numpy as np
from itertools import combinations

# Provided feature table (first 4 features only)
table4 = np.array([
    [0, 0, 0, 1],  # o1
    [0, 0, 0, 0],  # o2
    [0, 1, 1, 1],  # o3
    [1, 0, 0, 0],  # o4
    [1, 0, 0, 1],  # o5
    [0, 0, 1, 1],  # o6
    [0, 0, 1, 1],  # o7
    [0, 0, 0, 0],  # o8
    [0, 1, 1, 0],  # o9
    [0, 0, 1, 1]   # o10
])

# Minimum support threshold
min_support = 0.35 * len(table4)

# L1 matrix
L1 = np.array([
    [0, 0, 1, 0],  # Itemset {f3}
    [0, 0, 0, 1]   # Itemset {f4}
])

# Generate candidate itemsets for C2
def generate_candidates(L1):
    itemsets = [set(np.where(row)[0]) for row in L1]
    candidates = set()
    for comb in combinations(range(L1.shape[1]), 2):
        if set(comb[0:1]) in itemsets and set(comb[1:2]) in itemsets:
            candidates.add(tuple(sorted(comb)))
    return candidates

# Generate C2 from L1
C2_candidates = generate_candidates(L1)

# Filter candidates by support
def calculate_support(itemset, data):
    return np.sum(np.all(data[:, list(itemset)] == 1, axis=1))

C2 = []
for candidate in C2_candidates:
    support = calculate_support(candidate, table4)
    if support >= min_support:
        C2.append(candidate)

# Convert C2 to numpy array for comparison with options
C2_matrix = np.zeros((len(C2), table4.shape[1]), dtype=int)
for i, itemset in enumerate(C2):
    C2_matrix[i, list(itemset)] = 1

print("C2 matrix:")
print(C2_matrix)
