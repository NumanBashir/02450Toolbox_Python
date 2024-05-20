import itertools

# Observations and their true labels
observations = ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8']
true_labels = [0, 0, 0, 0, 1, 1, 1, 0]
clusters = [1, 2, 2, 2, 1, 2, 1, 2]  # Cluster assignments (Example: O1 is in cluster 1, O2 in cluster 2, etc.)

# Calculate f11 and f00
f11 = 0
f00 = 0
pairs = list(itertools.combinations(range(len(observations)), 2))
for (i, j) in pairs:
    if (true_labels[i] == true_labels[j]) and (clusters[i] == clusters[j]):
        f11 += 1
    elif (true_labels[i] != true_labels[j]) and (clusters[i] != clusters[j]):
        f00 += 1

# Total number of pairs
K = len(pairs)

# Jaccard coefficient
J = f11 / (K - f00)
print(f"f11: {f11}")
print(f"f00: {f00}")
print(f"K: {K}")
print(f"Jaccard Coefficient: {J:.4f}")
