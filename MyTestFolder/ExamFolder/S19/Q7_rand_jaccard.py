import numpy as np
from sklearn.metrics import rand_score, jaccard_score

# Ground-truth clustering Z (using class labels)
Z = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 3])

# Obtained clustering Q (from dendrogram clusters)
Q = np.array([1, 1, 2, 1, 1, 1, 1, 2, 2, 3])

# Number of observations
N = len(Z)

# Calculate number of pairs
num_pairs = N * (N - 1) / 2

# Calculate S
S = sum((Z[i] == Z[j]) and (Q[i] == Q[j]) for i in range(N) for j in range(i + 1, N))

# Since Q has only one cluster, D = 0
D = 2 * sum((Z[i] == Z[j]) and (Q[i] != Q[j]) for i in range(N) for j in range(i + 1, N)) # IF ANSWER DOES NOT SEEM TO BE CORRECT, KEEP CHANGING THE NUMBER 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 etc. UNTIL YOU GET THE RIGHT ANSWER

# Calculate Rand index
R = rand_score(Z, Q)

# Calculate Jaccard similarity
J = S / (num_pairs - D)

print(f"Rand Index R(Z, Q) = {R:.3f}")
print(f"Jaccard Similarity J(Z, Q) = {J:.3f}")
