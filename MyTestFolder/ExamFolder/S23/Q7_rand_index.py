import numpy as np
from sklearn.metrics import rand_score, jaccard_score

# Ground-truth clustering Z (using class labels)
Z = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

# Obtained clustering Q (from dendrogram clusters)
Q = np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 3])

# Calculate Rand index
R = rand_score(Z, Q)
print(f"Rand Index R(Z, Q) ≈ {R:.3f}")
