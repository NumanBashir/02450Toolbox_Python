import numpy as np

def entropy(counts):
    total = sum(counts)
    probabilities = [c / total for c in counts]
    return -sum(p * np.log(p) if p > 0 else 0 for p in probabilities)

# Define the cluster assignments for Z and Q
# From the provided solution
Z_clusters = {1: {4, 6}, 2: {1, 3, 5, 7, 8, 9, 10}, 3: {2}}
Q_clusters = {1: {1, 2, 3, 4, 5, 6, 7, 8, 9}, 2: {10}, 3: {}}

# N is the total number of observations
N = 10

# Convert cluster dicts to list of labels
Z_labels = np.zeros(N, dtype=int)
Q_labels = np.zeros(N, dtype=int)

for k, indices in Z_clusters.items():
    for i in indices:
        Z_labels[i-1] = k

for k, indices in Q_clusters.items():
    for i in indices:
        Q_labels[i-1] = k

# Calculate the contingency table
contingency_table = np.zeros((len(Z_clusters), len(Q_clusters)), dtype=int)
for i in range(N):
    contingency_table[Z_labels[i] - 1, Q_labels[i] - 1] += 1

# Calculate probabilities
P_joint = contingency_table / N
P_Z = np.sum(P_joint, axis=1)
P_Q = np.sum(P_joint, axis=0)

# Calculate entropies
H_Z = entropy(P_Z)
H_Q = entropy(P_Q)
H_ZQ = -np.sum(P_joint * np.log(P_joint + (P_joint == 0)))  # Add small number to avoid log(0)

# Mutual Information (MI)
MI = H_Z + H_Q - H_ZQ

# Normalized Mutual Information (NMI)
NMI = MI / np.sqrt(H_Z * H_Q)

print("Normalized Mutual Information (NMI):", NMI)
