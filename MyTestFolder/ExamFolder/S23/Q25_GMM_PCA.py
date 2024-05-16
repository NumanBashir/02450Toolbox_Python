import numpy as np

# Given component weights and covariance matrices
weights = np.array([0.5, 0.49, 0.01])
cov_matrices = [
    np.array([[1.1, 2.0], [2.0, 5.5]]),
    np.array([[1.1, 0.0], [0.0, 5.5]]),
    np.array([[1.5, 0.0], [0.0, 1.5]])
]

# Compute the weighted average covariance matrix
weighted_cov = sum(w * cov for w, cov in zip(weights, cov_matrices))

# Perform eigenvalue decomposition
eigvals, eigvecs = np.linalg.eigh(weighted_cov)

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigvals)[::-1]
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]

# Principal components
v1 = eigvecs[:, 0]
v2 = eigvecs[:, 1]

print("Weighted Average Covariance Matrix:")
print(weighted_cov)
print("\nEigenvalues:")
print(eigvals)
print("\nPrincipal Component Directions (Eigenvectors):")
print("v1:", v1)
print("v2:", v2)

# Check against options
options = {
    "A": ([-0.3, -1.0], [-1.0, 0.3]),
    "B": ([-0.3, -1.0], [0.0, 1.0]),
    "C": ([0.7, -0.7], [0.7, 0.7]),
    "D": ([0.7, 0.7], [0.7, -0.7])
}

for option, (v1_opt, v2_opt) in options.items():
    if np.allclose(v1, v1_opt, atol=0.1) and np.allclose(v2, v2_opt, atol=0.1):
        print(f"Option {option} matches computed principal components.")
