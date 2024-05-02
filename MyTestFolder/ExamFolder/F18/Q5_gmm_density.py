import numpy as np

# Parameters
sigma = 0.5
dimension = 10  # Dimensionality of the multivariate normal

# Distances from o3 to o7, o8, o9 extracted from your corrected Table 2
table2 = np.array([
    [0.0, 2.91, 0.63, 1.88, 1.02, 1.82, 1.92, 1.58, 1.08, 1.43],
    [2.91, 0.0, 3.23, 3.9, 2.88, 3.27, 3.48, 4.02, 3.08, 3.47],
    [0.63, 3.23, 0.0, 2.03, 1.06, 2.15, 2.11, 1.15, 1.09, 1.65],
    [1.88, 3.9, 2.03, 0.0, 2.52, 1.04, 2.25, 2.42, 2.18, 2.17],
    [1.02, 2.88, 1.06, 2.52, 0.0, 2.44, 2.38, 1.53, 1.71, 1.94],
    [1.82, 3.27, 2.15, 1.04, 2.44, 0.0, 1.93, 2.72, 1.98, 1.8],
    [1.92, 3.48, 2.11, 2.25, 2.38, 1.93, 0.0, 2.53, 2.09, 1.66],
    [1.58, 4.02, 1.15, 2.42, 1.53, 2.72, 2.53, 0.0, 1.68, 2.06],
    [1.08, 3.08, 1.09, 2.18, 1.71, 1.98, 2.09, 1.68, 0.0, 1.48],
    [1.43, 3.47, 1.65, 2.17, 1.94, 1.8, 1.66, 2.06, 1.48, 0.0]
])
distances = table2[2, [6, 7, 8]]  # o3 to o7, o8, o9

# Compute the Gaussian densities
denominator = np.sqrt((2 * np.pi * sigma**2)**dimension)
densities = [np.exp(-0.5 * (d**2) / sigma**2) / denominator for d in distances]

# Compute the mean density as the GMM result
p_o3 = np.mean(densities)

print(f"Density at observation o3: {p_o3}")
