import numpy as np # type: ignore

# Parameters for the Gaussian Mixture Model
#TODO update weights means variances based on task
weights = np.array([0.37, 0.29, 0.34])  # Weights for each cluster
means = np.array([6.12, 6.55, 6.93])    # Means for each cluster
variances = np.array([0.09, 0.13, 0.12])  # Variances for each cluster

# Function to compute Gaussian density
def gaussian_density(x, mu, sigma2):
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

#Observation for O8 TODO: update x based on the observation
x = 6.9

# Compute the density of this observation for each cluster
densities = np.array([gaussian_density(x, mu, sigma2) for mu, sigma2 in zip(means, variances)])

# Calculate the weighted densities to find the contribution from each cluster
weighted_densities = weights * densities

# Compute the probability of belonging to cluster 2
probability_cluster_2 = weighted_densities[1] / sum(weighted_densities)

print("Probability of O8 belonging to cluster 2:", probability_cluster_2)
