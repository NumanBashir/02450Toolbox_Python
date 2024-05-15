import numpy as np

def gaussian_mixture_density(distances, sigma, K):
    """
    Calculate the probability density at a given observation using a Gaussian mixture model.
    
    Parameters:
    distances (array): Array of distances between the observation and each component mean.
    sigma (float): Standard deviation of the Gaussian components.
    K (int): Number of components in the mixture model.
    
    Returns:
    float: The probability density at the given observation.
    """
    # Calculate the constant factor in the Gaussian density function
    factor = 1 / (K * (2 * np.pi * sigma**2)**0.5)
    
    # Calculate the exponent for each distance
    exponents = np.exp(-np.array(distances)**2 / (2 * sigma**2))
    
    # Calculate the probability density
    density = factor * np.sum(exponents)
    
    return density

# Example usage:
distances = [4.93, 3.02, 4.64, 4.15, 4.71, 1.95, 2.73, 0.73]  # Distances from o9 to o1, o2, ..., o8
sigma = 2
K = 8

density = gaussian_mixture_density(distances, sigma, K)
print(f"Probability density at the given observation: {density:.4f}")
