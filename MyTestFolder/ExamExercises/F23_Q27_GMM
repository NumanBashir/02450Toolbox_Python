import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the mean and covariance for each cluster for all options
gmm_options = {
    'A': {
        'weights': [3/8, 1/2, 1/8],
        'means': [[1.7, -8.4], [10.7, -11.5], [-7.4, 0.3]],
        'covariances': [[[3.0, 0.8], [0.8, 2.1]], 
                        [[0.6, 0.3], [0.3, 1.1]],
                        [[1.3, -1.5], [-1.5, 2.3]]]
    },
    'B': {
        'weights': [1/8, 3/8, 1/2],
        'means': [[1.7, -8.4], [10.7, -11.5], [-7.4, 0.3]],
        'covariances': [[[1.3, -1.5], [-1.5, 2.3]], 
                        [[0.6, 0.3], [0.3, 1.1]],
                        [[3.0, 0.8], [0.8, 2.1]]]
    },
    'C': {
        'weights': [1/8, 1/2, 3/8],
        'means': [[1.7, -8.4], [10.7, -11.5], [-7.4, 0.3]],
        'covariances': [[[3.0, 0.8], [0.8, 2.1]], 
                        [[0.6, 0.3], [0.3, 1.1]],
                        [[1.3, -1.5], [-1.5, 2.3]]]
    },
    'D': {
        'weights': [3/8, 1/2, 1/8],
        'means': [[1.7, -8.4], [10.7, -11.5], [-7.4, 0.3]],
        'covariances': [[[1.3, -1.5], [-1.5, 2.3]], 
                        [[3.0, 0.8], [0.8, 2.1]],
                        [[0.6, 0.3], [0.3, 1.1]]]
    }
}

# Function to sample from a Gaussian Mixture Model
def sample_gmm(gmm_params, num_samples=1000):
    samples = []
    for weight, mean, cov in zip(gmm_params['weights'], gmm_params['means'], gmm_params['covariances']):
        num_samples_cluster = int(num_samples * weight)
        samples_cluster = np.random.multivariate_normal(mean, cov, num_samples_cluster)
        samples.append(samples_cluster)
    return np.vstack(samples)

# Function to plot samples from a GMM
def plot_gmm_samples(gmm_params, num_samples=1000, title=''):
    samples = sample_gmm(gmm_params, num_samples)
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

# Generate plots for each GMM option
for option, params in gmm_options.items():
    plot_gmm_samples(params, title=f'GMM Option {option}')
