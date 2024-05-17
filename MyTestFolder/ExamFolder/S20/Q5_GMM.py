import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the mean and covariance for each cluster for all options
gmm_options = {
    'A': {
        'weights': [0.1425, 0.3235, 0.5340],
        'means': [[3.3884, 0.7424], [-1.9482, 0.6132], [0.2756, -0.5696]],
        'covariances': [[[0.1695, 0.0665], [0.0665, 0.1104]], 
                        [[1.2137, -0.0703], [-0.0703, 0.3773]],
                        [[2.0700, 0.1876], [0.1876, 0.1037]]]
    },
    'B': {
        'weights': [0.1425, 0.3235, 0.5340],
        'means': [[-1.9482, 0.6132], [3.3884, 0.7424], [0.2756, -0.5696]],
        'covariances': [[[0.1695, 0.0665], [0.0665, 0.1104]], 
                        [[1.2137, -0.0703], [-0.0703, 0.3773]],
                        [[2.0700, 0.1876], [0.1876, 0.1037]]]
    },
    'C': {
        'weights': [0.3235, 0.1425, 0.5340],
        'means': [[-1.9482, 0.6132], [3.3884, 0.7424], [0.2756, -0.5696]],
        'covariances': [[[0.1695, 0.0665], [0.0665, 0.1104]], 
                        [[2.0700, 0.1876], [0.1876, 0.1037]],
                        [[1.2137, -0.0703], [-0.0703, 0.3773]]]
    },
    'D': {
        'weights': [0.3235, 0.1425, 0.5340],
        'means': [[-1.9482, 0.6132], [3.3884, 0.7424], [0.2756, -0.5696]],
        'covariances': [[[0.1695, 0.0665], [0.0665, 0.1104]], 
                        [[2.0700, 0.1876], [0.1876, 0.1037]],
                        [[1.2137, -0.0703], [-0.0703, 0.3773]]]
    }
}

# Function to sample from a Gaussian Mixture Model
def sample_gmm(gmm_params, num_samples=1000):
    samples = []
    labels = []
    for idx, (weight, mean, cov) in enumerate(zip(gmm_params['weights'], gmm_params['means'], gmm_params['covariances'])):
        num_samples_cluster = int(num_samples * weight)
        samples_cluster = np.random.multivariate_normal(mean, cov, num_samples_cluster)
        samples.append(samples_cluster)
        labels.append(np.full(num_samples_cluster, idx))
    return np.vstack(samples), np.concatenate(labels)

# Function to plot samples from a GMM with color coding
def plot_gmm_samples(gmm_params, num_samples=1000, title=''):
    samples, labels = sample_gmm(gmm_params, num_samples)
    plt.figure(figsize=(8, 6))
    for idx, color in enumerate(['red', 'green', 'blue']):
        plt.scatter(samples[labels == idx, 0], samples[labels == idx, 1], alpha=0.5, label=f'Cluster {idx + 1}', color=color)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate plots for each GMM option
for option, params in gmm_options.items():
    plot_gmm_samples(params, title=f'GMM Option {option}')
