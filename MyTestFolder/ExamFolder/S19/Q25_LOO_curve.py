import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# Given dataset
data = np.array([3.918, -6.35, -2.677, -3.003])

# Function to calculate leave-one-out log-likelihood
def loo_log_likelihood(data, sigma):
    n = len(data)
    log_likelihood = 0
    
    for i in range(n):
        # Leave one out
        loo_data = np.delete(data, i)
        
        # Fit KDE on leave-one-out data
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(loo_data[:, None])
        
        # Calculate log-likelihood for the left out point
        log_likelihood += kde.score_samples(data[i].reshape(1, -1))[0]
    
    # Return the average negative log-likelihood
    return -log_likelihood / n

# Range of sigma values to evaluate
sigma_values = np.linspace(2, 8, 100)

# Calculate LOO log-likelihood for each sigma
loo_values = [loo_log_likelihood(data, sigma) for sigma in sigma_values]

# Plot the LOO curve
plt.figure(figsize=(10, 6))
plt.plot(sigma_values, loo_values, label='LOO Curve')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$E(\sigma)$')
plt.title('Leave-One-Out Cross-Validation Curve')
plt.legend()
plt.grid(True)
plt.show()
