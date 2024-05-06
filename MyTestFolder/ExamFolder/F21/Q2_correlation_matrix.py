import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the covariance matrices from options A to D
cov_matrices = {
    'A': np.array([[564.3, -77.5, 292.5, -388.5, 164.0],
                   [-77.5, 271.5, -72.5, 36.0, -42.0],
                   [292.5, -72.5, 392.4, -324.8, 248.1],
                   [-388.5, 36.0, -324.8, 369.9, -241.4],
                   [164.0, -42.0, 248.1, -241.4, 224.6]]),
    'B': np.array([[-564.3, -77.5, 292.5, -388.5, 164.0],
                   [-77.5, -271.5, -72.5, 36.0, -42.0],
                   [292.5, -72.5, -392.4, -324.8, 248.1],
                   [-388.5, 36.0, -324.8, -369.9, -241.4],
                   [164.0, -42.0, 248.1, -241.4, -224.6]]),
    'C': np.array([[224.6, 248.1, -42.0, -241.4, 164.0],
                   [248.1, 392.4, -72.5, -324.8, 292.5],
                   [-42.0, -72.5, 271.5, 36.0, -77.5],
                   [-241.4, -324.8, 36.0, 369.9, -388.5],
                   [164.0, 292.5, -77.5, -388.5, 564.3]]),
    'D': np.array([[-224.6, 248.1, -42.0, -241.4, 164.0],
                   [248.1, -392.4, -72.5, -324.8, 292.5],
                   [-42.0, -72.5, 271.5, 36.0, -77.5],
                   [-241.4, -324.8, 36.0, -369.9, -388.5],
                   [164.0, 292.5, -77.5, -388.5, -564.3]])
}

# Generate and plot data for each covariance matrix
mean = np.zeros(5)  # Assuming mean of zero for simplicity

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 20))

for ax, (label, cov) in zip(axes, cov_matrices.items()):
    # Generate multivariate normal data
    data = np.random.multivariate_normal(mean, cov, size=1000)
    
    # Plot scatter plots of the first two dimensions
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5)
    ax.set_title(f"Scatter Plot for Matrix {label}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)

plt.tight_layout()
plt.show()
