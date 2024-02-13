from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

# Simulated network traffic data (e.g., number of requests per minute)
X = np.array([[50], [55], [60], [200], [45], [50], [60], [58], [200]])
X_plot = np.linspace(0, 250, 1000)[:, np.newaxis]

# Applying Kernel Density Estimation
kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(X)
log_dens = kde.score_samples(X_plot)

# Plotting the density estimation
plt.fill_between(X_plot[:, 0], np.exp(log_dens), alpha=0.5)
plt.plot(X, np.full(X.shape, -0.01), '|k', markeredgewidth=1)
plt.title('Network Traffic Anomaly Detection')
plt.xlabel('Number of Requests per Minute')
plt.ylabel('Density')
plt.show()
