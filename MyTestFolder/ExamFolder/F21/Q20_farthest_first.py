import numpy as np

# Data points
x = np.array([0.4, 1.7, 3.7, 4.6])

# Given initial centroid
mu1 = 1.7

# Find the farthest point from the initial centroid
distances = np.abs(x - mu1)
mu2 = x[np.argmax(distances)]

# Find the next farthest point maximizing the distance to both mu1 and mu2
dist_to_mu1 = np.abs(x - mu1)
dist_to_mu2 = np.abs(x - mu2)
min_distances = np.minimum(dist_to_mu1, dist_to_mu2)
mu3 = x[np.argmax(min_distances)]

print(f"The next two centroids are: μ2 = {mu2}, μ3 = {mu3}")
