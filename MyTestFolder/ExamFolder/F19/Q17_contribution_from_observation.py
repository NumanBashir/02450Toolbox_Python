import numpy as np


table3 = np.array([
    [0.0, 4.2, 8.3, 3.9, 3.8, 4.6, 6.3, 4.8, 7.1, 4.9],
    [4.2, 0.0, 7.4, 2.6, 3.0, 3.2, 5.3, 3.1, 6.6, 4.6],
    [8.3, 7.4, 0.0, 6.3, 7.1, 5.5, 2.8, 5.4, 2.4, 5.3],
    [3.9, 2.6, 6.3, 0.0, 1.5, 1.6, 4.1, 1.8, 5.3, 2.4],
    [3.8, 3.0, 7.1, 1.5, 0.0, 2.4, 4.9, 2.8, 5.8, 3.2],
    [4.6, 3.2, 5.5, 1.6, 2.4, 0.0, 3.7, 1.7, 4.8, 2.3],
    [6.3, 5.3, 2.8, 4.1, 4.9, 3.7, 0.0, 3.8, 1.9, 3.6],
    [4.8, 3.1, 5.4, 1.8, 2.8, 1.7, 3.8, 0.0, 4.9, 2.1],
    [7.1, 6.6, 2.4, 5.3, 5.8, 4.8, 1.9, 4.9, 0.0, 4.4],
    [4.9, 4.6, 5.3, 2.4, 3.2, 2.3, 3.6, 2.1, 4.4, 0.0]
])

xT = [-1.76, -0, 0.06, 0.08, 0.65, 1.3]
table7_y = np.array([12, 6, 8, 10, 4, 2])


# K-nearest neighbors for observation 1 (zero-indexed, so observation 1 is at index 0)
K = 3

# For observation 1, find the K nearest neighbors (excluding itself)
# We use argsort to get indices of sorted distances, and then take the first K
neighbor_indices = np.argsort(table3[0])[1:K+1]

# Calculate the predicted y-value by averaging the y-values of the nearest neighbors
predicted_y1 = np.mean(table7_y[neighbor_indices])

# Calculate the squared loss for observation 1
squared_loss_obs1 = (table7_y[0] - predicted_y1) ** 2

print(squared_loss_obs1)