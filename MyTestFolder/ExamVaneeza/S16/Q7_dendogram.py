import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Distance matrix from the provided table
distance_matrix = np.array([
    [0.00, 4.84, 0.50, 4.11, 1.07, 4.10, 4.71, 4.70, 4.93],
    [4.84, 0.00, 4.00, 5.96, 4.12, 2.01, 5.36, 3.59, 3.02],
    [0.50, 4.00, 0.00, 4.07, 0.72, 3.75, 4.66, 4.48, 4.64],
    [4.11, 5.96, 4.07, 0.00, 4.48, 4.69, 2.44, 3.68, 4.15],
    [1.07, 4.12, 0.72, 4.48, 0.00, 3.54, 4.96, 4.62, 4.71],
    [4.10, 2.01, 3.75, 4.69, 3.54, 0.00, 3.72, 2.23, 1.95],
    [4.71, 5.36, 4.66, 2.44, 4.96, 3.72, 0.00, 2.03, 2.73],
    [4.70, 3.59, 4.48, 3.68, 4.62, 2.23, 2.03, 0.00, 0.73],
    [4.93, 3.02, 4.64, 4.15, 4.71, 1.95, 2.73, 0.73, 0.00]
])

# Convert the distance matrix to a condensed distance matrix (1D array)
condensed_dist_matrix = sch.distance.squareform(distance_matrix)

# Perform hierarchical clustering using single linkage
linked = sch.linkage(condensed_dist_matrix, method='single')

# Create the dendrogram
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(linked, labels=['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9'])
plt.title('Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
plt.show()
