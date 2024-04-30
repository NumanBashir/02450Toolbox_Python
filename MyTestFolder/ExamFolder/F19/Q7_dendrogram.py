import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Define the correct distance matrix
correct_distance_matrix = np.array([
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

# correct_distance_matrix = np.array([
#     [0.00, 4.84, 0.50, 4.11, 1.07, 4.10, 4.71, 4.70, 4.93],
#     [4.84, 0.00, 4.40, 5.96, 4.12, 2.01, 5.36, 3.59, 3.02],
#     [0.50, 4.40, 0.00, 4.07, 0.72, 3.75, 4.66, 4.48, 4.64],
#     [4.11, 5.96, 4.07, 0.00, 4.48, 4.69, 2.44, 3.68, 4.15],
#     [1.07, 4.12, 0.72, 4.48, 0.00, 3.54, 4.96, 4.62, 4.71],
#     [4.10, 2.01, 3.75, 4.69, 3.54, 0.00, 3.72, 2.23, 1.95],
#     [4.71, 5.36, 4.66, 2.44, 4.96, 3.72, 0.00, 2.03, 2.73],
#     [4.70, 3.59, 4.48, 3.68, 4.62, 2.23, 2.03, 0.00, 0.73],
#     [4.93, 3.02, 4.64, 4.15, 4.71, 1.95, 2.73, 0.73, 0.00]
# ])

# Convert the distance matrix to a condensed form
condensed_matrix = squareform(correct_distance_matrix)

# Perform hierarchical clustering using complete linkage

# TODO: If the question is minimum/single linkage use 'single' otherwise if maximum/complete linkage use 'complete' in the second parameter
linked = linkage(condensed_matrix, 'single')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=range(1, 10), # TODO: Remember to change the range corresponding to the size of matrix/table (if it's 9x9 then range is 1,10 if 10x10 range it 1,11)
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Index of Point')
plt.ylabel('Distance')
plt.show()
