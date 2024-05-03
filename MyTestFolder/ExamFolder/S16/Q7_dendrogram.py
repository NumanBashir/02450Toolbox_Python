import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

correct_distance_matrix = np.array([
    [0.00, 4.84, 0.50, 4.11, 1.07, 4.10, 4.71, 4.70, 4.93],
    [4.84, 0.00, 4.40, 5.96, 4.12, 2.01, 5.36, 3.59, 3.02],
    [0.50, 4.40, 0.00, 4.07, 0.72, 3.75, 4.66, 4.48, 4.64],
    [4.11, 5.96, 4.07, 0.00, 4.48, 4.69, 2.44, 3.68, 4.15],
    [1.07, 4.12, 0.72, 4.48, 0.00, 3.54, 4.96, 4.62, 4.71],
    [4.10, 2.01, 3.75, 4.69, 3.54, 0.00, 3.72, 2.23, 1.95],
    [4.71, 5.36, 4.66, 2.44, 4.96, 3.72, 0.00, 2.03, 2.73],
    [4.70, 3.59, 4.48, 3.68, 4.62, 2.23, 2.03, 0.00, 0.73],
    [4.93, 3.02, 4.64, 4.15, 4.71, 1.95, 2.73, 0.73, 0.00]
])

# Convert the distance matrix to a condensed form
condensed_matrix = squareform(correct_distance_matrix)

# Perform hierarchical clustering using complete linkage

# TODO: If the question is minimum/single linkage use 'single' otherwise if maximum/complete linkage use 'complete' in the second parameter
# maximum = complete
# minimum = single
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