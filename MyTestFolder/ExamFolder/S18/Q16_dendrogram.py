import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

correct_distance_matrix = np.array([
    [0, 8.55, 0.43, 1.25, 1.14, 3.73, 2.72, 1.63, 1.68, 1.28],
    [8.55, 0, 8.23, 8.13, 8.49, 6.84, 8.23, 8.28, 8.13, 7.66],
    [0.43, 8.23, 0, 1.09, 1.10, 3.55, 2.68, 1.50, 1.52, 1.05],
    [1.25, 8.13, 1.09, 0, 1.23, 3.21, 2.17, 1.29, 1.33, 0.56],
    [1.14, 8.49, 1.10, 1.23, 0, 3.20, 2.68, 1.56, 1.50, 1.28],
    [3.73, 6.84, 3.55, 3.21, 3.20, 0, 2.98, 2.66, 2.50, 3.00],
    [2.72, 8.23, 2.68, 2.17, 2.68, 2.98, 0, 2.28, 2.30, 2.31],
    [1.63, 8.28, 1.50, 1.29, 1.56, 2.66, 2.28, 0, 0.25, 1.46],
    [1.68, 8.13, 1.52, 1.33, 1.50, 2.50, 2.30, 0.25, 0, 1.44],
    [1.28, 7.66, 1.05, 0.56, 1.28, 3.00, 2.31, 1.46, 1.44, 0]
])

# Convert the distance matrix to a condensed form
condensed_matrix = squareform(correct_distance_matrix)

# Perform hierarchical clustering using complete linkage

# TODO: If the question is minimum/single linkage use 'single' otherwise if maximum/complete linkage use 'complete' in the second parameter
# maximum = complete
# minimum = single
# = average
# = weighted
# = centroid
linked = linkage(condensed_matrix, 'average')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=range(1, 11), # TODO: Remember to change the range corresponding to the size of matrix/table (if it's 9x9 then range is 1,10 if 10x10 range it 1,11)
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Index of Point')
plt.ylabel('Distance')
plt.show()
