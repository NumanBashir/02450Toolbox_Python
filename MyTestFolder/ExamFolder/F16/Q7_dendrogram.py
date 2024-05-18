import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Distance matrix from the provided table
correct_distance_matrix = np.array([
    [0, 0.534, 1.257, 1.671, 1.090, 1.315, 1.484, 1.253, 1.418],
    [0.534, 0, 0.727, 2.119, 1.526, 1.689, 1.214, 0.997, 1.056],
    [1.257, 0.727, 0, 2.809, 2.220, 2.342, 1.088, 0.965, 0.807],
    [1.671, 2.119, 2.809, 0, 0.601, 0.540, 3.135, 2.908, 3.087],
    [1.090, 1.526, 2.220, 0.601, 0, 0.331, 2.563, 2.338, 2.500],
    [1.315, 1.689, 2.342, 0.540, 0.331, 0, 2.797, 2.567, 2.708],
    [1.484, 1.214, 1.088, 3.135, 2.563, 2.797, 0, 0.275, 0.298],
    [1.253, 0.997, 0.965, 2.908, 2.338, 2.567, 0.275, 0, 0.343],
    [1.418, 1.056, 0.807, 3.087, 2.500, 2.708, 0.298, 0.343, 0]
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
linked = linkage(condensed_matrix, 'complete')

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
