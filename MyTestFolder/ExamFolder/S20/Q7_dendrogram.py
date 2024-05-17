import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

correct_distance_matrix = np.array([
    [0.0, 1.7, 1.4, 0.4, 2.2, 3.7, 5.2, 0.2, 4.3, 6.8, 6.0],
    [1.7, 0.0, 1.0, 2.0, 1.3, 2.6, 4.5, 1.8, 3.2, 5.9, 5.2],
    [1.4, 1.0, 0.0, 1.7, 0.9, 2.4, 4.1, 1.5, 3.0, 5.5, 4.8],
    [0.4, 2.0, 1.7, 0.0, 2.6, 4.0, 5.5, 0.3, 4.6, 7.1, 6.3],
    [2.2, 1.3, 0.9, 2.6, 0.0, 1.7, 3.4, 2.4, 2.1, 4.8, 4.1],
    [3.7, 2.6, 2.4, 4.0, 1.7, 0.0, 2.0, 3.8, 1.6, 3.3, 2.7],
    [5.2, 4.5, 4.1, 5.5, 3.4, 2.0, 0.0, 5.4, 2.5, 1.6, 0.9],
    [0.2, 1.8, 1.5, 0.3, 2.4, 3.8, 5.4, 0.0, 4.4, 6.9, 6.1],
    [4.3, 3.2, 3.0, 4.6, 2.1, 1.6, 2.5, 4.4, 0.0, 3.4, 2.9],
    [6.8, 5.9, 5.5, 7.1, 4.8, 3.3, 1.6, 6.9, 3.4, 0.0, 1.0],
    [6.0, 5.2, 4.8, 6.3, 4.1, 2.7, 0.9, 6.1, 2.9, 1.0, 0.0]
])

# Convert the distance matrix to a condensed form
condensed_matrix = squareform(correct_distance_matrix)

# Perform hierarchical clustering using complete linkage

# TODO: If the question is minimum/single linkage use 'single' otherwise if maximum/complete linkage use 'complete' in the second parameter
# maximum = complete
# minimum = single
linked = linkage(condensed_matrix, 'complete')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=range(1, 12), # TODO: Remember to change the range corresponding to the size of matrix/table (if it's 9x9 then range is 1,10 if 10x10 range it 1,11)
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Index of Point')
plt.ylabel('Distance')
plt.show()