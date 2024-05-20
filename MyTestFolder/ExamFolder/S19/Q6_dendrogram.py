import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Define the correct distance matrix
correct_distance_matrix = np.array([
    [0.0, 2.0, 5.7, 0.9, 2.9, 1.8, 2.7, 3.7, 5.3, 5.1],
    [2.0, 0.0, 5.6, 2.4, 2.5, 3.0, 3.5, 4.3, 6.0, 6.2],
    [5.7, 5.6, 0.0, 5.0, 5.1, 4.0, 3.3, 5.4, 1.2, 1.8],
    [0.9, 2.4, 5.0, 0.0, 2.7, 2.1, 2.2, 3.5, 4.6, 4.4],
    [2.9, 2.5, 5.1, 2.7, 0.0, 3.5, 3.7, 4.0, 5.8, 5.7],
    [1.8, 3.0, 4.0, 2.1, 3.5, 0.0, 1.7, 5.3, 3.8, 3.7],
    [2.7, 3.5, 3.3, 2.2, 3.7, 1.7, 0.0, 4.2, 3.1, 3.2],
    [3.7, 4.3, 5.4, 3.5, 4.0, 5.3, 4.2, 0.0, 5.5, 6.0],
    [5.3, 6.0, 1.2, 4.6, 5.8, 3.8, 3.1, 5.5, 0.0, 2.1],
    [5.1, 6.2, 1.8, 4.4, 5.7, 3.7, 3.2, 6.0, 2.1, 0.0]
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
           labels=range(1, 11), # TODO: Remember to change the range corresponding to the size of matrix/table (if it's 9x9 then range is 1,10 if 10x10 range it 1,11)
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Index of Point')
plt.ylabel('Distance')
plt.show()
