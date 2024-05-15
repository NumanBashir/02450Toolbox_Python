import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Define the correct distance matrix
correct_distance_matrix = np.array([
    [0.0, 5.0, 7.7, 6.1, 4.2, 11.0, 7.3, 9.0, 11.3, 1.4],
    [5.0, 0.0, 5.4, 4.0, 7.5, 7.9, 5.3, 6.8, 11.9, 3.5],
    [7.7, 5.4, 0.0, 5.2, 7.2, 6.1, 7.8, 6.7, 12.9, 6.4],
    [6.1, 4.0, 5.2, 0.0, 5.1, 5.4, 8.4, 3.3, 8.1, 4.8],
    [4.2, 7.5, 7.2, 5.1, 0.0, 8.7, 8.8, 6.6, 7.7, 4.1],
    [11.0, 7.9, 6.1, 5.4, 8.7, 0.0, 12.0, 4.2, 9.3, 9.8],
    [7.3, 5.3, 7.8, 8.4, 8.8, 12.0, 0.0, 11.0, 16.3, 6.7],
    [9.0, 6.8, 6.7, 3.3, 6.6, 4.2, 11.0, 0.0, 6.2, 7.8],
    [11.3, 11.9, 12.9, 8.1, 7.7, 9.3, 16.3, 6.2, 0.0, 10.4],
    [1.4, 3.5, 6.4, 4.8, 4.1, 9.8, 6.7, 7.8, 10.4, 0.0]
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
