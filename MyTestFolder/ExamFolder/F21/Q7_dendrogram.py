import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

correct_distance_matrix = np.array([
    [0.0, 53.8, 87.0, 67.4, 67.5, 71.2, 65.2, 117.9, 56.1, 90.3, 109.8],
    [53.8, 0.0, 69.9, 75.5, 62.9, 58.0, 63.0, 135.0, 84.1, 107.9, 131.5],
    [87.0, 69.9, 0.0, 49.7, 38.5, 19.3, 35.5, 91.8, 76.9, 78.7, 89.1],
    [67.4, 75.5, 49.7, 0.0, 24.2, 47.2, 47.0, 62.3, 33.4, 37.2, 60.0],
    [67.5, 62.9, 38.5, 24.2, 0.0, 37.7, 41.7, 79.5, 52.4, 60.2, 78.9],
    [71.2, 58.0, 19.3, 47.2, 37.7, 0.0, 21.5, 95.6, 68.3, 78.4, 91.0],
    [65.2, 63.0, 35.5, 47.0, 41.7, 21.5, 0.0, 96.0, 64.3, 75.5, 89.4],
    [117.9, 135.0, 91.8, 62.3, 79.5, 95.6, 96.0, 0.0, 66.9, 44.3, 24.2],
    [56.1, 84.1, 76.9, 33.4, 52.4, 68.3, 64.3, 66.9, 0.0, 39.2, 60.7],
    [90.3, 107.9, 78.7, 37.2, 60.2, 78.4, 75.5, 44.3, 39.2, 0.0, 39.4],
    [109.8, 131.5, 89.1, 60.0, 78.9, 91.0, 89.4, 24.2, 60.7, 39.4, 0.0]
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
           labels=range(1, 12), # TODO: Remember to change the range corresponding to the size of matrix/table (if it's 9x9 then range is 1,10 if 10x10 range it 1,11)
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Index of Point')
plt.ylabel('Distance')
plt.show()
