import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Define the correct distance matrix
correct_distance_matrix = np.array([
    [0, 0.2606, 1.1873, 2.4946, 2.9510, 2.5682, 3.4535, 2.4698],
    [0.2606, 0, 1.2796, 2.4442, 2.8878, 2.4932, 3.3895, 2.4216],
    [1.1873, 1.2796, 0, 2.8294, 3.6892, 2.9147, 4.1733, 2.2386],
    [2.4946, 2.4442, 2.8294, 0, 1.4852, 0.2608, 2.2941, 1.8926],
    [2.9510, 2.8878, 3.6892, 1.4852, 0, 1.5155, 1.0296, 3.1040],
    [2.5682, 2.4932, 2.9147, 0.2608, 1.5155, 0, 2.3316, 1.8870],
    [3.4535, 3.3895, 4.1733, 2.2941, 1.0296, 2.3316, 0, 3.7588],
    [2.4698, 2.4216, 2.2386, 1.8926, 3.1040, 1.8870, 3.7588, 0]
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
           labels=range(1, 9), # TODO: Remember to change the range corresponding to the size of matrix/table (if it's 9x9 then range is 1,10 if 10x10 range it 1,11)
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Index of Point')
plt.ylabel('Distance')
plt.show()
