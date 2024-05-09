import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Distance matrix from Table 4
table4 = np.array([
    [0, 725, 800, 150, 1000, 525, 600, 500, 400, 850],
    [725, 0, 75, 575, 275, 1250, 1325, 226, 325, 125],
    [800, 75, 0, 650, 200, 1325, 1400, 300, 400, 51],
    [150, 575, 650, 0, 850, 675, 750, 350, 250, 700],
    [1000, 275, 200, 850, 0, 1525, 1600, 500, 600, 150],
    [525, 1250, 1325, 675, 1525, 0, 75, 1025, 925, 1375],
    [600, 1325, 1400, 750, 1600, 75, 0, 1100, 1000, 1450],
    [500, 226, 300, 350, 500, 1025, 1100, 0, 100, 350],
    [400, 325, 400, 250, 600, 925, 1000, 100, 0, 450],
    [850, 125, 51, 700, 150, 1375, 1450, 350, 450, 0]
])

# Convert the distance matrix to a condensed form
condensed_matrix = squareform(table4)

# Perform hierarchical clustering using complete linkage

# TODO: If the question is minimum/single linkage use 'single' otherwise if maximum/complete linkage use 'complete' in the second parameter
# maximum = complete
# minimum = single
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