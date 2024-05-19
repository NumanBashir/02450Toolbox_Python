import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Distance matrix from the provided table
correct_distance_matrix = np.array([
    [0.0, 2.6, 2.8, 2.1, 0.8, 1.7, 3.9, 3.8, 4.1, 4.2],
    [2.6, 0.0, 1.8, 2.3, 2.7, 1.4, 1.9, 1.7, 2.0, 2.2],
    [2.8, 1.8, 0.0, 1.2, 2.9, 2.1, 2.2, 1.6, 2.3, 2.0],
    [2.1, 2.3, 1.2, 0.0, 2.4, 1.8, 2.6, 2.5, 3.0, 2.9],
    [0.8, 2.7, 2.9, 2.4, 0.0, 2.0, 4.2, 4.0, 4.4, 4.5],
    [1.7, 1.4, 2.1, 1.8, 2.0, 0.0, 2.4, 2.6, 2.7, 3.0],
    [3.9, 1.9, 2.2, 2.6, 4.2, 2.4, 0.0, 1.2, 0.6, 1.3],
    [3.8, 1.7, 1.6, 2.5, 4.0, 2.6, 1.2, 0.0, 1.0, 0.6],
    [4.1, 2.0, 2.3, 3.0, 4.4, 2.7, 0.6, 1.0, 0.0, 0.9],
    [4.2, 2.2, 2.0, 2.9, 4.5, 3.0, 1.3, 0.6, 0.9, 0.0]
])

# Convert the distance matrix to a condensed form
condensed_matrix = squareform(correct_distance_matrix)

# Perform hierarchical clustering using complete linkage
linked = linkage(condensed_matrix, 'complete')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendro = dendrogram(linked,
                    orientation='top',
                    labels=range(1, 11),  # Update the range based on the size of the matrix
                    distance_sort='descending',
                    show_leaf_counts=True)

# Add horizontal lines at each clustering step with labels
for i, d, c in zip(dendro['icoord'], dendro['dcoord'], dendro['color_list']):
    x = 0.5 * sum(i[1:3])
    y = d[1]
    plt.plot(i[1:3], [y, y], c)
    plt.text(x, y, f'{y:.2f}', va='bottom', ha='left', fontsize=12, color=c)

plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Index of Point')
plt.ylabel('Distance')
plt.show()
