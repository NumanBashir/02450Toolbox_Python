import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

correct_distance_matrix = np.array([
    [0.0, 1.3, 4.1, 3.8, 4.5, 2.4, 3.2, 2.7, 3.0, 3.9],
    [1.3, 0.0, 3.2, 3.1, 4.7, 2.3, 2.6, 2.2, 2.7, 4.2],
    [4.1, 3.2, 0.0, 0.4, 4.9, 2.7, 1.1, 1.6, 2.4, 4.8],
    [3.8, 3.1, 0.4, 0.0, 4.6, 2.5, 0.9, 1.3, 2.1, 4.5],
    [4.5, 4.7, 4.9, 4.6, 0.0, 3.1, 4.4, 3.7, 2.8, 2.3],
    [2.4, 2.3, 2.7, 2.5, 3.1, 0.0, 1.8, 1.2, 0.9, 2.8],
    [3.2, 2.6, 1.1, 0.9, 4.4, 1.8, 0.0, 1.0, 1.7, 4.1],
    [2.7, 2.2, 1.6, 1.3, 3.7, 1.2, 1.0, 0.0, 1.1, 3.6],
    [3.0, 2.7, 2.4, 2.1, 2.8, 0.9, 1.7, 1.1, 0.0, 2.9],
    [3.9, 4.2, 4.8, 4.5, 2.3, 2.8, 4.1, 3.6, 2.9, 0.0]
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
           labels=range(1, 11), # TODO: Remember to change the range corresponding to the size of matrix/table (if it's 9x9 then range is 1,10 if 10x10 range it 1,11)
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Index of Point')
plt.ylabel('Distance')
plt.axhline(y=2.25, color='r', linestyle='--') # Change the value of the y to make the intersection of the clusters
plt.show()


# Perform hierarchical clustering using single linkage
linked = linkage(squareform(correct_distance_matrix), method='single')

# Assuming you cut the dendrogram to form 3 clusters
cluster_assignments = fcluster(linked, t=3, criterion='maxclust')

# Simulate some true labels (ground-truth) for comparison
true_labels = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 3])  # Example ground-truth labels

# Calculate the Rand Index
rand_index = adjusted_rand_score(true_labels, cluster_assignments)

print("Rand Index:", rand_index)


