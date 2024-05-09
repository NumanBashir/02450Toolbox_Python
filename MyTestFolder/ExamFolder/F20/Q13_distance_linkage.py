import numpy as np

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

# Clusters (indices are zero-based, so o6 = index 5 etc.)
C2_indices = [5, 6]  # Corresponds to o6, o7
C3_indices = [7, 8, 9]  # Corresponds to o8, o9, o10

# TODO CHOOSE ANY OF THE FOLLOWING DISTANCE FUNCTIONS ACCORDING TO THE QUESTION
def average_linkage_distance(cluster1, cluster2, distance_matrix):
    total_distance = 0
    count = 0
    for i in cluster1:
        for j in cluster2:
            total_distance += distance_matrix[i, j]
            count += 1
    return total_distance / count

def single_linkage_distance(cluster1, cluster2, distance_matrix):
    return np.min([distance_matrix[i, j] for i in cluster1 for j in cluster2])

def complete_linkage_distance(cluster1, cluster2, distance_matrix):
    return np.max([distance_matrix[i, j] for i in cluster1 for j in cluster2])

def centroid_linkage_distance(cluster1, cluster2, data):
    centroid1 = np.mean(data[cluster1], axis=0)
    centroid2 = np.mean(data[cluster2], axis=0)
    return np.linalg.norm(centroid1 - centroid2)

def wards_method_distance(cluster1, cluster2, data):
    def sse(cluster, data):
        centroid = np.mean(data[cluster], axis=0)
        return np.sum((data[cluster] - centroid) ** 2)
    
    original_sse = sse(cluster1, data) + sse(cluster2, data)
    combined_cluster = np.concatenate([cluster1, cluster2])
    combined_sse = sse(combined_cluster, data)
    return combined_sse - original_sse

# TODO MAKE SURE TO CHANGE THE FUNCTION NAME ACCORDING TO THE QUESTION AND CODE ABOVE
distance_C2_C3 = average_linkage_distance(C2_indices, C3_indices, table4)
print(f"Distance between C2 and C3 using average linkage: {distance_C2_C3:.1f}")
