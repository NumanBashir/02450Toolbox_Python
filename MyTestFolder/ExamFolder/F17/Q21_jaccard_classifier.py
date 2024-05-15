import numpy as np

# Provided feature table
data = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0],  # O1
    [1, 0, 1, 0, 1, 0, 1, 0],  # O2
    [1, 0, 1, 0, 1, 0, 1, 0],  # O3
    [1, 0, 1, 0, 1, 0, 0, 1],  # O4
    [1, 0, 1, 0, 0, 1, 0, 1],  # O5
    [1, 0, 0, 1, 0, 1, 1, 0],  # O6
    [0, 1, 1, 0, 0, 1, 0, 1],  # O7
    [0, 1, 1, 0, 1, 0, 0, 1],  # O8
    [0, 1, 0, 1, 1, 0, 1, 0],  # O9
    [0, 1, 0, 1, 0, 1, 1, 0]   # O10
])

# Scores: low=0, mid=1, high=2
scores = np.array([1, 0, 0, 1, 2, 0, 2, 2, 0, 2])

# Extracting the vector for O10
O10 = data[-1]

# Initialize Jaccard similarities
jaccard_similarities = []

# Compute Jaccard similarity between O10 and O1-O9
for i in range(len(data) - 1):
    a = O10
    b = data[i]
    f11 = np.sum((a == 1) & (b == 1))
    f00 = np.sum((a == 0) & (b == 0))
    M = len(a)
    jaccard_similarity = f11 / (M - f00)
    jaccard_similarities.append(jaccard_similarity)

# Get indices of the three highest similarities
nearest_neighbors_indices = np.argsort(jaccard_similarities)[-3:]

# Get the scores of the three nearest neighbors
nearest_neighbors_scores = scores[nearest_neighbors_indices]

# Determine the majority class
from collections import Counter
majority_class = Counter(nearest_neighbors_scores).most_common(1)[0][0]

# Map majority class to color
class_map = {0: 'black', 1: 'red', 2: 'blue'}
classified_color = class_map[majority_class]

print(classified_color)
