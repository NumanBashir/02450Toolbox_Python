import numpy as np

# DOES NOT WORK YET!!!!

# Provided data and class labels
data = np.array([
    [1, 1, 1, 0, 0],  # o1
    [1, 1, 1, 0, 0],  # o2
    [1, 1, 1, 0, 0],  # o3
    [1, 1, 1, 0, 0],  # o4
    [1, 1, 1, 0, 0],  # o5
    [0, 1, 1, 0, 0],  # o6
    [0, 1, 0, 1, 1],  # o7
    [1, 1, 1, 0, 0],  # o8
    [1, 0, 1, 0, 0],  # o9
    [0, 0, 0, 1, 1],  # o10
    [0, 1, 0, 1, 1],  # o11
])

class_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

# Function to calculate Gini impurity
def gini_impurity(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# Calculate the Gini impurity for the whole dataset
gini_full = gini_impurity(class_labels)

# Split the dataset based on the median value of the first feature
median_value = np.median(data[:, 0])
left_split = class_labels[data[:, 0] <= median_value]
right_split = class_labels[data[:, 0] > median_value]

# Calculate the Gini impurity for the splits
gini_left = gini_impurity(left_split)
gini_right = gini_impurity(right_split)

# Calculate the weighted average Gini impurity of the splits
weight_left = len(left_split) / len(class_labels)
weight_right = len(right_split) / len(class_labels)
gini_split = weight_left * gini_left + weight_right * gini_right

# Calculate the purity gain
purity_gain = gini_full - gini_split

print("Gini impurity of the full dataset:", gini_full)
print("Gini impurity of the left split:", gini_left)
print("Gini impurity of the right split:", gini_right)
print("Weighted Gini impurity of the split:", gini_split)
print("Purity gain:", purity_gain)
