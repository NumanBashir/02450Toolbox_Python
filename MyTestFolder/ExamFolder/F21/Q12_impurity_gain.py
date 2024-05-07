import numpy as np

# Data from the table, with class labels assigned based on the provided information
data = np.array([
    [0, 0, 0, 1, 0, 0, 0, 1],  # o1 -> C1
    [0, 0, 1, 0, 0, 1, 0, 1],  # o2 -> C1
    [0, 0, 1, 0, 0, 1, 0, 1],  # o3 -> C2
    [0, 1, 0, 0, 0, 1, 0, 1],  # o4 -> C2
    [0, 0, 0, 0, 0, 1, 0, 1],  # o5 -> C2
    [0, 0, 1, 0, 1, 1, 0, 1],  # o6 -> C2
    [0, 0, 1, 0, 0, 1, 0, 1],  # o7 -> C2
    [1, 1, 0, 0, 0, 0, 1, 1],  # o8 -> C3
    [0, 1, 0, 0, 0, 0, 0, 1],  # o9 -> C3
    [0, 1, 0, 0, 0, 1, 0, 1],  # o10 -> C3
    [1, 1, 0, 0, 0, 0, 0, 0],  # o11 -> C3
])

# Class labels as described
class_labels = np.array([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3])

# Gini impurity calculation
def gini_impurity(classes):
    _, counts = np.unique(classes, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

# Compute Gini impurity for the root node
root_impurity = gini_impurity(class_labels)

# Compute Gini impurities for the split on f2
mask_f2_0 = data[:, 1] == 0
mask_f2_1 = data[:, 1] == 1

gini_f2_0 = gini_impurity(class_labels[mask_f2_0])
gini_f2_1 = gini_impurity(class_labels[mask_f2_1])

# Calculate the proportions for weighted average
prop_f2_0 = mask_f2_0.sum() / len(class_labels)
prop_f2_1 = mask_f2_1.sum() / len(class_labels)

# Weighted Gini impurity after the split
weighted_gini = prop_f2_0 * gini_f2_0 + prop_f2_1 * gini_f2_1

# Impurity gain from the split
gini_gain = root_impurity - weighted_gini

print(root_impurity, gini_f2_0, gini_f2_1, weighted_gini, gini_gain)
