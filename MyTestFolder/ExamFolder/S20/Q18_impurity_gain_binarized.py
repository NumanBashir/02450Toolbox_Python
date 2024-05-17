import numpy as np

# Table 5: Poverty dataset
table5 = np.array([
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

# classes (color coded as 0: red, 1: black)
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

# Calculate Gini impurity
def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

# Initial Gini impurity
I_r = gini_impurity(y)

# Split the dataset based on the feature x1
left_split = y[table5[:, 0] == 0]
right_split = y[table5[:, 0] == 1]

# Gini impurity for each split
I_v_left = gini_impurity(left_split)
I_v_right = gini_impurity(right_split)

# Number of samples in each split
N = len(y)
N_left = len(left_split)
N_right = len(right_split)

# Weighted Gini impurity after the split
I_split = (N_left / N) * I_v_left + (N_right / N) * I_v_right

# Purity gain Δ
delta = I_r - I_split

# Print results
print(f"Initial Gini impurity: {I_r:.3f}")
print(f"Gini impurity left split: {I_v_left:.3f}")
print(f"Gini impurity right split: {I_v_right:.3f}")
print(f"Weighted Gini impurity after split: {I_split:.3f}")
print(f"Purity gain Δ: {delta:.3f}")
