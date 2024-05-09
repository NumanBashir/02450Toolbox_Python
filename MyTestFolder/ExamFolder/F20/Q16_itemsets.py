import numpy as np
from itertools import combinations


# Binarized data matrix
binarized_data = np.array([
    [0, 0, 1, 0],  # o11
    [1, 0, 1, 0],  # o12
    [0, 1, 0, 0],  # o13
    [1, 0, 1, 1],  # o14
    [1, 0, 0, 0],  # o15
    [0, 0, 0, 1],  # o16
    [1, 0, 1, 0],  # o17
    [0, 1, 0, 0],  # o18
    [0, 0, 1, 0],  # o19
    [1, 0, 0, 0]   # o20
])


# Convert to numpy array for efficient computation
data_array = np.array(binarized_data)

# Calculate the support for each individual feature
supports = data_array.mean(axis=0)

# Find features with support greater than 0.25
features_above_threshold = np.nonzero(supports > 0.25)[0]

# Generate all non-empty combinations of features above the threshold
all_combinations = []
for i in range(1, len(features_above_threshold)+1):
    combinations_i = list(combinations(features_above_threshold, i))
    for combination in combinations_i:
        # Calculate support for each combination
        itemset_support = data_array[:, combination].all(axis=1).mean()
        if itemset_support > 0.25:
            # Adjust the index to match the feature numbering f1, f2, ..., f11
            adjusted_combination = [index + 1 for index in combination]
            all_combinations.append(adjusted_combination)

# Convert result to a format that matches the question options: {f1, f2, ...}
formatted_combinations = [{'f' + str(index) for index in combo} for combo in all_combinations]

print(formatted_combinations)
