# Let's write a Python script to compute the support for all non-empty itemsets
# and find those with support greater than 0.65 based on the provided data.

# Binarized dataset represented as a list of lists
# Each sublist represents a transaction and contains binary values indicating the presence (1) or absence (0) of items
import numpy as np
from itertools import combinations


# binarized_data = [
#     [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
#     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
#     [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
#     [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
#     [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
#     [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
#     [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
#     [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
# ]

binarized_data = np.array([
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # o1
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # o2
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # o3
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],  # o4
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # o5
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],  # o6
    [1, 1, 1, 0, 0, 1, 1, 1, 1, 0],  # o7
    [0, 1, 1, 1, 0, 1, 1, 0, 0, 1],  # o8
    [0, 0, 0, 0, 1, 1, 1, 0, 1, 1],  # o9
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 0]  # o10
])


# Convert to numpy array for efficient computation
data_array = np.array(binarized_data)

# Calculate the support for each individual feature
supports = data_array.mean(axis=0)

# Find features with support greater than 0.65
features_above_threshold = np.nonzero(supports > 0.55)[0]

# Generate all non-empty combinations of features above the threshold
all_combinations = []
for i in range(1, len(features_above_threshold)+1):
    combinations_i = list(combinations(features_above_threshold, i))
    for combination in combinations_i:
        # Calculate support for each combination
        itemset_support = data_array[:, combination].all(axis=1).mean()
        if itemset_support > 0.55:
            # Adjust the index to match the feature numbering f1, f2, ..., f11
            adjusted_combination = [index + 1 for index in combination]
            all_combinations.append(adjusted_combination)

# Convert result to a format that matches the question options: {f1, f2, ...}
formatted_combinations = [{'f' + str(index) for index in combo} for combo in all_combinations]

print(formatted_combinations)
