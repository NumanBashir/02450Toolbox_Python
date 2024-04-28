# Given the binarized data and the sets for the antecedent and consequent,
# we will calculate the support for both and the confidence of the rule.

# Re-define the binarized data array using the provided data
import numpy as np


binarized_data = np.array([
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
])

# Define indices for the antecedent and consequent according to 0-based indexing
antecedent_indices = [0, 2, 3, 4, 7]  # Adjusting to 0-based indexing: f1, f3, f4, f5, f8
consequent_indices = [1, 9]  # Adjusting to 0-based indexing: f2, f10

# Calculate the support for the antecedent
support_antecedent = np.mean(binarized_data[:, antecedent_indices].all(axis=1))

# Calculate the support for the antecedent U consequent (Union)
support_union = np.mean(binarized_data[:, antecedent_indices + consequent_indices].all(axis=1))

# Calculate the confidence of the rule as support(A U B) / support(A)
confidence = support_union / support_antecedent if support_antecedent != 0 else 0

print(support_antecedent, support_union, confidence)
