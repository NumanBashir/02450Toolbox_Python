import numpy as np
from fractions import Fraction

# Binarized data matrix
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
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 0]   # o10
])

# Antecedent and consequent indices (zero-based)
antecedent_indices = [0, 2, 7, 8]  # f1, f3, f8, f9
consequent_indices = [1, 5, 6]    # f2, f6, f7

# Calculate support of the antecedent
antecedent_support = np.all(binarized_data[:, antecedent_indices], axis=1)
support_A = np.sum(antecedent_support)

# Calculate support of both antecedent and consequent
full_rule_support = np.all(binarized_data[:, antecedent_indices + consequent_indices], axis=1)
support_A_and_B = np.sum(full_rule_support)

# Calculate confidence
confidence = support_A_and_B / support_A if support_A else 0
confidence_fraction = Fraction(confidence).limit_denominator()  # Converts to the nearest simplified fraction

print(f"Confidence of the rule {{f1, f3, f8, f9}} -> {{f2, f6, f7}}: {confidence:.3f} or as a fraction {confidence_fraction}")
