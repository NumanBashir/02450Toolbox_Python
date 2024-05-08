import numpy as np

# Binarized data matrix
binarized_data = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # o1
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # o2
    [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],  # o3
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],  # o4
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # o5
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # o6
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],  # o7
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],  # o8
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],  # o9
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]   # o10
])

# Antecedent and consequent indices (corrected to the problem statement)
antecedent_indices = [3, 5, 7, 9] 
consequent_indices = [11]

# Calculate support of both antecedent and consequent
full_rule_support = np.all(binarized_data[:, antecedent_indices + consequent_indices], axis=1)
support_A_and_B = np.sum(full_rule_support)

# Calculate support as a percentage of total observations
support_percentage = support_A_and_B / len(binarized_data) * 100

print(f"Support for the rule {{x^H_2, x^H_3, x^H_4, x^H_5}} -> {{x^H_6}}: {support_percentage:.1f}%")
