import numpy as np

binarized_data = np.array([
    [0, 0, 0, 1, 0, 0, 0, 1],  # o1
    [0, 0, 1, 0, 0, 1, 0, 1],  # o2
    [0, 0, 1, 0, 0, 1, 0, 1],  # o3
    [0, 1, 0, 0, 0, 1, 0, 1],  # o4
    [0, 0, 0, 0, 0, 1, 0, 1],  # o5
    [0, 0, 1, 0, 1, 1, 0, 1],  # o6
    [0, 0, 1, 0, 0, 1, 0, 1],  # o7
    [1, 1, 0, 0, 0, 0, 1, 1],  # o8
    [0, 1, 0, 0, 0, 0, 0, 1],  # o9
    [0, 1, 0, 0, 0, 1, 0, 1],  # o10
    [1, 1, 0, 0, 0, 0, 0, 0],  # o11
])

# Given class probabilities at the root node
p_C1 = 2 / 11  # 2 out of 11 for C1
p_C2 = 5 / 11  # 5 out of 11 for C2
p_C3 = 4 / 11  # 4 out of 11 for C3

# Gini impurity at the root
I_r = 1 - (p_C1**2 + p_C2**2 + p_C3**2)

# Class probabilities in each group after split on f2
# Group 1 (f2 = 0)
p_C1_v1 = 6 / 6
p_C2_v1 = 0 / 6
p_C3_v1 = 0 / 6
I_v1 = 1 - (p_C1_v1**2 + p_C2_v1**2 + p_C3_v1**2)

# Group 2 (f2 = 1)
p_C1_v2 = 0 / 5
p_C2_v2 = 4 / 5
p_C3_v2 = 1 / 5
I_v2 = 1 - (p_C1_v2**2 + p_C2_v2**2 + p_C3_v2**2)

# Weighted Gini impurity after split
weighted_I = (6 / 11) * I_v1 + (5 / 11) * I_v2

# Gini impurity gain
delta_I = I_r - weighted_I

print("Initial Gini Impurity:", I_r)
print("Gini Group 1:", I_v1)
print("Gini Group 2:", I_v2)
print("Weighted Gini After Split:", weighted_I)
print("Gini Gain:", delta_I)
