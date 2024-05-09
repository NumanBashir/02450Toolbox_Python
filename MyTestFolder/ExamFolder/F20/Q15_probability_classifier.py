import numpy as np

# Binarized data matrix
binarized_data = np.array([
    [0, 0, 1, 0, 1],  # o11
    [1, 0, 1, 0, 1],  # o12
    [0, 1, 0, 0, 1],  # o13
    [1, 0, 1, 1, 1],  # o14
    [1, 0, 0, 0, 1],  # o15
    [0, 0, 0, 1, 1],  # o16
    [1, 0, 1, 0, 2],  # o17
    [0, 1, 0, 0, 2],  # o18
    [0, 0, 1, 0, 2],  # o19
    [1, 0, 0, 0, 2]   # o20
])

# Calculate the prior probabilities p(x5 = 1) and p(x5 = 2)
p_x5_1 = np.sum(binarized_data[:, 4] == 1) / len(binarized_data)
p_x5_2 = np.sum(binarized_data[:, 4] == 2) / len(binarized_data)

# Conditional probabilities for x5 = 1
subset1 = binarized_data[binarized_data[:, 4] == 1]
p_f1_0_given_x5_1 = np.sum(subset1[:, 0] == 0) / subset1.shape[0]
p_f2_0_f3_1_given_x5_1 = np.sum((subset1[:, 1] == 0) & (subset1[:, 2] == 1)) / subset1.shape[0]
p_f4_0_given_x5_1 = np.sum(subset1[:, 3] == 0) / subset1.shape[0]

# Conditional probabilities for x5 = 2
subset2 = binarized_data[binarized_data[:, 4] == 2]
p_f1_0_given_x5_2 = np.sum(subset2[:, 0] == 0) / subset2.shape[0]
p_f2_0_f3_1_given_x5_2 = np.sum((subset2[:, 1] == 0) & (subset2[:, 2] == 1)) / subset2.shape[0]
p_f4_0_given_x5_2 = np.sum(subset2[:, 3] == 0) / subset2.shape[0]

# Calculate the classifier for p(x5 = 1 | f1 = 0, f2 = 0, f3 = 1, f4 = 0)
numerator = p_f1_0_given_x5_1 * p_f2_0_f3_1_given_x5_1 * p_f4_0_given_x5_1 * p_x5_1
denominator = (p_f1_0_given_x5_1 * p_f2_0_f3_1_given_x5_1 * p_f4_0_given_x5_1 * p_x5_1 +
               p_f1_0_given_x5_2 * p_f2_0_f3_1_given_x5_2 * p_f4_0_given_x5_2 * p_x5_2)

probability_x5_1 = numerator / denominator
print(f'p(x5 = 1|f1 = 0, f2 = 0, f3 = 1, f4 = 0) = {probability_x5_1}')
