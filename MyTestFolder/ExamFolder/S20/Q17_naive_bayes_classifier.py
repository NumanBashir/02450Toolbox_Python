import numpy as np

# Provided feature table
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

# Corresponding class labels
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

# Indices for f2 and f3
f2_index = 1
f3_index = 2

# Calculate the prior probabilities
p_y1 = np.mean(y == 1)
p_y0 = np.mean(y == 0)

# Calculate the conditional probabilities for y = 1
p_f2_1_y1 = np.mean(table5[y == 1][:, f2_index] == 1)
p_f3_0_y1 = np.mean(table5[y == 1][:, f3_index] == 0)

# Calculate the conditional probabilities for y = 0
p_f2_1_y0 = np.mean(table5[y == 0][:, f2_index] == 1)
p_f3_0_y0 = np.mean(table5[y == 0][:, f3_index] == 0)

# Apply Bayes' theorem
numerator = p_f2_1_y1 * p_f3_0_y1 * p_y1
denominator = numerator + (p_f2_1_y0 * p_f3_0_y0 * p_y0)

p_y1_given_f2_1_f3_0 = numerator / denominator

print("P(y_b = 1):", p_y1)
print("P(f2 = 1 | y_b = 1):", p_f2_1_y1)
print("P(f3 = 0 | y_b = 1):", p_f3_0_y1)
print("P(f2 = 1 | y_b = 0):", p_f2_1_y0)
print("P(f3 = 0 | y_b = 0):", p_f3_0_y0)
print("P(y_b = 1 | f2 = 1, f3 = 0):", p_y1_given_f2_1_f3_0)
