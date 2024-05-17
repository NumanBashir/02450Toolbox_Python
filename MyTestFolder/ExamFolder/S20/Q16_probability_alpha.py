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

# Regularization factor
alpha = 1

# Count occurrences
occurrences_y1 = np.sum(y == 1)
occurrences_f2_1_f3_1_y1 = np.sum((table5[y == 1][:, f2_index] == 1) & (table5[y == 1][:, f3_index] == 1))

# Apply the regularization
probability = (occurrences_f2_1_f3_1_y1 + alpha) / (occurrences_y1 + 2 * alpha)

print("Occurrences of y_b = 1:", occurrences_y1)
print("Occurrences of f_2 = 1 and f_3 = 1 given y_b = 1:", occurrences_f2_1_f3_1_y1)
print("Probability p(f_2 = 1, f_3 = 1 | y_b = 1):", probability)
