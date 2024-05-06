import numpy as np

# Provided feature table
table4 = np.array([
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

# Assuming alternate distribution of y=1 and y=0
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# Calculate probabilities
features = [0, 1, 5]  # Indices for f1, f2, f6
values = [1, 1, 0]    # Corresponding values for f1=1, f2=1, f6=0

# Filtering data based on y=1 and y=0
data_y1 = table4[y == 1]
data_y0 = table4[y == 0]

# Conditional probabilities P(f_k = v | y = 1)
p_f1_y1 = np.mean(data_y1[:, features[0]] == values[0])
p_f2_y1 = np.mean(data_y1[:, features[1]] == values[1])
p_f6_y1 = np.mean(data_y1[:, features[5]] == values[5])

# Prior P(y=1)
p_y1 = np.mean(y == 1)

# Total probability P(f1=1, f2=1, f6=0)
p_f1_f2_f6 = np.mean((table4[:, features[0]] == values[0]) & 
                     (table4[:, features[1]] == values[1]) & 
                     (table4[:, features[5]] == values[5]))

# Applying Bayes' theorem
p_y1_given_f1_f2_f6 = (p_f1_y1 * p_f2_y1 * p_f6_y1 * p_y1) / p_f1_f2_f6

print("P(f1=1 | y=1):", p_f1_y1)
print("P(f2=1 | y=1):", p_f2_y1)
print("P(f6=0 | y=1):", p_f6_y1)
print("P(y=1):", p_y1)
print("P(f1=1, f2=1, f6=0):", p_f1_f2_f6)
print("P(y=1 | f1=1, f2=1, f6=0):", p_y1_given_f1_f2_f6)
