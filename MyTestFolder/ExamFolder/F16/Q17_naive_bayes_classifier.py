import numpy as np

# Data from Table 5
table5 = np.array([
    [1, 1, 1, 1, 0, 1],  
    [0, 0, 0, 0, 0, 0],  
    [1, 1, 0, 1, 0, 0],  
    [0, 1, 1, 0, 1, 0],  
    [1, 1, 1, 1, 1, 1],  
    [0, 0, 0, 0, 0, 0],  
    [1, 1, 0, 1, 0, 0],  
    [0, 1, 1, 0, 1, 0],  
    [1, 1, 1, 1, 0, 1],  
    [0, 1, 1, 0, 1, 0],   
    [0, 0, 0, 0, 0, 0],   
    [1, 1, 0, 1, 0, 0],   
    [0, 1, 1, 0, 1, 0],   
    [0, 1, 1, 0, 1, 0],   
])

# Extract x1, x2, and y
x1 = table5[:, 0]
x2 = table5[:, 1]
y = table5[:, 5]

# Number of instances
N = len(y)

# Prior probabilities
P_y1 = np.sum(y) / N
P_y0 = 1 - P_y1

# Likelihoods P(x1=1|y), P(x2=1|y)
P_x1_1_y1 = np.sum((x1 == 1) & (y == 1)) / np.sum(y == 1)
P_x2_1_y1 = np.sum((x2 == 1) & (y == 1)) / np.sum(y == 1)
P_x1_1_y0 = np.sum((x1 == 1) & (y == 0)) / np.sum(y == 0)
P_x2_1_y0 = np.sum((x2 == 1) & (y == 0)) / np.sum(y == 0)

# Calculate the numerator and denominator for P(y=1|x1=1, x2=1)
numerator = P_y1 * P_x1_1_y1 * P_x2_1_y1
denominator = numerator + (P_y0 * P_x1_1_y0 * P_x2_1_y0)

# Posterior probability P(y=1|x1=1, x2=1)
P_y1_given_x1_1_x2_1 = numerator / denominator

# Print the result
print(f"P(y=1|x1=1, x2=1) = {P_y1_given_x1_1_x2_1:.4f}")
