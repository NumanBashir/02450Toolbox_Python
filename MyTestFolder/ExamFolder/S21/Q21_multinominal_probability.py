import numpy as np

# Given linear transformations
w1_x1 = 0.03
w2_x1 = -1.89

# Compute the exponentials
exp_w1_x1 = np.exp(w1_x1)
exp_w2_x1 = np.exp(w2_x1)
exp_w3_x1 = np.exp(0)  # Implicitly zero for the baseline class

# Compute the sum of exponentials
sum_exp = exp_w1_x1 + exp_w2_x1 + exp_w3_x1

# Calculate the probability for class y = 1
p_y1 = exp_w1_x1 / sum_exp

print(f"Probability that observation i = 1 is assigned to the low demand class (y = 1): {p_y1:.2f}")
