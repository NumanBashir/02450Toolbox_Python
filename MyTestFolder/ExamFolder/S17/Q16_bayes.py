# Given probabilities
P_x4_0 = 0.5938
P_x1_8_given_x4_0 = 0.6316
P_x1_8_given_x4_1 = 0.1538

# Calculate P(x4 = 1)
P_x4_1 = 1 - P_x4_0

# Calculate P(x1 = 8)
P_x1_8 = (P_x1_8_given_x4_0 * P_x4_0) + (P_x1_8_given_x4_1 * P_x4_1)

# Calculate P(x4 = 0 | x1 = 8)
P_x4_0_given_x1_8 = (P_x1_8_given_x4_0 * P_x4_0) / P_x1_8

print(f"P(x4 = 0 | x1 = 8): {P_x4_0_given_x1_8 * 100:.1f}%")
