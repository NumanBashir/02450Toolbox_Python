# Conditional probabilities from Table 3 for (x̂1 = 1, x̂2 = 0)
p_x1_1_x2_0_given_y1 = 0.00  # y = 1
p_x1_1_x2_0_given_y2 = 0.85  # y = 2
p_x1_1_x2_0_given_y3 = 0.16  # y = 3

# Prior probabilities
p_y1 = 0.44  # p(y = 1)
p_y2 = 0.36  # p(y = 2)
p_y3 = 0.20  # p(y = 3)

# Marginal probability of (x̂1 = 1, x̂2 = 0)
p_x1_1_x2_0 = (p_x1_1_x2_0_given_y1 * p_y1 +
               p_x1_1_x2_0_given_y2 * p_y2 +
               p_x1_1_x2_0_given_y3 * p_y3)

# Posterior probability of y = 3 given (x̂1 = 1, x̂2 = 0)
p_y3_given_x1_1_x2_0 = (p_x1_1_x2_0_given_y3 * p_y3) / p_x1_1_x2_0

print(p_y3_given_x1_1_x2_0)
