import numpy as np

# Probabilities of pollution levels
p_y1 = 391 / 981
p_y2 = 241 / 981
p_y3 = 349 / 981

# Probabilities of high O3 given pollution levels
p_x1_given_y1 = 64 / 391
p_x1_given_y2 = 66 / 241
p_x1_given_y3 = 206 / 349

# Convert to probabilities of low O3 given pollution levels
p_x0_given_y1 = 1 - p_x1_given_y1
p_x0_given_y2 = 1 - p_x1_given_y2
p_x0_given_y3 = 1 - p_x1_given_y3

# Bayes' Theorem to find p(y = 2 | x = 0)
numerator = p_x0_given_y2 * p_y2
denominator = (p_x0_given_y1 * p_y1) + (p_x0_given_y2 * p_y2) + (p_x0_given_y3 * p_y3)
p_y2_given_x0 = numerator / denominator

print("Probability of medium pollution level given low O3 concentration (p(y = 2 | x = 0)):", p_y2_given_x0)
