# Alpha values from each round
alpha = [-0.168, -0.325, -0.185, 0.207]

# Predictions for each test observation from each round
# y_test1 = [2, 1, 1, 2] corresponds to predictions [2, 1, 1, 2]
# y_test2 = [1, 2, 2, 1] corresponds to predictions [1, 2, 2, 1]

# Calculate the weighted sum of votes for y_test1
weighted_sum_test1_y1 = (alpha[1] + alpha[2])  # Rounds where y=1 for y_test1
weighted_sum_test1_y2 = (alpha[0] + alpha[3])  # Rounds where y=2 for y_test1

# Calculate the weighted sum of votes for y_test2
weighted_sum_test2_y1 = (alpha[0] + alpha[3])  # Rounds where y=1 for y_test2
weighted_sum_test2_y2 = (alpha[1] + alpha[2])  # Rounds where y=2 for y_test2

print((weighted_sum_test1_y1, weighted_sum_test1_y2), (weighted_sum_test2_y1, weighted_sum_test2_y2))

