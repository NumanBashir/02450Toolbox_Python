# Given probabilities
P_y1 = 0.4917
P_x1_1_y1 = 0.3220
P_x1_1_y0 = 0.1639
P_y0 = 1 - P_y1  # Complement of P(y=1)

# Calculate P(x1=1)
P_x1_1 = P_x1_1_y1 * P_y1 + P_x1_1_y0 * P_y0

# Calculate P(y=1 | x1=1) using Bayes' theorem
P_y1_given_x1_1 = (P_x1_1_y1 * P_y1) / P_x1_1

# Convert to percentage
P_y1_given_x1_1_percentage = P_y1_given_x1_1 * 100

# Print the result
print(f"P(y=1 | x1=1) = {P_y1_given_x1_1:.4f} or {P_y1_given_x1_1_percentage:.2f}%")

# Expected output: 0.6552 or 65.52%
