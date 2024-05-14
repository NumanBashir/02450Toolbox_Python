# Given probabilities
P_A_given_B = 1  # Probability that an NBA player makes a very high salary
P_B = 2 / 1_000_000  # Probability of being an NBA player
P_A = 0.002  # Probability of making a very high salary

# Calculate P(B|A) using Bayes' Theorem
P_B_given_A = (P_A_given_B * P_B) / P_A

# Convert to percentage
P_B_given_A_percentage = P_B_given_A * 100

print(P_B_given_A_percentage)

