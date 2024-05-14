import numpy as np

# Provided feature table
data = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0],  # O1
    [1, 0, 1, 0, 1, 0, 1, 0],  # O2
    [1, 0, 1, 0, 1, 0, 1, 0],  # O3
    [1, 0, 1, 0, 1, 0, 0, 1],  # O4
    [1, 0, 1, 0, 0, 1, 0, 1],  # O5
    [1, 0, 0, 1, 0, 1, 1, 0],  # O6
    [0, 1, 1, 0, 0, 1, 0, 1],  # O7
    [0, 1, 1, 0, 1, 0, 0, 1],  # O8
    [0, 1, 0, 1, 1, 0, 1, 0],  # O9
    [0, 1, 0, 1, 0, 1, 1, 0]   # O10
])

# Scores: low=0, mid=1, high=2
scores = np.array([1, 0, 0, 1, 2, 0, 2, 2, 0, 2])

# Calculate prior probabilities
P_high = np.sum(scores == 2) / len(scores)
P_mid = np.sum(scores == 1) / len(scores)
P_low = np.sum(scores == 0) / len(scores)

# Calculate likelihoods for high score
P_HH_given_high = np.sum((data[:, 1] == 1) & (scores == 2)) / np.sum(scores == 2)
P_WL_given_high = np.sum((data[:, 2] == 1) & (scores == 2)) / np.sum(scores == 2)

# Calculate likelihoods for mid score
P_HH_given_mid = np.sum((data[:, 1] == 1) & (scores == 1)) / np.sum(scores == 1)
P_WL_given_mid = np.sum((data[:, 2] == 1) & (scores == 1)) / np.sum(scores == 1)

# Calculate likelihoods for low score
P_HH_given_low = np.sum((data[:, 1] == 1) & (scores == 0)) / np.sum(scores == 0)
P_WL_given_low = np.sum((data[:, 2] == 1) & (scores == 0)) / np.sum(scores == 0)

# Posterior probability for high score
numerator_high = P_HH_given_high * P_WL_given_high * P_high
denominator = (P_HH_given_low * P_WL_given_low * P_low +
               P_HH_given_mid * P_WL_given_mid * P_mid +
               numerator_high)

P_high_given_HH_WL = numerator_high / denominator

print(P_high_given_HH_WL)