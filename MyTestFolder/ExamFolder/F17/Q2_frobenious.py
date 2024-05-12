import numpy as np

# Given singular values
sigma_1 = 11.1
sigma_2 = 7.2
sigma_3 = 5.2

# Frobenius norm squared
frobenius_norm_squared = 212

# Calculate the sum of the squares of the given singular values
sum_known_squares = sigma_1**2 + sigma_2**2 + sigma_3**2

# Calculate the square of the fourth singular value
sigma_4_squared = frobenius_norm_squared - sum_known_squares

# Calculate the fourth singular value
sigma_4 = np.sqrt(sigma_4_squared)

print("value of σ4 is:",sigma_4)