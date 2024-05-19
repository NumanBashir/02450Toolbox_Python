import numpy as np
from scipy.stats import beta

# Function to calculate Beta distribution parameters and mean
def calculate_beta_params(n10, n01):
    alpha = n01 + 1
    beta_param = n10 + 1
    mean = alpha / (alpha + beta_param)
    return alpha, beta_param, mean

# Provided data for each option
options_data = {
    'A': {'M1_vs_M2': (15, 5), 'M1_vs_M3': (5, 28)},
    'B': {'M1_vs_M2': (16, 3), 'M1_vs_M3': (7, 24)},
    'C': {'M1_vs_M2': (18, 2), 'M1_vs_M3': (8, 25)},
    'D': {'M1_vs_M2': (33, 2), 'M1_vs_M3': (35, 11)},
}

# Calculate Beta parameters and means for each option
for option, tests in options_data.items():
    print(f"Option {option}:")
    for test, (n10, n01) in tests.items():
        alpha, beta_param, mean = calculate_beta_params(n10, n01)
        print(f"  {test}: alpha={alpha}, beta={beta_param}, mean={mean:.3f}")

# Expected means from the problem statement
expected_means = {
    'A': {'M1_vs_M2': 5/20, 'M1_vs_M3': 28/33},
    'B': {'M1_vs_M2': 3/19, 'M1_vs_M3': 24/31},
    'C': {'M1_vs_M2': 2/20, 'M1_vs_M3': 25/33},
    'D': {'M1_vs_M2': 2/36, 'M1_vs_M3': 11/46},
}

# Verify the means
print("\nVerification of means:")
for option, tests in expected_means.items():
    print(f"Option {option}:")
    for test, expected_mean in tests.items():
        alpha, beta_param, calculated_mean = calculate_beta_params(*options_data[option][test])
        print(f"  {test}: expected_mean={expected_mean:.3f}, calculated_mean={calculated_mean:.3f}")
