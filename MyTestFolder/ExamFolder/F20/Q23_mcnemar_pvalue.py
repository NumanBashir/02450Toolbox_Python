# Import necessary library
from scipy.stats import binom

# Define the values of b and c
b = 28  # M1 correct, M2 incorrect
c = 35  # M2 correct, M1 incorrect

# Calculate the total number of trials
N = b + c

# Select the smaller of b and c for calculating the p-value using the CDF of the binomial distribution
m = min(b, c)

# Calculate the p-value using the binomial CDF
# Assuming θ = 1/2, since we expect the performances of the two classifiers to be equal under the null hypothesis
# The factor of 2 accounts for the two-tailed test
p_value = 2 * binom.cdf(m, N, 0.5)

# Print the p-value rounded to two decimals
print(f"The p-value from McNemar's test using the binomial distribution is approximately {p_value:.2f}")
