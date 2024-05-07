import numpy as np
from scipy.stats import t

# Given losses
losses = np.array([1, 3, 3, 1, 2, 3, 1])

# Degrees of freedom: N - 1
nu = len(losses) - 1

# Calculate the mean of the losses
mu = np.mean(losses)

# Calculate the standard deviation (unbiased estimator, hence ddof=1)
sigma = np.sqrt(np.var(losses))

# Calculate expected sigma for the answer confirmation
expected_sigma = 1 / np.sqrt(7)

print(f"Degrees of freedom (ν): {nu}")
print(f"Mean (μ): {mu}")
print(f"Standard deviation (σ): {sigma}")
print(f"Expected σ (from answer choice): {expected_sigma}")
print(np.sqrt(1/7))

# Validate if the calculated σ matches the expected from the answer
if np.isclose(sigma, expected_sigma):
    print("Calculated σ matches the expected σ from the answer choice.")
else:
    print("There is a discrepancy in the calculated σ and the expected σ from the answer.")
