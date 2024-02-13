import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# Probability of success
p = 0.5

# Generate random samples from a Bernoulli distribution
samples = bernoulli.rvs(p, size=1000)

# Calculate probabilities of success (1) and failure (0)
prob_success = bernoulli.pmf(1, p)
prob_failure = bernoulli.pmf(0, p)

print(f"Probability of Success: {prob_success}")
print(f"Probability of Failure: {prob_failure}")

# Plot the distribution of samples
plt.hist(samples, bins=[-0.5, 0.5, 1.5], weights=np.ones(len(samples)) / len(samples), edgecolor='black')
plt.xticks([0, 1], ['Failure (0)', 'Success (1)'])
plt.ylabel('Probability')
plt.title('Bernoulli Distribution of Samples')
plt.show()
