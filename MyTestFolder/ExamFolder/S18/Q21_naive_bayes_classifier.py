import numpy as np

# Data matrix from the provided table
data = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Safe
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Unsafe
    [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],  # Safe
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],  # Safe
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Safe
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Unsafe
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],  # Unsafe
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],  # Unsafe
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],  # Unsafe
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]   # Safe
])

# TODO CHANGE THESE INDICES ACCORDING TO QUESTION
# Define safe and unsafe indices based on the description
safe_indices = [0, 2, 3, 4, 9]  # O1, O3, O4, O5, O10
unsafe_indices = [1, 5, 6, 7, 8]  # O2, O6, O7, O8, O9

# Features for x^H_2, x^H_3, x^H_4, x^H_5 are 3, 5, 7, 9
features_indices = [3, 5, 7, 9]

# Calculate P(Safe) and P(Unsafe)
p_safe = len(safe_indices) / len(data)
p_unsafe = len(unsafe_indices) / len(data)

# Calculate likelihoods for each feature given Safe and Unsafe
likelihoods_safe = [np.mean(data[safe_indices, idx] == 1) for idx in features_indices]
likelihoods_unsafe = [np.mean(data[unsafe_indices, idx] == 1) for idx in features_indices]

# Calculate combined likelihood for P(x^H_2=1, x^H_3=1, x^H_4=1, x^H_5=1 | Safe) and same for Unsafe
likelihood_safe = np.prod(likelihoods_safe)
likelihood_unsafe = np.prod(likelihoods_unsafe)

# Calculate evidence P(x^H_2=1, x^H_3=1, x^H_4=1, x^H_5=1)
evidence = likelihood_safe * p_safe + likelihood_unsafe * p_unsafe

# Calculate posterior probability P(Safe|x^H_2=1, x^H_3=1, x^H_4=1, x^H_5=1)
posterior_safe = (likelihood_safe * p_safe) / evidence

print(f"Probability of being safe given x^H_2=1, x^H_3=1, x^H_4=1, x^H_5=1: {posterior_safe:.3f} or as a fraction {posterior_safe:.3f}/{1:.3f}")