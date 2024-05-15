import numpy as np

# Given values
sigma = 0.168
p_xi = 0.26 # TODO LOOK AT THE PEAK DENSITY LEVEL ON THE FIGURE

# Calculate N
N = 1 / (p_xi * np.sqrt(2 * np.pi * sigma**2))

print(f"The number of observations (N) is approximately: {N:.0f}")
