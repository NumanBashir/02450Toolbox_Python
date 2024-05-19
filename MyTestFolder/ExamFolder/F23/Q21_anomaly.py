import numpy as np # type: ignore

# Training observations
#TODO: update
X = np.array([-3, -1, 5, 6])

# Test observations
#TODO: update
test_observations = np.array([-4, 2])

# Standard deviation
σ = np.sqrt(0.5)

# Gaussian kernel function
def K(x):
    return (1 / np.sqrt(2 * np.pi * σ**2)) * np.exp(-x**2 / (2 * σ**2))

# Calculate KDE for each test observation
for x_test in test_observations:
    KDE = np.mean(K(x_test - X))
    print(f"KDE for x = {x_test}: {KDE}")
    
    # Check if the KDE is less than 0.015
    #TODO: update
    if KDE < 0.015:
        print(f"The observation at x = {x_test} is an anomaly.")
    else:
        print(f"The observation at x = {x_test} is not an anomaly.")