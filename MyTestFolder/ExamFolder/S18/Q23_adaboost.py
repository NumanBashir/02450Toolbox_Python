import numpy as np

# Weights of each observation in each round
weights = np.array([
    [0.1000, 0.0714, 0.0469, 0.0319],  # O1
    [0.1000, 0.0714, 0.0469, 0.0319],  # O2
    [0.1000, 0.1667, 0.1094, 0.2059],  # O3
    [0.1000, 0.0714, 0.0469, 0.0319],  # O4
    [0.1000, 0.1667, 0.1094, 0.2059],  # O5
    [0.1000, 0.0714, 0.0469, 0.0882],  # O6
    [0.1000, 0.0714, 0.0469, 0.0319],  # O7
    [0.1000, 0.1667, 0.3500, 0.2383],  # O8
    [0.1000, 0.0714, 0.1500, 0.1021],  # O9
    [0.1000, 0.0714, 0.0469, 0.0319],  # O10
])

# Observation classifications for each round (1 for correctly classified, -1 for incorrectly)
classifications = np.array([
    [-1, -1,  1, -1],  # O1
    [ 1, -1,  1,  1],  # O2
    [-1, -1, -1, -1],  # O3
    [ 1,  1,  1,  1],  # O4
    [-1, -1,  1,  1],  # O5
    [ 1,  1,  1,  1],  # O6
    [ 1,  1,  1,  1],  # O7
    [-1, -1, -1, -1],  # O8
    [ 1, -1,  1,  1],  # O9
    [ 1,  1,  1,  1],  # O10
])

# Calculate error rates for each round
error_rates = np.sum(weights * (classifications == -1), axis=0)

# Calculate alpha values for each round
alphas = 0.5 * np.log((1 - error_rates) / error_rates)

# Calculate total voting strength for each observation
total_votes = np.dot(classifications, alphas)

# Determine final classification based on the sign of the total vote
final_classifications = ['safe' if vote > 0 else 'unsafe' for vote in total_votes]

# Print the results
observations = [f'O{i+1}' for i in range(10)]
for obs, classification in zip(observations, final_classifications):
    print(f"Observation {obs} is classified as {classification}")
