import numpy as np

# Initial data and labels (binary features)
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

# Scores: low=0 (black), mid=1 (red), high=2 (blue)
scores = np.array([1, 0, 0, 1, 2, 0, 2, 2, 0, 2])

# Initialize weights
N = len(data)
weights = np.ones(N) / N

# Adaboost parameters
alpha = []

# Classifiers from the given decision boundaries (simulated here)
def classifier_round_1(x1, x2):
    return 1 if x1 < 6.7 else 0  # Simulate decision boundary for round 1

def classifier_round_2(x1, x2):
    return 1 if x1 < 6.3 else 0  # Simulate decision boundary for round 2

# Apply the first classifier
predictions_round_1 = np.array([classifier_round_1(x[0], x[1]) for x in data])
errors_round_1 = (predictions_round_1 != scores).astype(int)
epsilon_1 = np.sum(weights * errors_round_1) / np.sum(weights)
alpha_1 = 0.5 * np.log((1 - epsilon_1) / epsilon_1)

# Update weights (though alpha_1 = 0 will keep them unchanged)
weights *= np.exp(alpha_1 * errors_round_1)

# Apply the second classifier with updated weights
predictions_round_2 = np.array([classifier_round_2(x[0], x[1]) for x in data])
errors_round_2 = (predictions_round_2 != scores).astype(int)
epsilon_2 = np.sum(weights * errors_round_2) / np.sum(weights)
alpha_2 = 0.5 * np.log((1 - epsilon_2) / epsilon_2)

# Classify the new observation based on combined classifiers
new_observation = np.array([6, 240])

# First classifier's prediction
prediction_1 = classifier_round_1(new_observation[0], new_observation[1])
# Second classifier's prediction
prediction_2 = classifier_round_2(new_observation[0], new_observation[1])

# Weighted vote
vote = alpha_1 * prediction_1 + alpha_2 * prediction_2

# Final classification based on weighted vote
final_classification = 1 if vote > 0.5 * (alpha_1 + alpha_2) else 0
classified_color = "black plus" if final_classification == 0 else "red cross"

# Output the result
print(classified_color)
