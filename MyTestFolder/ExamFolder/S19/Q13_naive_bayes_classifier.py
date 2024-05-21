import numpy as np

# Provided feature table
table4 = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # o1
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # o2
    [0, 1, 1, 1, 1, 1, 0, 0, 0],  # o3
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # o4
    [1, 0, 0, 1, 0, 0, 0, 0, 0],  # o5
    [0, 0, 1, 1, 0, 0, 0, 1, 0],  # o6
    [0, 0, 1, 1, 1, 0, 0, 0, 0],  # o7
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # o8
    [0, 1, 1, 0, 1, 0, 0, 0, 0],  # o9
    [0, 0, 1, 1, 1, 0, 1, 0, 0]   # o10
])

# Class labels
y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])

# Features for the query
query_features = np.array([0, 1, 0])

# Function to calculate conditional probabilities
def calculate_conditional_prob(feature_index, feature_value, class_value):
    class_indices = np.where(y == class_value)[0]
    class_data = table4[class_indices]
    return np.mean(class_data[:, feature_index] == feature_value)

# Classes
classes = np.unique(y)
n_classes = len(classes)

# Calculate prior probabilities P(y=j)
priors = np.array([np.sum(y == j) / len(y) for j in classes])

# Calculate conditional probabilities P(fk=v|y=j) for each feature
probabilities = np.zeros(n_classes)
for j in classes:
    conditional_probs = 1
    for k, feature_value in enumerate(query_features):
        conditional_probs *= calculate_conditional_prob([1, 3, 4][k], feature_value, j)
    probabilities[j] = conditional_probs * priors[j]

# Calculate total probability
total_prob = np.sum(probabilities)

# Normalize to get the posterior probabilities P(y=j|f1, f2, f3)
posterior_probs = probabilities / total_prob

# Probability of class y=2
print(f"P(y=2 | f2=0, f4=1, f5=0): {posterior_probs[2]:.4f}")
