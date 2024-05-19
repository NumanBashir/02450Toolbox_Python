import numpy as np
from collections import Counter

# Distance matrix from the provided table
correct_distance_matrix = np.array([
    [0.0, 2.6, 2.8, 2.1, 0.8, 1.7, 3.9, 3.8, 4.1, 4.2],
    [2.6, 0.0, 1.8, 2.3, 2.7, 1.4, 1.9, 1.7, 2.0, 2.2],
    [2.8, 1.8, 0.0, 1.2, 2.9, 2.1, 2.2, 1.6, 2.3, 2.0],
    [2.1, 2.3, 1.2, 0.0, 2.4, 1.8, 2.6, 2.5, 3.0, 2.9],
    [0.8, 2.7, 2.9, 2.4, 0.0, 2.0, 4.2, 4.0, 4.4, 4.5],
    [1.7, 1.4, 2.1, 1.8, 2.0, 0.0, 2.4, 2.6, 2.7, 3.0],
    [3.9, 1.9, 2.2, 2.6, 4.2, 2.4, 0.0, 1.2, 0.6, 1.3],
    [3.8, 1.7, 1.6, 2.5, 4.0, 2.6, 1.2, 0.0, 1.0, 0.6],
    [4.1, 2.0, 2.3, 3.0, 4.4, 2.7, 0.6, 1.0, 0.0, 0.9],
    [4.2, 2.2, 2.0, 2.9, 4.5, 3.0, 1.3, 0.6, 0.9, 0.0]
])

# Class labels
class_labels = np.array(['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2'])

# Leave-one-out cross-validation
correct_predictions = 0
total_observations = len(class_labels)

for i in range(total_observations):
    # Leave one out
    distances = correct_distance_matrix[i, :]
    distances[i] = np.inf  # Ignore the distance to itself
    
    # Find the indices of the 3 nearest neighbors
    nearest_neighbors_indices = np.argsort(distances)[:3]
    
    # Get the class labels of the nearest neighbors
    nearest_neighbors_labels = class_labels[nearest_neighbors_indices]
    
    # Predict the class based on the majority class of the neighbors
    predicted_class = Counter(nearest_neighbors_labels).most_common(1)[0][0]
    
    # Compare the predicted class with the true class
    if predicted_class == class_labels[i]:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_observations

# Print results
print(f"Accuracy: {accuracy:.2f}")

# Determine which accuracy value corresponds to the given choices
if accuracy == 1/5:
    print("Accuracy matches option A: accuracy = 1/5")
elif accuracy == 3/5:
    print("Accuracy matches option B: accuracy = 3/5")
elif accuracy == 4/5:
    print("Accuracy matches option C: accuracy = 4/5")
elif accuracy == 1:
    print("Accuracy matches option D: accuracy = 1")
else:
    print("Accuracy does not match any of the provided options")
