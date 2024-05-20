import numpy as np
from collections import Counter

# Distance matrix from the provided table
correct_distance_matrix = np.array([
    [0.0, 2.0, 5.7, 0.9, 2.9, 1.8, 2.7, 3.7, 5.3, 5.1],
    [2.0, 0.0, 5.6, 2.4, 2.5, 3.0, 3.5, 4.3, 6.0, 6.2],
    [5.7, 5.6, 0.0, 5.0, 5.1, 4.0, 3.3, 5.4, 1.2, 1.8],
    [0.9, 2.4, 5.0, 0.0, 2.7, 2.1, 2.2, 3.5, 4.6, 4.4],
    [2.9, 2.5, 5.1, 2.7, 0.0, 3.5, 3.7, 4.0, 5.8, 5.7],
    [1.8, 3.0, 4.0, 2.1, 3.5, 0.0, 1.7, 5.3, 3.8, 3.7],
    [2.7, 3.5, 3.3, 2.2, 3.7, 1.7, 0.0, 4.2, 3.1, 3.2],
    [3.7, 4.3, 5.4, 3.5, 4.0, 5.3, 4.2, 0.0, 5.5, 6.0],
    [5.3, 6.0, 1.2, 4.6, 5.8, 3.8, 3.1, 5.5, 0.0, 2.1],
    [5.1, 6.2, 1.8, 4.4, 5.7, 3.7, 3.2, 6.0, 2.1, 0.0]
])

# Class labels
class_labels = np.array(['C1', 'C1', 'C2', 'C2', 'C2', 'C3', 'C3', 'C3', 'C3', 'C3'])

# Leave-one-out cross-validation
correct_predictions = 0
total_observations = len(class_labels)

for i in range(total_observations):
    # Leave one out
    distances = correct_distance_matrix[i, :]
    distances[i] = np.inf  # Ignore the distance to itself
    
    # Find the indices of the 3 nearest neighbors
    nearest_neighbors_indices = np.argsort(distances)[:3] # TODO CHANGE THIS TO THE K NUMBER OF NEIGHBORS
    
    # Get the class labels of the nearest neighbors
    nearest_neighbors_labels = class_labels[nearest_neighbors_indices]
    
    # Predict the class based on the majority class of the neighbors
    predicted_class = Counter(nearest_neighbors_labels).most_common(1)[0][0]
    
    # Compare the predicted class with the true class
    if predicted_class == class_labels[i]:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_observations
error_rate = 1 - accuracy

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Error Rate: {error_rate:.2f}")
