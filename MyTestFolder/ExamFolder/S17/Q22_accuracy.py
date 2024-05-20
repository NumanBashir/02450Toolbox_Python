import numpy as np
from collections import Counter

# Distance matrix from the provided table
correct_distance_matrix = np.array([
    [0, 0.2606, 1.1873, 2.4946, 2.9510, 2.5682, 3.4535, 2.4698],
    [0.2606, 0, 1.2796, 2.4442, 2.8878, 2.4932, 3.3895, 2.4216],
    [1.1873, 1.2796, 0, 2.8294, 3.6892, 2.9147, 4.1733, 2.2386],
    [2.4946, 2.4442, 2.8294, 0, 1.4852, 0.2608, 2.2941, 1.8926],
    [2.9510, 2.8878, 3.6892, 1.4852, 0, 1.5155, 1.0296, 3.1040],
    [2.5682, 2.4932, 2.9147, 0.2608, 1.5155, 0, 2.3316, 1.8870],
    [3.4535, 3.3895, 4.1733, 2.2941, 1.0296, 2.3316, 0, 3.7588],
    [2.4698, 2.4216, 2.2386, 1.8926, 3.1040, 1.8870, 3.7588, 0]
])

# Class labels
class_labels = np.array(['C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C1'])

# Leave-one-out cross-validation
correct_predictions = 0
total_observations = len(class_labels)

for i in range(total_observations):
    # Leave one out
    distances = correct_distance_matrix[i, :]
    distances[i] = np.inf  # Ignore the distance to itself
    
    # Find the indices of the 3 nearest neighbors
    nearest_neighbors_indices = np.argsort(distances)[:1] # TODO CHANGE THIS TO THE K NUMBER OF NEIGHBORS
    
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
print(f"Correct Predictions: {correct_predictions}")
