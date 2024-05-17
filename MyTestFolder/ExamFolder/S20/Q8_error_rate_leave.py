import numpy as np
from collections import Counter

# Distance matrix from the provided table
correct_distance_matrix = np.array([
    [0.0, 1.7, 1.4, 0.4, 2.2, 3.7, 5.2, 0.2, 4.3, 6.8, 6.0],
    [1.7, 0.0, 1.0, 2.0, 1.3, 2.6, 4.5, 1.8, 3.2, 5.9, 5.2],
    [1.4, 1.0, 0.0, 1.7, 0.9, 2.4, 4.1, 1.5, 3.0, 5.5, 4.8],
    [0.4, 2.0, 1.7, 0.0, 2.6, 4.0, 5.5, 0.3, 4.6, 7.1, 6.3],
    [2.2, 1.3, 0.9, 2.6, 0.0, 1.7, 3.4, 2.4, 2.1, 4.8, 4.1],
    [3.7, 2.6, 2.4, 4.0, 1.7, 0.0, 2.0, 3.8, 1.6, 3.3, 2.7],
    [5.2, 4.5, 4.1, 5.5, 3.4, 2.0, 0.0, 5.4, 2.5, 1.6, 0.9],
    [0.2, 1.8, 1.5, 0.3, 2.4, 3.8, 5.4, 0.0, 4.4, 6.9, 6.1],
    [4.3, 3.2, 3.0, 4.6, 2.1, 1.6, 2.5, 4.4, 0.0, 3.4, 2.9],
    [6.8, 5.9, 5.5, 7.1, 4.8, 3.3, 1.6, 6.9, 3.4, 0.0, 1.0],
    [6.0, 5.2, 4.8, 6.3, 4.1, 2.7, 0.9, 6.1, 2.9, 1.0, 0.0]
])
# Ændre i tabellen 

# True classes
true_classes = np.array(['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2'])

# Number of neighbors
k = 3

#ændre i K værdi

# Function to perform leave-one-out cross-validation
def leave_one_out_knn(distance_matrix, true_classes, k):
    n = len(true_classes)
    misclassified_count = 0

    for i in range(n):
        # Exclude the current observation
        distances = np.delete(distance_matrix[i], i)
        classes = np.delete(true_classes, i)

        # Get the indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        nearest_classes = classes[nearest_indices]

        # Predict the class based on the majority vote
        predicted_class = Counter(nearest_classes).most_common(1)[0][0]

        # Check if the prediction is correct
        if predicted_class != true_classes[i]:
            misclassified_count += 1

    # Calculate the error rate
    error_rate = misclassified_count / n
    return error_rate, misclassified_count

# Calculate the error rate
error_rate, misclassified_count = leave_one_out_knn(correct_distance_matrix, true_classes, k)

# Output the misclassified count and the error rate
print(f"Number of misclassified observations: {misclassified_count}")
print(f"Error rate: {error_rate:.2f} or {misclassified_count}/{len(true_classes)}")
