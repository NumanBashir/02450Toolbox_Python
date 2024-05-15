import numpy as np
from collections import Counter

# Distance matrix from the provided table
distance_matrix = np.array([
    [0.00, 4.84, 0.50, 4.11, 1.07, 4.10, 4.71, 4.70, 4.93],
    [4.84, 0.00, 4.00, 5.96, 4.12, 2.01, 5.36, 3.59, 3.02],
    [0.50, 4.00, 0.00, 4.07, 0.72, 3.75, 4.66, 4.48, 4.64],
    [4.11, 5.96, 4.07, 0.00, 4.48, 4.69, 2.44, 3.68, 4.15],
    [1.07, 4.12, 0.72, 4.48, 0.00, 3.54, 4.96, 4.62, 4.71],
    [4.10, 2.01, 3.75, 4.69, 3.54, 0.00, 3.72, 2.23, 1.95],
    [4.71, 5.36, 4.66, 2.44, 4.96, 3.72, 0.00, 2.03, 2.73],
    [4.70, 3.59, 4.48, 3.68, 4.62, 2.23, 2.03, 0.00, 0.73],
    [4.93, 3.02, 4.64, 4.15, 4.71, 1.95, 2.73, 0.73, 0.00]
])
# Ændre i tabellen 

# True classes
true_classes = np.array(['C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2'])

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
error_rate, misclassified_count = leave_one_out_knn(distance_matrix, true_classes, k)

# Output the misclassified count and the error rate
print(f"Number of misclassified observations: {misclassified_count}")
print(f"Error rate: {error_rate:.2f} or {misclassified_count}/{len(true_classes)}")
