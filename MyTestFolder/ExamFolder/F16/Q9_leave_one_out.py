import numpy as np
from collections import Counter

# Distance matrix from the provided table
correct_distance_matrix = np.array([
    [0, 0.534, 1.257, 1.671, 1.090, 1.315, 1.484, 1.253, 1.418],
    [0.534, 0, 0.727, 2.119, 1.526, 1.689, 1.214, 0.997, 1.056],
    [1.257, 0.727, 0, 2.809, 2.220, 2.342, 1.088, 0.965, 0.807],
    [1.671, 2.119, 2.809, 0, 0.601, 0.540, 3.135, 2.908, 3.087],
    [1.090, 1.526, 2.220, 0.601, 0, 0.331, 2.563, 2.338, 2.500],
    [1.315, 1.689, 2.342, 0.540, 0.331, 0, 2.797, 2.567, 2.708],
    [1.484, 1.214, 1.088, 3.135, 2.563, 2.797, 0, 0.275, 0.298],
    [1.253, 0.997, 0.965, 2.908, 2.338, 2.567, 0.275, 0, 0.343],
    [1.418, 1.056, 0.807, 3.087, 2.500, 2.708, 0.298, 0.343, 0]
])
# Ændre i tabellen 

# True classes
true_classes = np.array(['C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C3', 'C3', 'C3'])

# Number of neighbors
k = 2

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
