import numpy as np # type: ignore

def compute_metrics(confusion_matrix):
    # Calculate the number of correct predictions (sum of diagonal elements)
    correct_predictions = np.trace(confusion_matrix)

    # Calculate the total number of predictions (sum of all elements)
    total_predictions = np.sum(confusion_matrix)

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions

    # Initialize arrays to store precision and recall for each class
    precision = np.zeros(confusion_matrix.shape[0])
    recall = np.zeros(confusion_matrix.shape[0])

    # Calculate precision and recall for each class
    for i in range(confusion_matrix.shape[0]):
        precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]) if np.sum(confusion_matrix[:, i]) != 0 else 0
        recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]) if np.sum(confusion_matrix[i, :]) != 0 else 0

    return accuracy, precision, recall

# Example of a 3x3 confusion matrix
confusion_matrix_3x3 = np.array([
    [36, 15],  # Predictions for Class 1
    [10, 39],  # Predictions for Class 1
])

# Calculate metrics
accuracy, precision, recall = compute_metrics(confusion_matrix_3x3)
error_rate = 1 - accuracy

# Print the results
print(f"The accuracy of the classifier is: {accuracy:.4f}")
print("Precision for each class:", precision)
print("Recall for each class:", recall)
print("Error rate:", error_rate)
