import numpy as np

# Binarized dataset (Table 8)
data = np.array([
    [1, 1, 0, 1, 0, 0, 'C1'],
    [1, 0, 1, 1, 0, 0, 'C1'],
    [0, 1, 1, 1, 1, 1, 'C1'],
    [0, 1, 1, 1, 1, 1, 'C1'],
    [1, 1, 0, 1, 1, 0, 'C2'],
    [0, 1, 1, 0, 1, 1, 'C2'],
    [0, 1, 1, 1, 0, 1, 'C2'],
    [0, 1, 0, 1, 1, 1, 'C2'],
    [0, 1, 0, 1, 1, 1, 'C2'],
    [1, 1, 1, 1, 1, 0, 'C2']
])

# Classifier f1: f1(b1, b2, b3, b4, b5, b6) = C1 if b3 = 1 and b4 = 1, otherwise C2
#TODO update based on what the task says 
def classifier_f1(b):
    b3, b4 = b[2], b[3]
    if b3 == 1 and b4 == 1:
        return 'C1'
    else:
        return 'C2'

# Calculate predictions and actual classes
predictions = []
actual_classes = data[:, -1]
for i in range(data.shape[0]):
    b = data[i, :-1].astype(int)
    prediction = classifier_f1(b)
    predictions.append(prediction)

# Calculate error rate epsilon_1
misclassified = np.sum(predictions != actual_classes)
total = data.shape[0]
epsilon_1 = misclassified / total

# Calculate the importance alpha_1 of the classifier f1
alpha_1 = 0.5 * np.log((1 - epsilon_1) / epsilon_1)

# Print the results
print(f"Predictions: {predictions}")
print(f"Actual Classes: {actual_classes.tolist()}")
print(f"Misclassified: {misclassified}")
print(f"Total Observations: {total}")
print(f"Error Rate (epsilon_1): {epsilon_1:.2f}")
print(f"Importance (alpha_1): {alpha_1:.2f}")
