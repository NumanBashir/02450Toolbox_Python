import numpy as np

# Define the matrix R where each row corresponds to a split and columns to class counts
R = np.array([
    [108, 112, 56],
    [58, 75, 116]
])

# Calculate total observations
N = R.sum()

# Calculate the number of observations in the largest classes for each split
n1 = np.max(R[0])
n2 = np.max(R[1])

# Calculate accuracy
accuracy = (n1 + n2) / N
print(f"Accuracy: {accuracy:.2f}")
