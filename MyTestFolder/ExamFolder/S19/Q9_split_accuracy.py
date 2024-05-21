import numpy as np

# Define the matrix Rki
R = np.array([
    [223, 40],
    [251, 108],
    [197, 161]
])

# Total number of observations
N = np.sum(R)

# Calculate the number of correctly classified observations
n1 = np.max(R[:, 0])  # Maximum in the first split
n2 = np.max(R[:, 1])  # Maximum in the second split

# Total correctly classified
correctly_classified = n1 + n2

# Calculate accuracy
accuracy = correctly_classified / N

print(f"Accuracy: {accuracy:.4f}")
