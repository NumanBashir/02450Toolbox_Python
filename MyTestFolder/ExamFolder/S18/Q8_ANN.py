import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ANN parameters corrected
w1 = np.array([[0.0189, -0.4256],    # Weights for first and second hidden neuron for x5
               [0.9159, 3.7336]])    # Weights for first and second hidden neuron for x6
b1 = np.array([0.3799e-6, -0.8003])  # Biases for the hidden layer
w2 = np.array([0.0429e-6, 5.0741])   # Weights for output layer from each hidden neuron
b2 = -0.3440e-6                      # Bias for the output layer

# Define the ANN function using the corrected values
def ann(x5, x6):
    x = np.array([x5, x6])
    h = sigmoid(np.dot(w1, x) + b1)
    y = np.dot(w2, h) + b2
    return y

# Create a grid of x5 and x6 values for plotting
x5 = np.linspace(0, 24, 100)
x6 = np.linspace(3, 0, 100)  # Values from 3 to 0
X5, X6 = np.meshgrid(x5, x6)
Y = np.zeros_like(X5)

# Compute the output for each (x5, x6) pair
for i in range(X5.shape[0]):
    for j in range(X6.shape[1]):
        Y[i, j] = ann(X5[i, j], X6[i, j])

# Normalize the output for better visualization
Y_normalized = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

# Flip Y_normalized vertically
Y_normalized = np.flipud(Y_normalized)

# Plot the output
plt.figure(figsize=(8, 6))
cp = plt.contourf(X5, X6, Y_normalized, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.title('Normalized Output of the Trained ANN')
plt.xlabel('x5')
plt.ylabel('x6')
plt.gca().invert_yaxis()  # Optionally invert y-axis
plt.show()

# Check outputs at specific points for manual validation
points = [(0, 0), (20, 3)]
outputs = [ann(x5, x6) for x5, x6 in points]
print("Outputs at specified points:", outputs)
