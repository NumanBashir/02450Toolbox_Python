import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the ANN function
def ann_output(x1, x2):
    w1_1 = np.array([-1.2, -1.3, 0.6])
    w1_2 = np.array([-1.0, -0.0, 0.9])
    w2 = np.array([-0.3, 0.5])
    w2_0 = 2.2

    h1 = sigmoid(w1_1[0] + w1_1[1] * x1 + w1_1[2] * x2)
    h2 = sigmoid(w1_2[0] + w1_2[1] * x1 + w1_2[2] * x2)
    
    output = w2_0 + w2[0] * h1 + w2[1] * h2
    return output

# Create a grid of (x1, x2) points
x1_range = np.linspace(-3, 3, 100)
x2_range = np.linspace(-3, 3, 100)
x1, x2 = np.meshgrid(x1_range, x2_range)

# Compute the ANN output for each point in the grid
z = ann_output(x1, x2)

# Plot the results
plt.figure(figsize=(8, 6))
contour = plt.contourf(x1, x2, z, levels=50, cmap='RdYlBu')
plt.colorbar(contour)
plt.title('ANN Output')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
