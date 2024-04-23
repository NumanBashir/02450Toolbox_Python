import numpy as np
import matplotlib.pyplot as plt

# Define the tanh activation function
def tanh(x):
    return np.tanh(x)

# Given weights
w_1 = np.array([2.2, 0.7, -0.3])
w_2 = np.array([-0.2, 0.8, 0.4])
w = np.array([-0.7, 0.5])
w_0 = 2.2

# Define the ANN function f(x, w)
def f(x1, x2, w_1, w_2, w, w_0):
    h1_output = tanh(np.dot([1, x1, x2], w_1))
    h2_output = tanh(np.dot([1, x1, x2], w_2))
    return tanh(w_0 + w[0] * h1_output + w[1] * h2_output)

# Create a grid of x1 and x2 values
x1_vals = np.linspace(-2, 2, 300)
x2_vals = np.linspace(-2, 2, 300)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Compute the ANN output for each point on the grid
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = f(X1[i, j], X2[i, j], w_1, w_2, w, w_0)

# Plot the ANN output
plt.figure(figsize=(8, 6))
contour = plt.contourf(X1, X2, Z, levels=np.linspace(-1, 1, 11), cmap='RdYlBu')
plt.colorbar(contour)
plt.title('ANN Output for Continuous Attribute $x_3$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
