import numpy as np
import matplotlib.pyplot as plt

# Define the logistic sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define multiple weight vectors and corresponding transformations
configurations = [
    (np.array([0.31, -0.06, 0.07]), lambda b1, b2: np.array([1, b1**2, b2**3])),  # Example 1
    (np.array([0.72, 3.13, -0.25]), lambda b1, b2: np.array([1, b1**2, b2**3])), # Example 2
    (np.array([0.31, -0.06, 0.07]), lambda b1, b2: np.array([1, b1**3, b2**2])), # Example 3
    (np.array([0.72, 3.13, -0.25]), lambda b1, b2: np.array([1, b1**3, b2**2])) # Example 4
]

# Function to plot the contour for given weights and transformation
def plot_contour(weights, transform, title):
    # Create a grid of points (b1, b2)
    b1_vals = np.linspace(-4, 4, 300)
    b2_vals = np.linspace(-4, 4, 300)
    B1, B2 = np.meshgrid(b1_vals, b2_vals)

    # Apply the transformation and calculate the probability for each point
    probabilities = np.zeros_like(B1)
    for i in range(B1.shape[0]):
        for j in range(B1.shape[1]):
            b1, b2 = B1[i, j], B2[i, j]
            transformed_b = transform(b1, b2)
            probabilities[i, j] = sigmoid(weights.dot(transformed_b))

    # Plot the probabilities
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(B1, B2, probabilities, levels=np.linspace(0, 1, 11), cmap='RdYlBu')
    plt.colorbar(contour)
    plt.title(title)
    plt.xlabel('$b_1$')
    plt.ylabel('$b_2$')
    plt.show()

# Generate plots for each configuration
for index, (weights, transform) in enumerate(configurations):
    plot_title = f'Logistic Regression Classifier (Option {index+1})'
    plot_contour(weights, transform, plot_title)
