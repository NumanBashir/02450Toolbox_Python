import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

# Weights for the neural network
w1_1 = np.array([-0.5, -0.1])
w1_2 = np.array([0.9, 2.0])
w2 = np.array([1.4, -1.0, 0.4])  # Adjusted to include the bias and two weights for output neuron

# Define the neural network function
def neural_network(x):
    # x should be [1, x2] for bias and input
    n1 = sigmoid(np.dot(w1_1, x))  # Activation from first hidden neuron
    n2 = sigmoid(np.dot(w1_2, x))  # Activation from second hidden neuron
    
    # Output from network
    # Updated to correctly use the second weight for n2 if that was your intention
    output = sigmoid(w2[0] + w2[1] * n1 + w2[2] * n2)
    return output

# Generate a range of x2 values for plotting
x2_values = np.linspace(-2, 2, 400)
outputs = []

# Calculate network output for each x2 value
for x2 in x2_values:
    x_vector = np.array([1, x2])  # Bias and input x2
    output = neural_network(x_vector)
    outputs.append(output)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x2_values, outputs, label='ANN output', color='blue')
plt.title('Output of the ANN as a Function of Attribute x2')
plt.xlabel('x2')
plt.ylabel('Output f(x)')
plt.grid(True)
plt.legend()
plt.show()
