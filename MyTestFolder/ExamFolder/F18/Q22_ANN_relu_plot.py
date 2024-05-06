import numpy as np
import matplotlib.pyplot as plt

# TODO: CHOOSE ANY OF THE ACTIVATION FUNCTIONS FROM BELOW AND IF UNSURE WHICH TO CHOOSE, TEST ALL AND SEE WHICH MATCHES

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x) # Relu will give rectified linear outputs

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)

def softmax(x):
    """Softmax activation function."""
    e_x = np.exp(x - np.max(x))  # shift values for numerical stability
    return e_x / e_x.sum(axis=0)  # sum across features if the input is a vector

def linear(x):
    """Linear activation function."""
    return x

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, x * alpha)

def elu(x, alpha=1.0):
    """ELU activation function."""
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))


# TODO: CHANGE THE WEIGHTS --> Weights for the neural network
w1_1 = np.array([-1.8, -1.1])
w1_2 = np.array([-0.6, 3.8])
w2 = np.array([-0.8, -0.1, 2.1])  # TODO: [w0, w(2)] --> This includes the bias and two weights for the output neuron

# Define the neural network function
def neural_network(x):
    # x should be [1, x7] to include bias and input

    # TODO: UPDATE FUNCTION NAME FROM CHOSEN ACTIVATION FUNCTION
    n1 = relu(np.dot(w1_1, x))  # Activation from first hidden neuron
    n2 = relu(np.dot(w1_2, x))  # Activation from second hidden neuron
    
    # Output from network
    # No activation function here, just linear combination
    output = w2[0] + w2[1] * n1 + w2[2] * n2
    return output

# Generate a range of x7 values for plotting
x7_values = np.linspace(-2, 2, 400)
outputs = []

# Calculate network output for each x7 value
for x7 in x7_values:
    x_vector = np.array([1, x7])  # Bias and input x7
    output = neural_network(x_vector)
    outputs.append(output)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x7_values, outputs, label='ANN output', color='blue')
plt.title('Output of the ANN as a Function of Attribute x7')
plt.xlabel('x7')
plt.ylabel('Output f(x7)')
plt.grid(True)
plt.legend()
plt.show()
