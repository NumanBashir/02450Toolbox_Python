import numpy as np # type: ignore

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


# Input vector (including bias term)
x = np.array([1, 6, 120, 3.2, 0, 4])

# Weights for hidden layer
w1_1 = np.array([-4, 1, 0.01, 1, -1, -1])
w2_1 = np.array([-10, 1, -0.02, 1, 1, 1])

# Weights for output layer
w0_2 = 7
w1_2 = 8
w2_2 = 9

# Compute the activations for the hidden layer
h1_1 = tanh(np.dot(w1_1, x))
h1_2 = tanh(np.dot(w2_1, x))

# Compute the final output
output = w0_2 + w1_2 * h1_1 + w2_2 * h1_2
# TODO REMEMBER TO REMOVE AND ADD THE THE AMOUNT OF X VALUES

print("Predicted fuel consumption:", output)
