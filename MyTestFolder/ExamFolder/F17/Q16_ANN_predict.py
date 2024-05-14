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



# Inputs
x1 = 6.8
x2 = 225
x3 = 0.44
x4 = 0.68

w1_1 = np.array([21.78, -1.65, 0, -13.26, -8,46])
w2_1 = np.array([-9.6, -0.44, 0.01, 14.54, 9.5])

w0_2 = 2.84
w1_2 = 3.25
w2_2 = 3.46
# Calculating the output using the given weights and activation function
# TODO REMEMBER TO REMOVE AND ADD THE THE AMOUNT OF X VALUES
output = (w1_2 * relu( w1_1[0]+ w1_1[1] * x1 + w1_1[2] * x2 + w1_1[3] * x3 + w1_1[4] * x4) +
          w2_2 * relu(w2_1[0] + w2_1[1] * x1 + w2_1[2] * x2 + w2_1[3] * x3 + w2_1[4] * x4) +
          w0_2)

print("Predicted fuel consumption:", output)
