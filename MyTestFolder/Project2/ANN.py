import numpy as np

# Define the ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Function to compute the output of the ANN
def predict_ann(input_vector, weights_hidden, weights_output, bias_output):
    # Calculate the hidden layer activations
    hidden_activations = relu(np.dot(weights_hidden, input_vector))
    # Calculate the output
    output = bias_output + np.dot(weights_output, hidden_activations)
    return output

# Example usage:
# Given weights for the hidden layer
weights_hidden = np.array([[21.78, -1.65, 0, -13.26, -8.46],
                           [-9.6, -0.44, 0.01, 14.54, 9.5]])  # Adjusted to match input dimensions

# Given weights for the output layer
weights_output = np.array([3.25, 3.46])
bias_output = 2.84

# Given input
input_vector = np.array([6.8, 225, 0.44, 0.68])

# Predict the output
predicted_score = predict_ann(input_vector, weights_hidden, weights_output, bias_output)
print(predicted_score)
