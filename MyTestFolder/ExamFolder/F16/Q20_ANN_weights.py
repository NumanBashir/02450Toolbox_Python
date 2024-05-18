def relu(x):
    return max(0, x)


# Given weights
w13, w14, w15 = 0.5, 0.5, -0.5
w23, w24, w25 = 0.5, -0.5, 0.25
w36, w46, w56 = 0.25, -0.25, 0.25
# Input values
x1, x2 = 1, 1
# Calculate inputs to hidden layer neurons
n3_input = x1 * w13 + x2 * w23
n4_input = x1 * w14 + x2 * w24
n5_input = x1 * w15 + x2 * w25
# Calculate outputs of hidden layer neurons
n3_output = relu(n3_input)
n4_output = relu(n4_input)
n5_output = relu(n5_input)
# Calculate input to output layer neuron
n6_input = n3_output * w36 + n4_output * w46 + n5_output * w56
# Calculate output of output layer neuron
n6_output = relu(n6_input)
# Print the result
print(f"Output of the neural network (ŷ) = {n6_output}")
# Expected output: 0.25