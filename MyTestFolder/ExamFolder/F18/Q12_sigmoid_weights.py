import numpy as np

'''
This script does not calculate the exact weight, but it finds which answer is closest. And that will be the answer. 
'''



# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Input feature and target probability from the graph
x8 = 1
target_probability = 0.69  # Change this based on the figure's indication -- LOOK AT Xi = 1 AND SEE WHERE IT INTERSECTS WITH THE GRAPH -> TARGET

# Weight options -- MAKE SURE TO CHANGE THESE FROM THE QUESTION
weight_options = {
    'A': np.array([-0.93, 1.72]),
    'B': np.array([-2.82, 0.0]),
    'C': np.array([1.36, 0.4]),
    'D': np.array([-0.65, 0.0])
}

# Calculate predictions
for label, weights in weight_options.items():
    prediction = sigmoid(np.dot(weights, [1, x8]))  # [1, x8] accounts for the bias
    print(f"Option {label}: Prediction = {prediction:.4f}, Difference from target = {abs(prediction - target_probability):.4f}")

# Identify which weight gives the closest prediction to the target
closest = min(weight_options, key=lambda k: abs(sigmoid(np.dot(weight_options[k], [1, x8])) - target_probability))
print(f"Weight set {closest} gives the closest prediction to the target.")
