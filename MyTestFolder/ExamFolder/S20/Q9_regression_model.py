import numpy as np

# Given observation x6
x6 = np.array([-0.06, -0.28, 0.43, -0.30, -0.36, 0, 0, 0, 0, 1])

# Coefficients of the logistic regression model
coefficients = np.array([0.76, 1.76, -0.32, -0.96, 6.64, -5.13, -2.06, 96.73, 1.03, -2.74])
intercept = 1.41

# Calculate the linear combination (z)
z = intercept + np.dot(coefficients, x6)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Calculate the predicted probability
predicted_probability = sigmoid(z)

print(f"Predicted probability for x6: {predicted_probability}")

# Determine if the predicted probability is less than 1%
if predicted_probability < 0.01:
    print("The probability observation x6 belongs to class y = 1 is less than 1%.")
else:
    print("The probability observation x6 belongs to class y = 1 is not less than 1%.")
