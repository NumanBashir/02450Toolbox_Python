import numpy as np

# Weight vectors for the multinomial regression model
# TODO: Change these
w1 = np.array([0.04, 1.32, -1.48])
w2 = np.array([-0.03, 0.7, -0.85])

# Input observation
# TODO: Look out for the given b_i values
b = np.array([1, -5.52, -4.69])  # Including the bias term b0 = 1

# Compute the logits
eta1 = np.dot(w1, b)
eta2 = np.dot(w2, b)

# Compute the probability for each class using the softmax function
def softmax(eta):
    exp_eta = np.exp(eta - np.max(eta))  # Numerical stability improvement
    return exp_eta / np.sum(exp_eta)

# Assuming a three-class system, and the logits for the third class (eta3) is zero (implicit)
eta = np.array([eta1, eta2, 0])
p_y_given_x = softmax(eta)

print("Class-assignment probability vector:", p_y_given_x)
