import numpy as np

# Provided data matrix X
X = np.array([
    [-0.6, -0.6,  2.5, -0.1],
    [-0.8, -0.3, -1.0,  1.2],
    [-0.7,  0.3, -0.2, -0.1],
    [ 1.4,  1.0,  0.1, -2.8],
    [-0.2,  0.8, -1.2,  0.7]
])

# Provided observation x4
x4 = np.array([1.4, 1.0, 0.1, -2.8])

# Calculate the mean of the original data X
mean_X = np.mean(X, axis=0)

# Center the observation x4 by subtracting the mean
tilde_x4 = x4 - mean_X

# Provided principal component matrix V (first two components)
V = np.array([
    [ 0.43, -0.26,  0.22, -0.84],
    [ 0.17, -0.37,  0.81,  0.42],
    [ 0.33,  0.88,  0.34, -0.01],
    [-0.82,  0.14,  0.42, -0.36]
])
V_2 = V[:, :2]

# Project the centered x4 onto the space spanned by the first two principal components
z = V_2.T.dot(tilde_x4)

# Reconstruct the approximation of x4 from the projection, adding back the mean
x4_hat = V_2.dot(z) + mean_X
print(x4_hat)
