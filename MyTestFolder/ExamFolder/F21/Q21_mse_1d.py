import numpy as np

# Given data points
x = np.array([1, 2, 3, 4])
y = np.array([6, 2, 3, 4])

# Apply the transformation to the features
X_transformed = np.column_stack([
    np.cos(np.pi * x / 2),
    np.sin(np.pi * x / 2)
])

# Solve for weights using the normal equation
# Compute the pseudo-inverse of X_transformed
X_pinv = np.linalg.pinv(X_transformed)

# Compute the weights
w_star = X_pinv @ y

print("Computed weights w*:", w_star)
