import numpy as np

# Data
x = np.array([1, 2, 3, 4])
y = np.array([6, 2, 3, 4])
w = np.array([-np.sqrt(3) / np.sqrt(20)])
w0 = 15 / 4

# Standardize the feature matrix
x_mean = np.mean(x)
x_std = np.sqrt(3 / 5) * (x - x_mean)
X_std = np.vstack(x_std)  # Make sure it's a column vector

# Compute predictions
predictions = X_std.dot(w) + w0

# Compute residuals and sum of squared residuals
residuals = y - predictions
sum_squared_residuals = np.sum(residuals**2)

# Total cost from ridge regression formula (given as 8)
total_cost = 8

# Calculate lambda
lambda_reg = (total_cost - sum_squared_residuals) / (np.linalg.norm(w)**2)

print("Standardized X:", X_std)
print("Residuals:", residuals)
print("Sum of squared residuals:", sum_squared_residuals)
print("Regularization constant lambda:", lambda_reg)
