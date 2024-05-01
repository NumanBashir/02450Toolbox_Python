import numpy as np

# Synthetic dataset
x = np.array([-3.4, -1.3, 0.5, 2.4, 4.2])
y = np.array([-2.9, -0.4, 0.7, 2.5, 4.5])

# Standardization parameters
mu = np.mean(x)
sigma = np.std(x, ddof=1)  # Using ddof=1 for sample standard deviation

# Standardize x
x_standardized = (x - mu) / sigma

# Lambda for regularization
lambda_ = 0.7

# Calculate the intercept w0 as per the provided formula
w0 = np.mean(y)  # Since the formula was w0 = 1/5 * sum(yi), which simplifies to the mean of y

# Calculate w1 using the corrected formula
w1 = np.sum(x_standardized * (y - w0)) / (np.sum(x_standardized**2) + lambda_)

# Predicting for x2 (value at index 1 in the original x array, corresponding to -1.3)
x2_standardized = (-1.3 - mu) / sigma # This needs to be looked in the table to make sure it is the right number
y_pred = w1 * x2_standardized + w0

print("w0:", w0)
print("w1:", w1)
print("Predicted y for x2:", y_pred)
