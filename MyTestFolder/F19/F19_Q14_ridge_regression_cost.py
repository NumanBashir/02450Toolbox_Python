import numpy as np

# Given data from the question
x = np.array([2, 5, 6, 7])  # Feature values
y = np.array([6, 7, 7, 9])  # Target values
w = 0.6  # Weight vector w
lambda_ridge = 2  # Lambda for ridge regression

# Calculate the sample mean (mu) and sample standard deviation (sigma) of x
mu = np.mean(x)
sigma = np.std(x, ddof=1)  # ddof=1 to get the sample standard deviation

# # Mean and standard deviation for standardization of x
# mu = np.mean(x)
# sigma = np.std(x)

# Expected value of y (mean of y)
E_y = np.mean(y)

# Standardize x values and predict y
standardized_x = (x - mu) / sigma
predicted_y = w * standardized_x + E_y

# Compute the ridge regression cost
residuals = y - predicted_y
squared_residuals = residuals ** 2
ridge_penalty = lambda_ridge * (w ** 2)

# Total cost
cost = np.sum(squared_residuals) + ridge_penalty

print("Predicted y values:", predicted_y)
print("Cost:", cost)
