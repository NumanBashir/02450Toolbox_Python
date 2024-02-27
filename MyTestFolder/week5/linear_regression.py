from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Load the Boston housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor
tree_regressor = DecisionTreeRegressor(max_depth=3, random_state=42)

# Fit the model to the training data
tree_regressor.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = tree_regressor.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared score
mse = np.mean((y_pred - y_test) ** 2)
r2_score = tree_regressor.score(X_test, y_test)

mse, r2_score
