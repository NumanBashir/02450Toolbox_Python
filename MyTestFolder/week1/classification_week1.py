import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generating synthetic data for demonstration
np.random.seed(42)  # Ensuring reproducibility
n_samples = 200
annual_income = np.random.normal(50, 15, n_samples)  # Simulated income in $1,000s
credit_score = np.random.normal(600, 100, n_samples)  # Simulated credit scores
features = np.column_stack((annual_income, credit_score))

# Generating synthetic outcomes based on simplified criteria
outcomes = (annual_income > 45) & (credit_score > 650)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.25, random_state=42)

# Training a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Visualizing the dataset and decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 50, X[:, 1].max() + 50
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.5, levels=[0, 0.5, 1], cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Annual Income (in $1,000s)')
    plt.ylabel('Credit Score')
    plt.title('Loan Approval Prediction')
    plt.colorbar(ticks=[0, 1])
    plt.show()

plot_decision_boundary(features, outcomes, model)
