import numpy as np
import matplotlib.pyplot as plt

# Define the data points (example data)
x1 = np.random.rand(200)
x2 = np.random.rand(200)
labels = np.zeros(200)
labels[:100] = 1  # Class 1 (red crosses)
labels[100:] = 2  # Class 2 (black plusses)

# Generate decision boundaries
x = np.column_stack((x1, x2))

# Define the decision boundaries as per the new rules
def decision_boundary_A(x):
    return x[0] < 0.6

def decision_boundary_B(x):
    return np.linalg.norm(x - np.array([0.8, 0.4]), ord=np.inf) < 0.3

def decision_boundary_C(x):
    return np.linalg.norm(x - np.array([0.4, 0.6]), ord=2) < 0.2

def decision_boundary_D(x):
    return x[1] > 0.5

# Define the decision tree with new conditions
def decision_tree(x):
    if decision_boundary_A(x):
        return 1  # Class 1 (red cross)
    else:
        if decision_boundary_B(x):
            if decision_boundary_C(x):
                return 1  # Class 1 (red cross)
            else:
                return 2  # Class 2 (black plus)
        else:
            if decision_boundary_D(x):
                return 1  # Class 1 (red cross)
            else:
                return 2  # Class 2 (black plus)

# Apply the decision tree to classify points
predictions = np.array([decision_tree(point) for point in x])

# Plot the points and decision boundaries
plt.figure(figsize=(8, 6))

# Plot points
plt.scatter(x1[labels == 1], x2[labels == 1], color='red', marker='x', label='Class 1 (red crosses)')
plt.scatter(x1[labels == 2], x2[labels == 2], color='black', marker='+', label='Class 2 (black plusses)')

# Plot decision boundaries
xx, yy = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 1, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
decision_map = np.array([decision_tree(point) for point in grid])
decision_map = decision_map.reshape(xx.shape)

plt.contourf(xx, yy, decision_map, alpha=0.3, cmap='coolwarm')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Boundaries and Data Points')
plt.legend()
plt.show()
