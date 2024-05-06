import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, p, subplot):
    h = .02  # step size in the mesh
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    # Create an instance of Neighbours Classifier and fit the data with specified p-norm.
    clf = KNeighborsClassifier(1, metric='minkowski', p=p)
    clf.fit(X, y)

    # Plot the decision boundary by assigning a color in the grid.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    subplot.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
    subplot.set_title(f"2-Class classification (p={p})")

# Training data: two points
X = np.array([[0.301, 0.514], [0.34, 0.672]]) # TODO CHANGE THESE NUMBERS FROM THE QUESTION
y = np.array([0, 1])  # Class labels

# Create a figure with subplots for each p-norm
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
norms = [4, 1, 2, float('inf')]  # TODO CHANGE THESE VALUES FOR EACH ANSWER AND CHECK WHICH IS CORRECT
for ax, p in zip(axs.ravel(), norms):
    plot_decision_boundary(X, y, p, ax)

plt.tight_layout()
plt.show()
