import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Generer eksempeldata
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# Definer klassifikatorer
classifiers = {
    'Logistic Regression': LogisticRegression(),
    '1-NN': KNeighborsClassifier(n_neighbors=1),
    '3-NN': KNeighborsClassifier(n_neighbors=3),
    'ANN with 1 hidden unit': MLPClassifier(hidden_layer_sizes=(1,), max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=2)
}

# Plot beslutningsgrænser
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

for ax, (name, clf) in zip(axes, classifiers.items()):
    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
    ax.set_title(name)
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_3$')

plt.tight_layout()
plt.show()
