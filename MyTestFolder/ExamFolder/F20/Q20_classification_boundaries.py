import numpy as np
import matplotlib.pyplot as plt

# Define a function for the Euclidean distance from a point to the center
def norm(x, center):
    return np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)

# Define the classification logic based on the decision tree
def classify_points(x, centers, thresholds):
    d_A = norm(x, centers['A'])
    d_B = norm(x, centers['B'])
    d_C = norm(x, centers['C'])

    # TODO THESE NEED TO BE CHANGED TO MATCH THE DECISION TREE
    class1 = (d_A >= thresholds['A']) | ((d_A < thresholds['A']) & (d_B >= thresholds['B']))
    class3 = (d_A < thresholds['A']) & (d_B < thresholds['B'])
    class2 = (d_A < thresholds['A']) & (d_C < thresholds['C'])

    return class1, class2, class3

# Plotting function
def plot_classification(centers, thresholds):
    x1, x2 = np.meshgrid(np.linspace(0, 6, 400), np.linspace(0, 6, 400))
    x = np.array([x1.ravel(), x2.ravel()])
    
    class1, class2, class3 = classify_points(x, centers, thresholds)
    plt.figure(figsize=(8, 6))
    plt.scatter(x[0, class1], x[1, class1], color='blue', label='Class 1', alpha=0.3, s=10)
    plt.scatter(x[0, class2], x[1, class2], color='red', label='Class 2', alpha=0.3, s=10)
    plt.scatter(x[0, class3], x[1, class3], color='yellow', label='Class 3', alpha=0.3, s=10)
    plt.title('Classification based on Decision Tree')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)
    plt.show()

# TODO CHANGE THESE VALUES TO MATCH THE OPTIONS FROM THE QUESTION
centers = {'A': [6, 4], 'B': [4, 2], 'C': [2, 4]} 
thresholds = {'A': 3, 'B': 3, 'C': 3}  # TODO CHANGE THE VALUES FROM THE QUESTION

# Generate the plot
plot_classification(centers, thresholds)
