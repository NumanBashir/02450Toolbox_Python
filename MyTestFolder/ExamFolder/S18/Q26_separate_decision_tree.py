import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic dataset
np.random.seed(0)
x = np.random.uniform(0, 1, (1000, 2))
y = np.zeros(1000)

# Define a rough boundary similar to the image
ellipse_center = [0.5, 0.65]
ellipse_axes = [0.25, 0.15]

# Label points within the ellipse as red crosses (class 1), others as black plusses (class 2)
inside_ellipse = ((x[:, 0] - ellipse_center[0])**2 / ellipse_axes[0]**2 + 
                  (x[:, 1] - ellipse_center[1])**2 / ellipse_axes[1]**2) <= 1
y[inside_ellipse] = 1  # Red crosses
y[~inside_ellipse] = 2  # Black plusses

# Decision rules for each option
decision_rules_options = {
    'A': {'A': np.array([1/2, 1/2]), 'B': np.array([3/4, 1/2]), 'C': np.array([1/4, 1/2]), 'D': np.array([3/16, 9/16])},
    'B': {'A': np.array([1/2, 1/2]), 'B': np.array([3/4, 1/2]), 'C': np.array([1/4, 1/2]), 'D': np.array([3/16, 9/16])},
    'C': {'A': np.array([1/2, 1/2]), 'B': np.array([3/4, 1/2]), 'C': np.array([1/4, 1/2]), 'D': np.array([3/16, 9/16])},
    'D': {'A': np.array([1/2, 1/2]), 'B': np.array([3/4, 1/2]), 'C': np.array([1/4, 1/2]), 'D': np.array([3/16, 9/16])}
}

# Define the decision pathway according to the decision rules and norms specified
def decision_pathway(point, rules):
    A, B, C, D = rules['A'], rules['B'], rules['C'], rules['D']
    if np.linalg.norm(point - A, ord=np.inf) <= 1/4:
        if np.linalg.norm(point - B, ord=2) <= 1/4:
            return 1
        else:
            if np.linalg.norm(point - C, ord=1) <= 1/4:
                return 1
            else:
                if np.linalg.norm(point - D, ord=2) <= 1/20:
                    return 2
                else:
                    return 1
    else:
        return 2

# Test each option
for option, rules in decision_rules_options.items():
    predictions = np.array([decision_pathway(pt, rules) for pt in x])
    accuracy = np.sum(predictions == y) / len(y)
    print(f"Accuracy for Option {option}: {accuracy:.2%}")

    # Plot results for visual confirmation
    plt.figure()
    plt.scatter(x[predictions == 1, 0], x[predictions == 1, 1], c='red', marker='x', label='Classified as Class 1')
    plt.scatter(x[predictions == 2, 0], x[predictions == 2, 1], c='black', marker='+', label='Classified as Class 2')
    plt.title(f'Results for Option {option}')
    plt.legend()
    plt.show()
