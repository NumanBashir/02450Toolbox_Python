import numpy as np

def project_onto_component(V, x, component_index):
    """
    Projects observation x onto the specified principal component.
    
    :param V: PCA loading matrix where each column is a principal component.
    :param x: The observation as a numpy array.
    :param component_index: Index (zero-based) of the principal component in V.
    :return: Projection scalar value.
    """
    principal_component = V[:, component_index]
    projection = np.dot(x, principal_component)
    return projection

# Define the PCA loading matrix V as provided
V = np.array([
    [0.48, 0.09, -0.57, 0.52, 0.42],
    [0.51, 0.03, -0.27, -0.82, 0.05],
    [-0.15, 0.98, 0.03, -0.07, 0.08],
    [-0.54, -0.16, -0.14, -0.25, 0.78],
    [0.45, 0.01, 0.77, 0.05, 0.46]
])

# Hypothetical observations for the given options
observations = {
    'A': np.array([0.301, 0.514, 0.672, 0.340, 0.564]),
    'B': np.array([0.672, 0.301, 0.514, 0.340, 0.564]),
    'C': np.array([0.301, 0.672, 0.514, 0.340, 0.564]),
    'D': np.array([0.514, 0.301, 0.672, 0.340, 0.564])
}

# Choose the principal component to project onto (zero-based index)
component_index = 0  # Change this based on the question's focus (e.g., for the second component use 1)

# Calculate and print the projections for each observation
for label, obs in observations.items():
    projection = project_onto_component(V, obs, component_index)
    print(f"Projection of observation {label} onto component {component_index + 1}: {projection}")
