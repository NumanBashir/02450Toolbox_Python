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
    [0.43, -0.5, 0.7, -0.25, -0.07],
    [0.38, 0.85, 0.3, -0.2, 0.03],
    [0.46, -0.13, -0.61, -0.61, -0.15],
    [-0.48, -0.0, 0.13, -0.63, 0.6],
    [-0.48, 0.1, 0.16, -0.36, -0.78]
])

# Hypothetical observations for the given options
observations = {
    'A': np.array([1, 1, 1, -1, -1]),
    'B': np.array([1, -1, 1, -1, 0]),
    'C': np.array([-1, 1, -1, 1, 0]),
    'D': np.array([-1, -1, 0, -1, -1])
}

# Function to calculate projection onto principal components
def calculate_projections(observation, V):
    projections = np.dot(V.T, observation)
    return projections

# Calculate and print projections for each observation
for label, obs in observations.items():
    projections = calculate_projections(obs, V)
    print(f'Projections for observation {label}: {projections}')

# Determine the sign of the projections for the specified principal components
for label, obs in observations.items():
    projections = calculate_projections(obs, V)
    if label == 'A':
        print(f"Observation A: Projection onto PC1: {projections[0]}")
    if label == 'B':
        print(f"Observation B: Projection onto PC2: {projections[1]}")
    if label == 'C':
        print(f"Observation C: Projection onto PC1: {projections[0]}")
    if label == 'D':
        print(f"Observation D: Projection onto PC1: {projections[0]}")
