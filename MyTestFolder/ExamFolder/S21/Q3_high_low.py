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
    [0.11, -0.8, 0.3, -0.17, -0.48],
    [-0.58, -0.31, 0.01, -0.5, 0.56],
    [0.49, 0.08, -0.49, -0.72, -0.07],
    [0.6, -0.36, 0.04, 0.27, 0.66],
    [-0.23, -0.36, -0.82, 0.37, -0.09]
])

# Hypothetical observations for the given options
observations = {
    'A': np.array([0, -1, 1, 1, -1]),
    'B': np.array([1, 0, -1, 1, -1]),
    'C': np.array([1, -1, 0, -1, 0]),
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
        print(f"Observation B: Projection onto PC4: {projections[3]}")
    if label == 'C':
        print(f"Observation C: Projection onto PC2: {projections[1]}")
    if label == 'D':
        print(f"Observation D: Projection onto PC1: {projections[0]}")
