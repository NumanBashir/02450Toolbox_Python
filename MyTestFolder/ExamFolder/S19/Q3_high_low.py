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
    [0.94, -0.12, 0.32, -0.0, 0.0],
    [0.01, 0.0, -0.02, 0.0, -1.0],
    [-0.01, 0.07, 0.07, 0.99, -0.0],
    [0.11, 0.99, 0.06, -0.08, 0.0],
    [-0.33, -0.02, 0.94, -0.07, -0.02]
])

# Hypothetical observations for the given options
observations = {
    'A': np.array([-1, 0, 0, 0, 1]),
    'B': np.array([-1, 0, 0, 0, -1]),
    'C': np.array([-1, 0, 0, 0, 1]),
    'D': np.array([0, 1, 0, 0, 0])
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
        print(f"Observation B: Projection onto PC3: {projections[2]}")
    if label == 'C':
        print(f"Observation C: Projection onto PC1: {projections[0]}")
    if label == 'D':
        print(f"Observation D: Projection onto PC5: {projections[4]}")
