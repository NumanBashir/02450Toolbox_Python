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
v1 = -0.3413
v2 = -0.6508
V = np.array([
    [-0.5939, 0.2906, v1, 0.0621, 0.6652],
    [-0.6521, 0.0759, 0.0004, 0.3813, v2],
    [0.2028, -0.5105, -0.7036, 0.4508, 0.0010],
    [-0.3696, -0.5414, -0.1781, -0.7244, -0.1173],
    [-0.2102, -0.5967, 0.5973, 0.3503, 0.3467]
])

# Define the observation vectors (low=-1, high=1)
observations = {
    'A': np.array([-1, 1, -1, 1, 1]),
    'B': np.array([0, 1, 1, -1, 1]),
    'C': np.array([-1, 0, 1, 1, 1]),
    'D': np.array([1, 1, -1, 1, 1])
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
