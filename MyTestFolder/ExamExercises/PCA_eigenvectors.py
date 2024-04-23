# Provided singular values from Σ
import numpy as np


singular_values = np.array([3.7, 3.04, 0.56, 0.48])

# Calculate the proportion of total variance explained by the first two components
variance_explained_first_two = (singular_values[0]**2 + singular_values[1]**2) / np.sum(singular_values**2)

# Provided eigenvector matrix V
V = np.array([
    [0.43, -0.26, 0.22, -0.84],
    [0.17, -0.37, 0.81, 0.42],
    [0.33, 0.88, 0.34, -0.01],
    [-0.82, 0.14, 0.42, -0.36]
])

# Check the signs of the coefficients of the first eigenvector
first_eigenvector_signs = np.sign(V[:, 0])
signs = {
    'Temperature': first_eigenvector_signs[0],
    'Pressure': first_eigenvector_signs[1],
    'Humidity': first_eigenvector_signs[2],
    'Vacuum': first_eigenvector_signs[3]
}

print(variance_explained_first_two) 
print(signs)
