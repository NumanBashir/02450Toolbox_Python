import numpy as np

# The singular values from the matrix S in the PCA decomposition
singular_values = np.array([14.4, 8.19, 7.83, 6.91, 6.01]) ## CHANGE THIS ARRAY

# Calculate the variance explained by each principal component
# Variance explained is the square of the singular values divided by the sum of squares of all singular values
variance_explained = singular_values**2 / np.sum(singular_values**2)

# Calculate cumulative variance explained by the first N components
cumulative_variance_explained = np.cumsum(variance_explained)

# Store the results in a dictionary for easy interpretation
variance_results = {
    "First_Component": variance_explained[0],
    "First_Four_Components": cumulative_variance_explained[3],
    "Last_Four_Components": np.sum(variance_explained[1:]),
    "First_Three_Components": cumulative_variance_explained[2]
}

print(variance_results)
