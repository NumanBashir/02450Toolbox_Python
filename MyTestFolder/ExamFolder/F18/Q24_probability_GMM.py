import numpy as np
import scipy.stats as stats

# Mixture weights
pis = np.array([0.15, 0.53, 0.32])

# Means and standard deviations of the components (estimated from your description)
means = np.array([1.46, 1.6, 1.25])  # TOP OF THE GRAPH
std_devs = np.array([1.3, 1.2, 1.4])  # THESE CAN BE READ ON THE GRAPH --> ESTIMATE THE 95% QUANTILE

# Test observation's x-value (estimated or given)
x_test = 0

# Calculate Gaussian PDF for each component at the x-value of the test observation
pdf_values = [stats.norm.pdf(x_test, mean, std) for mean, std in zip(means, std_devs)]

# Calculate responsibilities (gamma values)
gamma_values = pis * pdf_values / np.dot(pis, pdf_values)
gamma_component_3 = gamma_values[2]

print("Probability that the test observation is assigned to component 3 (γ_ik):", gamma_component_3)
