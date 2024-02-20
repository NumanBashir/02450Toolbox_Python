import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the normal distribution
mean_rainfall = 3.0 # average rainfall in mm
std_dev_rainfall = 1.0 # standard deviation of rainfall in mm

# Generate values on the x axis between 0 and twice the mean rainfall
rainfall_values = np.linspace(0, mean_rainfall * 2, 1000)

# Calculate the probability density for each value
probability_densities = norm.pdf(rainfall_values, mean_rainfall, std_dev_rainfall)

# Plot the probability density function for rainfall
plt.plot(rainfall_values, probability_densities, label='Probability Density')

# Fill under the probability density curve for a specific range
plt.fill_between(rainfall_values, probability_densities, where=(rainfall_values > 2) & (rainfall_values < 4), color='skyblue', alpha=0.5, label='2-4 mm probability')

# Add titles and labels
plt.title('Rainfall Probability Density Function')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Probability Density')
plt.legend()

# Show the plot
plt.show()
