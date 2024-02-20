import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

# Simulate heights of a population
mean_height = 170  # cm for an adult population
std_dev_height = 10  # cm

# Generate a sample of 1000 heights
heights = np.random.normal(mean_height, std_dev_height, 1000)

# Plot the distribution of heights
plt.hist(heights, bins=30, density=True, alpha=0.6, color='g')

# Add a best fit line
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_height, std_dev_height)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mean_height, std_dev_height)
plt.title(title)

plt.xlabel('Height (cm)')
plt.ylabel('Probability Density')
plt.show()

# Now, simulate a multivariate normal distribution for height and weight
mean = [mean_height, 70]  # 170 cm tall, 70 kg weight
covariance = [[100, 20], [20, 36]]  # covariance between height and weight

# Generate a bivariate sample of 1000 individuals
sample = np.random.multivariate_normal(mean, covariance, 1000)

# Plot the distribution of heights and weights
plt.figure(figsize=(10, 8))
x, y = sample.T
plt.scatter(x, y, alpha=0.5)

# Show the best fit line
x, y = np.mgrid[140:200:.5, 50:90:.5]
pos = np.dstack((x, y))
rv = multivariate_normal(mean, covariance)
contour = plt.contour(x, y, rv.pdf(pos), colors='r')

plt.title('Multivariate Normal Distribution')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid(True)
plt.show()
