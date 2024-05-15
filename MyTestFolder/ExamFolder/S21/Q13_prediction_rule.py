import numpy as np

# Data from Table 3
x1 = np.array([-1.1, -0.8, 0.08, 0.18, 0.34, 0.6, 1.42, 1.68])
yr = np.array([12, 5, 10, 23, 6, 17, 14, 13])

# Define the splitting rules from the tree
split1 = 0.13
split2 = 0.26

# Calculate the mean of yr for each region defined by the splits
# Region 1: x1 <= 0.13
region1_indices = x1 <= split1
region1_mean = np.mean(yr[region1_indices])

# Region 2: 0.13 < x1 <= 0.26
region2_indices = (x1 > split1) & (x1 <= split2)
region2_mean = np.mean(yr[region2_indices])

# Region 3: x1 > 0.26
region3_indices = x1 > split2
region3_mean = np.mean(yr[region3_indices])

# Print the means for each region
print(f"Region 1 mean (x1 <= 0.13): {region1_mean}")
print(f"Region 2 mean (0.13 < x1 <= 0.26): {region2_mean}")
print(f"Region 3 mean (x1 > 0.26): {region3_mean}")

# Plot the prediction rule to match with the figures
import matplotlib.pyplot as plt

x_plot = np.linspace(-1.5, 2, 500)
y_plot = np.piecewise(x_plot, [x_plot <= split1, (x_plot > split1) & (x_plot <= split2), x_plot > split2], [region1_mean, region2_mean, region3_mean])

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='Prediction rule')
plt.scatter(x1, yr, color='red', label='Data points')
plt.axvline(x=split1, color='green', linestyle='--', label='Split at 0.13')
plt.axvline(x=split2, color='blue', linestyle='--', label='Split at 0.26')
plt.xlabel('$x_1$')
plt.ylabel('$\hat{y}_r$')
plt.title('Prediction rule based on regression tree')
plt.legend()
plt.show()
