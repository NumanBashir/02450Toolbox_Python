import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('MyTestFolder/ExamFolder/F20/penguins.csv')

# Assuming 'x1', 'x2', 'x3', 'x4' map to certain columns; here we need exact column names to proceed.
# For example purposes, let's say they correspond to the first four numeric columns:
features = df.select_dtypes(include=[np.number]).columns[:4]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# Principal component matrix
V = np.array([
    [0.45, -0.60, -0.64, 0.15],
    [-0.40, -0.80, 0.43, -0.16],
    [0.58, -0.01, 0.24, -0.78],
    [0.55, -0.08, 0.59, 0.58]
])

# Project the standardized data onto principal components
transformed_data = scaled_data @ V

# New observation
new_observation = np.array([-1, -1, -1, 1])
projected_observation = new_observation @ V

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
titles = ["PC1 vs PC4", "PC2 vs PC4", "PC1 vs PC3", "PC3 vs PC4"]
indices = [(0, 3), (1, 3), (0, 2), (2, 3)]
colors = ['blue' if species == 'Adelie' else 'green' if species == 'Gentoo' else 'red' for species in df['species']]

for ax, (i, j), title in zip(axs.flatten(), indices, titles):
    ax.scatter(transformed_data[:, i], transformed_data[:, j], c=colors, alpha=0.5)
    ax.scatter(projected_observation[i], projected_observation[j], c='black', marker='x', s=100, label='New Observation')
    ax.set_xlabel(f'PC{i+1}')
    ax.set_ylabel(f'PC{j+1}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
