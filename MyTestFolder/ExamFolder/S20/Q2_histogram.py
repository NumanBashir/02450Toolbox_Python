import numpy as np
import matplotlib.pyplot as plt

# Given statistics
stats = {
    'BirthRt': {'mean': 29.46, 'std': 13.62, 'percentiles': [14.6, 29, 42.575]},
    'DeathRt': {'mean': 10.73, 'std': 4.66, 'percentiles': [7.7, 9.5, 12.4]},
    'InfMort': {'mean': 55.28, 'std': 46.05, 'percentiles': [13.025, 43, 88.25]},
    'LExpM': {'mean': 61.38, 'std': 9.67, 'percentiles': [55.2, 63.4, 68.55]}
}

# Number of samples to generate
num_samples = 1000

# Generate sample data
data_samples = {key: np.random.normal(loc=value['mean'], scale=value['std'], size=num_samples) for key, value in stats.items()}

# Create a 2x2 grid of histograms
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot histograms
for ax, (key, value) in zip(axs.flatten(), stats.items()):
    data = data_samples[key]
    percentiles = value['percentiles']
    
    ax.hist(data, bins=30, alpha=0.7, color='b', edgecolor='black')
    ax.set_title(f'Histogram of {key}')
    ax.set_xlabel(key)
    ax.set_ylabel('Frequency')
    
    # Add vertical lines for percentiles
    for perc in percentiles:
        ax.axvline(perc, color='r', linestyle='dashed', linewidth=1)
        ax.text(perc, ax.get_ylim()[1]*0.9, f'{perc}', color='r', ha='center')
    
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
