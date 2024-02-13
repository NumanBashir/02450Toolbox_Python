import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic dataset
np.random.seed(42)  # For reproducible results
study_times = np.random.normal(5, 2, 100)  # Average study time of 5 hours with a standard deviation of 2
test_scores = study_times * 10 + np.random.normal(0, 5, 100)  # Test scores based on study time

# Calculate covariance and correlation
covariance = np.cov(study_times, test_scores)
correlation = np.corrcoef(study_times, test_scores)

print(f"Covariance matrix:\n{covariance}")
print(f"Correlation coefficient:\n{correlation}")

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(study_times, test_scores, color='blue', alpha=0.5, label='Data points')
plt.title('Study Time vs. Test Scores')
plt.xlabel('Study Time (hours)')
plt.ylabel('Test Scores')
plt.grid(True)
plt.legend()
plt.show()
