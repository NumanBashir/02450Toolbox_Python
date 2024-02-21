import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Considering 'G3' as the target variable and 'studytime' as one of the feature
# We can also consider another feature for visualization, let's take 'absences' for this example
# Assuming 'absences' is also a numerical feature

df = pd.read_csv('MyTestFolder/Project1/student-mat-selected.csv', sep=';')
# Extracting the relevant features for PCA
features_for_pca = df[['studytime', 'absences', 'G3']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_for_pca)

# Apply PCA to reduce to two components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Assuming we have a binary variable in the dataset that can be used to color the points
# For demonstration, let's classify 'G3' based on a simple median split
median_g3 = np.median(df['G3'])
colors = ['red' if g3 < median_g3 else 'green' for g3 in df['G3']]

# Plotting the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5)
plt.title('2 Component PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(['Below Median G3', 'Above Median G3'])
plt.grid(True)
plt.show()
