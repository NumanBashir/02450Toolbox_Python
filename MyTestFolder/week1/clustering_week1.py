from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Simulated customer data (e.g., age and average spending)
X = np.array([[30, 500], [20, 600], [35, 200], [25, 700], [40, 300], [33, 400]])

# Applying KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Average Spending')
plt.title('Customer Segmentation')
plt.show()
