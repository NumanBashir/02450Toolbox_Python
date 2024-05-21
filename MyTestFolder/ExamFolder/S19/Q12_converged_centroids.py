import numpy as np

# Dataset
x = np.array([1.0, 1.2, 1.8, 2.3, 2.6, 3.4, 4.0, 4.1, 4.2, 4.6])

# Initial cluster centers
mu1, mu2, mu3 = 1.8, 3.3, 3.6
centers = np.array([mu1, mu2, mu3])

# Function to perform k-means
def k_means(x, centers, K=3):
    while True:
        # Assign points to the nearest cluster center
        distances = np.abs(x[:, np.newaxis] - centers)
        labels = np.argmin(distances, axis=1)
        
        # Calculate new centers
        new_centers = np.array([x[labels == k].mean() for k in range(K)])
        
        # Check for convergence
        if np.all(new_centers == centers):
            break
        centers = new_centers
    return centers

# Run k-means algorithm
final_centers = k_means(x, centers)

print(f"The final cluster centers are: {final_centers}")
