import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define three orthogonal vectors in 3D space
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

# Normalize these vectors to get orthonormal vectors (in this case, they are already orthonormal)
v1_orthonormal = v1 / np.linalg.norm(v1)
v2_orthonormal = v2 / np.linalg.norm(v2)
v3_orthonormal = v3 / np.linalg.norm(v3)

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot orthogonal vectors
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='Orthogonal v1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='Orthogonal v2')
ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='g', label='Orthogonal v3')

# Plot orthonormal vectors (in this case, same as orthogonal since original vectors are unit vectors)
ax.quiver(0, 0, 0, v1_orthonormal[0], v1_orthonormal[1], v1_orthonormal[2], color='r', alpha=0.5, linestyle='dotted', label='Orthonormal v1')
ax.quiver(0, 0, 0, v2_orthonormal[0], v2_orthonormal[1], v2_orthonormal[2], color='b', alpha=0.5, linestyle='dotted', label='Orthonormal v2')
ax.quiver(0, 0, 0, v3_orthonormal[0], v3_orthonormal[1], v3_orthonormal[2], color='g', alpha=0.5, linestyle='dotted', label='Orthonormal v3')

# Setting plot limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Adding labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Orthogonal and Orthonormal Vectors in 3D')
plt.show()
