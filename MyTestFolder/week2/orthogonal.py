import numpy as np
import matplotlib.pyplot as plt

# Define two orthogonal vectors
v1 = np.array([2, 2])
v2 = np.array([-2, 2])

# Normalize v1 and v2 to get orthonormal vectors
v1_orthonormal = v1 / np.linalg.norm(v1)
v2_orthonormal = v2 / np.linalg.norm(v2)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))

# Plot orthogonal vectors
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Orthogonal v1')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Orthogonal v2')

# Plot orthonormal vectors
ax.quiver(0, 0, v1_orthonormal[0], v1_orthonormal[1], angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5, width=0.01, label='Orthonormal v1')
ax.quiver(0, 0, v2_orthonormal[0], v2_orthonormal[1], angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.5, width=0.01, label='Orthonormal v2')

# Setting plot limits
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Adding labels and legend
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.title('Orthogonal and Orthonormal Vectors')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
