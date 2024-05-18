import numpy as np
import matplotlib.pyplot as plt

# Generate the data
np.random.seed(0)
num_points = 500
theta = np.linspace(0, 2 * np.pi, num_points)
radius = 0.35
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)

# Generate points inside the circle (class 1)
x1 = np.random.uniform(-radius, radius, size=num_points)
y1 = np.random.uniform(-radius, radius, size=num_points)
class1 = np.array([x1, y1]).T
class1 = class1[np.linalg.norm(class1, axis=1) <= radius]

# Generate points outside the circle (class 2)
x2 = np.random.uniform(-0.5, 0.5, size=num_points)
y2 = np.random.uniform(-0.5, 0.5, size=num_points)
class2 = np.array([x2, y2]).T
class2 = class2[np.linalg.norm(class2, axis=1) > radius]

# Plot the data
plt.figure(figsize=(8, 8))
plt.scatter(class1[:, 0], class1[:, 1], color='black', label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], color='red', label='Class 2')

# Plot the decision boundaries
# Option A: ||x||1 <= 1/4 and ||x||2 <= 3/8
x = np.linspace(-0.5, 0.5, 400)
plt.plot(x, 1/4 - np.abs(x), 'g--', label='Option A: ||x||1 <= 1/4 and ||x||2 <= 3/8')

# Option B: ||x||2 <= 3/8 and ||x||∞ <= 1/4
plt.plot(x, 3/8 * np.ones_like(x), 'b--', label='Option B: ||x||2 <= 3/8 and ||x||∞ <= 1/4')
plt.plot(1/4 * np.ones_like(x), x, 'b--')

# Option C: ||x||1 <= 1/2 and ||x||∞ <= 1/2
plt.plot(x, 1/2 - np.abs(x), 'm--', label='Option C: ||x||1 <= 1/2 and ||x||∞ <= 1/2')
plt.plot(1/2 * np.ones_like(x), x, 'm--')
plt.plot(-1/2 * np.ones_like(x), x, 'm--')

# Option D: ||x||1 <= 1/2 and ||x||∞ <= 3/8
plt.plot(x, 3/8 * np.ones_like(x), 'c--', label='Option D: ||x||1 <= 1/2 and ||x||∞ <= 3/8')
plt.plot(-3/8 * np.ones_like(x), x, 'c--')
plt.plot(3/8 * np.ones_like(x), x, 'c--')
plt.plot(x, 1/2 - np.abs(x), 'c--')

# Finalize the plot
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.axhline(0, color='k', linestyle='--', lw=0.5)
plt.axvline(0, color='k', linestyle='--', lw=0.5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Decision Boundaries for Two-Class Classification Problem')
plt.grid(True)
plt.show()
