import numpy as np
from scipy.spatial import distance

a = np.array((26, 0, 2, 0, 0, 0, 0))
b = np.array((19, 0, 0, 0, 0, 0, 0))

# Manhattan norm (p=1)
manhattan_dist = distance.cityblock(a, b)
print("Manhattan distance:", manhattan_dist)

# Euclidean norm (p=2)
dst = distance.euclidean(a, b)
print("Euclidean distance:", dst)

# Infinity norm (p=infinity)
infinity_norm_distance = np.max(np.abs(a - b))
print("Infinity norm distance:", infinity_norm_distance)
