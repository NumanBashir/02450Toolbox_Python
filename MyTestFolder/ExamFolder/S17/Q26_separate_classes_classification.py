import numpy as np

# Define the point to check
x = np.array([0.5, 0.6])
c1 = np.array([0, 1])
c2 = np.array([1, 0])
c3 = np.array([1, 1])

# Euclidean Distance (p=2)
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

# Manhattan Distance (p=1)
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

# Chebyshev Distance (p=∞)
def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))

# Calculate distances for the random point
d1_euc = euclidean_distance(x, c1)
d2_euc = euclidean_distance(x, c2)
d3_euc = euclidean_distance(x, c3)

d1_man = manhattan_distance(x, c1)
d2_man = manhattan_distance(x, c2)
d3_man = manhattan_distance(x, c3)

d1_che = chebyshev_distance(x, c1)
d2_che = chebyshev_distance(x, c2)
d3_che = chebyshev_distance(x, c3)

print(f"Euclidean Distance to c1: {d1_euc:.2f}")
print(f"Euclidean Distance to c2: {d2_euc:.2f}")
print(f"Euclidean Distance to c3: {d3_euc:.2f}")
print(f"Manhattan Distance to c1: {d1_man:.2f}")
print(f"Manhattan Distance to c2: {d2_man:.2f}")
print(f"Manhattan Distance to c3: {d3_man:.2f}")
print(f"Chebyshev Distance to c1: {d1_che:.2f}")
print(f"Chebyshev Distance to c2: {d2_che:.2f}")
print(f"Chebyshev Distance to c3: {d3_che:.2f}")

# Check Option A
option_A = d1_euc > 0.5 and d2_che > 0.5 and d3_man > 0.75
print(f"Option A result: {option_A}")

# Check Option B
option_B = d1_che > 0.5 and d2_euc > 0.5 and d3_man > 0.75
print(f"Option B result: {option_B}")

# Check Option C
option_C = d1_euc > 0.5 and d2_man > 0.5 and d3_che > 0.75
print(f"Option C result: {option_C}")

# Check Option D
option_D = d1_che > 0.5 and d2_man > 0.5 and d3_che > 0.75
print(f"Option D result: {option_D}")
