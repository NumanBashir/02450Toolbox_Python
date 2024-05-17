import numpy as np

# Vectors
x_35 = np.array([-1.24, -0.26, -1.04])
x_53 = np.array([-0.60, -0.86, -0.50])

# p-norm distance function
def p_norm_distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

# Calculate p-norm distances
d_p1 = p_norm_distance(x_35, x_53, 1)
d_p4 = p_norm_distance(x_35, x_53, 4)
d_p_inf = np.max(np.abs(x_35 - x_53))

# Cosine similarity
cosine_similarity = np.dot(x_35, x_53) / (np.linalg.norm(x_35) * np.linalg.norm(x_53))

# Print results
print(f"d_p1(x_35, x_53) = {d_p1:.2f}")
print(f"d_p4(x_35, x_53) = {d_p4:.2f}")
print(f"d_p_inf(x_35, x_53) = {d_p_inf:.2f}")
print(f"cos(x_35, x_53) = {cosine_similarity:.2f}")
