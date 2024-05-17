import numpy as np

# Vectors
x_35 = np.array([-1.24, -0.26, -1.04])
x_53 = np.array([-0.60, -0.86, -0.50])

# p-norm distance function
def p_norm_distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

# Euclidean Distance (p=2)
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

# Manhattan Distance (p=1)
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

# Chebyshev Distance (p=∞)
def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))

# Hamming Distance (for binary vectors)
def hamming_distance(x, y):
    return np.sum(x != y)

# Jaccard Similarity (for binary vectors)
def jaccard_similarity(x, y):
    intersection = np.sum(np.logical_and(x, y))
    union = np.sum(np.logical_or(x, y))
    return intersection / union

# Pearson Correlation
def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

# Calculate distances and similarities
d_p1 = p_norm_distance(x_35, x_53, 1)
d_p2 = euclidean_distance(x_35, x_53)
d_p4 = p_norm_distance(x_35, x_53, 4)
d_p_inf = chebyshev_distance(x_35, x_53)
cosine_similarity = np.dot(x_35, x_53) / (np.linalg.norm(x_35) * np.linalg.norm(x_53))
# Assuming binary vectors for Hamming and Jaccard calculations, converting continuous to binary
x_35_binary = np.array(x_35 > 0, dtype=int)
x_53_binary = np.array(x_53 > 0, dtype=int)
hamming_dist = hamming_distance(x_35_binary, x_53_binary)
jaccard_sim = jaccard_similarity(x_35_binary, x_53_binary)
pearson_corr = pearson_correlation(x_35, x_53)

# Print results
print(f"d_p1(x_35, x_53) = {d_p1:.2f} (Manhattan Distance)")
print(f"d_p2(x_35, x_53) = {d_p2:.2f} (Euclidean Distance)")
print(f"d_p4(x_35, x_53) = {d_p4:.2f}")
print(f"d_p_inf(x_35, x_53) = {d_p_inf:.2f} (Chebyshev Distance)")
print(f"cos(x_35, x_53) = {cosine_similarity:.2f} (Cosine Similarity)")
print(f"hamming_distance(x_35, x_53) = {hamming_dist} (Hamming Distance)")
print(f"jaccard_similarity(x_35, x_53) = {jaccard_sim:.2f} (Jaccard Similarity)")
print(f"pearson_correlation(x_35, x_53) = {pearson_corr:.2f} (Pearson Correlation)")
