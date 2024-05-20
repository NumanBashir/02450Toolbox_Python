import numpy as np

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

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def calculate_smc(vec1, vec2):
    matches = np.sum(vec1 == vec2)
    return matches / len(vec1)

def calculate_jaccard_similarity(vec1, vec2):
    positive_matches = np.sum(np.logical_and(vec1, vec2))
    negative_matches = np.sum(np.logical_and(np.logical_not(vec1), np.logical_not(vec2)))
    return positive_matches / (len(vec1) - negative_matches)

# Replace these with the actual binary vectors for A and B
# vector_a = np.array([1,0,1,0,1,0]) #Feel free to write which observation for better overview eg. S1
# vector_b = np.array([1,0,1,0,0,1])

vector_a = np.array([1,0,1,0,0,1]) #Feel free to write which observation for better overview eg. S1
vector_b = np.array([1,0,1,0,1,0])

cosine_similarity = calculate_cosine_similarity(vector_a, vector_b)
smc = calculate_smc(vector_a, vector_b)
jaccard_similarity = calculate_jaccard_similarity(vector_a, vector_b)
manhattan_distance_value = manhattan_distance(vector_a, vector_b)
euclidean_distance_value = euclidean_distance(vector_a, vector_b)

print(f"Cosine Similarity: {cosine_similarity}")
print(f"Simple Matching Coefficient: {smc}")
print(f"Jaccard Similarity: {jaccard_similarity}")
print(f"Manhattan Distance: {manhattan_distance_value}")
print(f"Euclidean Distance: {euclidean_distance_value}")
