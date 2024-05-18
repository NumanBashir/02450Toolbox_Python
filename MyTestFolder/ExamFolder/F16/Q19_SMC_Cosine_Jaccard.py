import numpy as np

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

vector_a = np.array([1,1,1,1,0,1]) #Feel free to write which observation for better overview eg. S1
vector_b = np.array([1,1,0,1,0,0])

cosine_similarity = calculate_cosine_similarity(vector_a, vector_b)
smc = calculate_smc(vector_a, vector_b)
jaccard_similarity = calculate_jaccard_similarity(vector_a, vector_b)

print(f"Cosine Similarity: {cosine_similarity}")
print(f"Simple Matching Coefficient: {smc}")
print(f"Jaccard Similarity: {jaccard_similarity}")
