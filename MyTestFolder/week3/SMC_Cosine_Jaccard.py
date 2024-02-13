import numpy as np

# Transcribed dataset from the image
data = np.array([
    [1, 1, 1, 0, 0],  # Transaction 1
    [0, 1, 1, 0, 1]   # Transaction 2
])

# Define functions to calculate cosine similarity, SMC, and Jaccard coefficient
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def simple_matching_coefficient(x, y):
    return np.sum(x == y) / len(x)

def jaccard_coefficient(x, y):
    intersection = np.sum(np.logical_and(x, y))
    union = np.sum(np.logical_or(x, y))
    return intersection / union

# Calculate metrics for the two transactions
cos_sim = cosine_similarity(data[0], data[1])
smc = simple_matching_coefficient(data[0], data[1])
jaccard = jaccard_coefficient(data[0], data[1])

print(cos_sim, smc, jaccard)

print(cos_sim.as_integer_ratio())
