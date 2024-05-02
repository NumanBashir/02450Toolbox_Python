import numpy as np

# Texts
s1 = "the bag of words representation should not give you a hard time"
s2 = "remember the representation should be a vector"

# Convert texts into sets of words
words_s1 = set(s1.split())
words_s2 = set(s2.split())

# Number of unique words in each document
n1 = len(words_s1)
n2 = len(words_s2)

# Number of common words
f11 = len(words_s1.intersection(words_s2))

# Compute cosine similarity
cosine_similarity = f11 / (np.sqrt(n1) * np.sqrt(n2))

print(f"Cosine similarity between s1 and s2: {cosine_similarity:.6f}")
