import numpy as np
from itertools import combinations

#kig på confidence reglerne som den beregner alle ud, også giver den svar som du skal match for se hvilken det er af regler.

# Data fra tabellen (binary matrix)
data = np.array([
    [0, 1, 1, 0, 1],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# Funktion til at beregne support for et givet itemset
def calculate_support(data, itemset):
    itemset = list(itemset)
    count = np.sum(np.all(data[:, itemset] == 1, axis=1))
    return count

# Funktion til at beregne confidence for en given regel
def calculate_confidence(data, antecedent, consequent):
    union = antecedent + consequent
    support_union = calculate_support(data, union)
    support_antecedent = calculate_support(data, antecedent)
    confidence = support_union / support_antecedent
    return confidence

# Definer reglerne
rules = [
    ([2, 3], [4]),  # {f3, f4} -> {f5}
    ([0, 4], [3]),  # {f1, f5} -> {f4}
    ([0, 3], [4]),  # {f1, f4} -> {f5}
    ([1, 3], [0])   # {f2, f4} -> {f1}
]

# Beregn confidence for hver regel
confidences = [calculate_confidence(data, antecedent, consequent) for antecedent, consequent in rules]

# Udskriv resultaterne
for i, confidence in enumerate(confidences):
    print(f"Confidence for rule {i+1}: {confidence:.6f}")

# Find den regel med den højeste confidence
max_confidence = max(confidences)
best_rule_index = confidences.index(max_confidence) + 1

print(f"\nThe rule with the highest confidence is rule {best_rule_index} with a confidence of {max_confidence:.6f}")
