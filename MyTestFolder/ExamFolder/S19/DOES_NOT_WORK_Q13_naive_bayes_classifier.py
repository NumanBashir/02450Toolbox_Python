import numpy as np
from fractions import Fraction

# Provided feature table (Table 4)
table4 = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # o1
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # o2
    [0, 1, 1, 1, 1, 1, 0, 0, 0],  # o3
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # o4
    [1, 0, 0, 1, 0, 0, 0, 0, 0],  # o5
    [0, 0, 1, 1, 0, 0, 0, 1, 0],  # o6
    [0, 0, 1, 1, 1, 0, 0, 0, 0],  # o7
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # o8
    [0, 1, 1, 0, 1, 0, 0, 0, 0],  # o9
    [0, 0, 1, 1, 1, 0, 1, 0, 0]   # o10
])

# Class labels
y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])

# Features for the given observation
f2 = 0
f4 = 1
f5 = 0

# Calculate class probabilities
classes, counts = np.unique(y, return_counts=True)
class_probs = counts / len(y)

# Function to calculate the conditional probability P(f|y)
def conditional_prob(feature_index, feature_value, class_label):
    class_indices = np.where(y == class_label)[0]
    feature_values = table4[class_indices, feature_index]
    return np.sum(feature_values == feature_value) / len(class_indices)

# Calculate the numerator for each class
numerators = []
for class_label in classes:
    p_f2_given_y = conditional_prob(1, f2, class_label)
    p_f4_given_y = conditional_prob(3, f4, class_label)
    p_f5_given_y = conditional_prob(4, f5, class_label)
    numerator = p_f2_given_y * p_f4_given_y * p_f5_given_y * class_probs[class_label]
    numerators.append(numerator)

# Calculate the denominator
denominator = sum(numerators)

# Calculate the probability P(y=2 | f2=0, f4=1, f5=0)
p_y2_given_f2_f4_f5 = numerators[2] / denominator

# Scale up to avoid floating point errors
scale = 10000  # scaling factor to convert to integer
numerator_int = int(numerators[2] * scale)
denominator_int = int(denominator * scale)

# Convert to fraction format
prob_fraction = Fraction(numerator_int, denominator_int).limit_denominator()

print(f"P(y=2 | f2=0, f4=1, f5=0): {p_y2_given_f2_f4_f5:.4f}")
print(f"As a fraction: {prob_fraction}")
