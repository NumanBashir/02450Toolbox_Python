import numpy as np # type: ignore

# Given data
points = np.array([
    [-0.4, -0.8],  # o1
    [-0.9, 0.3],   # o2
    [0, 0.9],      # o3
    [1, -0.1],     # o4
    [0.8, -0.7],   # o5
    [0.1, 0.8]     # o6
])
classes = np.array(['C1', 'C1', 'C1', 'C2', 'C2', 'C2'])

# Test point
test_point = np.array([0, 0])

# Distance calculations
d2 = np.linalg.norm(points - test_point, axis=1)
d1 = np.sum(np.abs(points - test_point), axis=1)
d_inf = np.max(np.abs(points - test_point), axis=1)

# Get the indices of the 3 nearest neighbors for each distance metric
nn_d2 = np.argsort(d2)[:3]
nn_d1 = np.argsort(d1)[:3]
nn_d_inf = np.argsort(d_inf)[:3]

# Get the classes of the 3 nearest neighbors for each distance metric
classes_d2 = classes[nn_d2]
classes_d1 = classes[nn_d1]
classes_d_inf = classes[nn_d_inf]

# Determine the majority class for each distance metric
from collections import Counter

majority_class_d2 = Counter(classes_d2).most_common(1)[0][0]
majority_class_d1 = Counter(classes_d1).most_common(1)[0][0]
majority_class_d_inf = Counter(classes_d_inf).most_common(1)[0][0]

# Print results
print(f"Nearest neighbors for d2: {classes_d2}, Majority class: {majority_class_d2}")
print(f"Nearest neighbors for d1: {classes_d1}, Majority class: {majority_class_d1}")
print(f"Nearest neighbors for d_inf: {classes_d_inf}, Majority class: {majority_class_d_inf}")

# Check which distance metrics predict the same label
same_label_d1_d_inf = majority_class_d1 == majority_class_d_inf
same_label_d2_d1 = majority_class_d2 == majority_class_d1
same_label_d2_d_inf = majority_class_d2 == majority_class_d_inf

print(f"d1 and d_inf predict the same label: {same_label_d1_d_inf}")
print(f"d2 and d1 predict the same label: {same_label_d2_d1}")
print(f"d2 and d_inf predict the same label: {same_label_d2_d_inf}")
