import numpy as np

# Define the binarized dataset with corrected understanding or labels if necessary
binarized_data = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0],  # o1 C1
    [0, 0, 1, 0, 0, 1, 0, 0],  # o2 C1
    [0, 0, 1, 0, 0, 1, 0, 1],  # o3 C2 (Calabria)
    [0, 1, 0, 0, 0, 1, 0, 1],  # o4 C2 (Calabria)
    [0, 0, 0, 0, 0, 1, 0, 1],  # o5 C2 (Calabria)
    [0, 0, 1, 0, 1, 1, 0, 1],  # o6 C2 (Calabria)
    [0, 0, 1, 0, 0, 1, 0, 1],  # o7 C2 (Calabria)
    [1, 1, 0, 0, 0, 0, 1, 0],  # o8 C3
    [0, 1, 0, 0, 0, 0, 0, 0],  # o9 C3
    [0, 1, 0, 0, 0, 1, 0, 0],  # o10 C3
    [1, 1, 0, 0, 0, 0, 0, 0],  # o11 C3
])

# Variables to filter the data for f1 = 0 and f6 = 1
f1_index = 0
f6_index = 5
class_c2_index = 7

# Filter for observations where f1 is 0 and f6 is 1
indices = np.where((binarized_data[:, f1_index] == 0) & (binarized_data[:, f6_index] == 1))[0]

# Number of these observations
total_with_f1_0_and_f6_1 = len(indices)

# Count how many of these are from Calabria
count_c2 = np.sum(binarized_data[indices, class_c2_index] == 1)

# Conditional probability calculation
conditional_probability = count_c2 / total_with_f1_0_and_f6_1 if total_with_f1_0_and_f6_1 > 0 else 0

# Output results
print("Total with f1 = 0 and f6 = 1:", total_with_f1_0_and_f6_1)
print("Count from Calabria (C2) with f1 = 0 and f6 = 1:", count_c2)
print("Conditional Probability p(C2=1 | f1=0, f6=1):", conditional_probability)
