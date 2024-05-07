import numpy as np

# Define the binarized dataset with corrected understanding or labels if necessary
data = np.array([
    [0, 0, 0, 1, 0, 0, 0, 1],  # o1 -> C1
    [0, 0, 1, 0, 0, 1, 0, 1],  # o2 -> C1
    [0, 0, 1, 0, 0, 1, 0, 1],  # o3 -> C2
    [0, 1, 0, 0, 0, 1, 0, 1],  # o4 -> C2
    [0, 0, 0, 0, 0, 1, 0, 1],  # o5 -> C2
    [0, 0, 1, 0, 1, 1, 0, 1],  # o6 -> C2
    [0, 0, 1, 0, 0, 1, 0, 1],  # o7 -> C2
    [1, 1, 0, 0, 0, 0, 1, 1],  # o8 -> C3
    [0, 1, 0, 0, 0, 0, 0, 1],  # o9 -> C3
    [0, 1, 0, 0, 0, 1, 0, 1],  # o10 -> C3
    [1, 1, 0, 0, 0, 0, 0, 0],  # o11 -> C3
])

# Define the class labels for the data
classes = np.array([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3])  # Classes corresponding to each observation

# Filter the data where f1 = 0 and f6 = 1
mask_f1_0_f6_1 = (data[:, 0] == 0) & (data[:, 5] == 1)

# Filtered class labels for the condition f1 = 0 and f6 = 1
filtered_classes = classes[mask_f1_0_f6_1]

# Count the occurrences of class C2 in the filtered dataset
count_C2 = np.sum(filtered_classes == 2)

# Total number of samples meeting the condition f1 = 0 and f6 = 1
total_filtered = len(filtered_classes)

# Calculate the conditional probability p(C2 | f1 = 0, f6 = 1)
conditional_probability = count_C2 / total_filtered if total_filtered > 0 else 0
print(conditional_probability, count_C2, total_filtered)
