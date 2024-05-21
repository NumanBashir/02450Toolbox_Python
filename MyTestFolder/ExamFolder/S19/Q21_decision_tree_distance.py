import numpy as np

x = np.array([3, 3]) # CHOOSE A POINT TO VERIFY FROM (TAKE SOMETHING THAT COULD BE SUSPICIOUS)
b = np.array([6, 0]) # CHOOSE A POINT TO VERIFY AGAINST FROM THE QUESTION (THE DECISION NODE, A, B, C, D)

# Calculate L1 norm (Manhattan distance p=1)
L1_distance = np.sum(np.abs(x - b))

# Calculate L2 norm (Euclidean distance p=2)
L2_distance = np.sqrt(np.sum((x - b)**2))

print(f"L1 (Manhattan) distance: {L1_distance}")
print(f"L2 (Euclidean) distance: {L2_distance}")
