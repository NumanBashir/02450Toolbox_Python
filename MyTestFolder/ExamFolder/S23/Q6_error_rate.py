import numpy as np

# TODO DOES NOT WORK CHECK DOCS OF EXPLANATION!!!!!!

# Provided data
errors_outer_fold_1 = np.array([
    [0, 0, 1, 0, 0],  # Errors for K=1
    [1, 1, 1, 1, 1],  # Errors for K=3
    [1, 1, 1, 0, 0]   # Errors for K=4
])

errors_outer_fold_2 = np.array([
    [0, 0, 1, 0, 1],  # Errors for K=1
    [0, 0, 1, 0, 0],  # Errors for K=3
    [0, 0, 1, 0, 0]   # Errors for K=4
])

# Outer fold splits (indexing starts at 0 for Python arrays)
outer_fold_1_train = [0, 1, 3, 7, 8]
outer_fold_2_train = [2, 4, 5, 6, 9]

# Determine the best K based on the provided error matrices
# Since the prompt says to prefer the lowest value of K with equal performance,
# we choose K=1, 3, and 4 in that order of preference.
best_k_for_fold_1 = 1  # K=1 (since K=1 has the least error in fold 1)
best_k_for_fold_2 = 1  # K=1 (since K=1 has the least error in fold 2)

# Error rates for the best K in each fold
error_rate_fold_1 = np.mean(errors_outer_fold_1[0])
error_rate_fold_2 = np.mean(errors_outer_fold_2[0])

# Display the results
print("Error rates for the best K in each outer fold:")
print(f"Outer fold 1 (best K=1): {error_rate_fold_1}")
print(f"Outer fold 2 (best K=1): {error_rate_fold_2}")

# Choosing the correct answer based on calculated error rates
if error_rate_fold_1 == 0.2 and error_rate_fold_2 == 0.2:
    answer = "B. E_test i=1 = 0.2 and E_test i=2 = 0.2"
elif error_rate_fold_1 == 0.1 and error_rate_fold_2 == 0.3:
    answer = "A. E_test i=1 = 0.1 and E_test i=2 = 0.3"
elif error_rate_fold_1 == 0.8 and error_rate_fold_2 == 0.4:
    answer = "C. E_test i=1 = 0.8 and E_test i=2 = 0.4"
elif error_rate_fold_1 == 0.2 and error_rate_fold_2 == 0.6:
    answer = "D. E_test i=1 = 0.2 and E_test i=2 = 0.6"
else:
    answer = "None of the given options match the calculated error rates."

print(f"Correct answer: {answer}")
