def calculate_rand_index():
    # Define the cluster indices and true class labels
    cluster_indices = [1, 2, 1, 1, 1, 3, 1, 1, 1, 1]
    true_labels = [1, 2, 1, 1, 1, 2, 2, 2, 2, 1]

    # Initialize counts
    f11 = 0  # same class, same cluster
    f00 = 0  # different class, different cluster
    N = len(cluster_indices)

    # Calculate f11 and f00
    for i in range(N):
        for j in range(i + 1, N):
            same_cluster = cluster_indices[i] == cluster_indices[j]
            same_class = true_labels[i] == true_labels[j]

            if same_class and same_cluster:
                f11 += 1
            elif not same_class and not same_cluster:
                f00 += 1

    # Total pairs K
    K = N * (N - 1) / 2

    # Calculate Rand Index
    R = (f11 + f00) / K
    return R

# Call the function and print the Rand Index
rand_index = calculate_rand_index()
print(f"The Rand Index is: {rand_index:.4f}")

