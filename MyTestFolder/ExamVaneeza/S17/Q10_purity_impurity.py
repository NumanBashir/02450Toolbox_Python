def calculate_impurity(prob):
    """Calculate impurity based on the maximum class probability."""
    return 1 - prob

def calculate_purity_gain(total_samples, root_counts, split_counts):
    """
    Calculate purity gain for a given split.
    
    Parameters:
    - total_samples: Total number of samples in the dataset.
    - root_counts: Tuple (low, high) for the root node.
    - split_counts: List of tuples [(low, high), (low, high), ...] for each split branch.
    
    Returns:
    - purity_gain: The purity gain of the split.
    """
    # Calculate root impurity
    root_total = sum(root_counts)
    root_prob = max(root_counts) / root_total
    root_impurity = calculate_impurity(root_prob)
    
    # Calculate weighted impurity for each split
    weighted_impurity = 0
    for counts in split_counts:
        split_total = sum(counts)
        split_prob = max(counts) / split_total
        split_impurity = calculate_impurity(split_prob)
        weighted_impurity += (split_total / total_samples) * split_impurity
    
    # Calculate purity gain
    purity_gain = root_impurity - weighted_impurity
    return purity_gain

# Eksempelværdier
total_samples = 32
root_counts = (17, 15)  # (low, high)
#summen af gears low (17), er fundet ved 13+2+2 ( fra tabel 3)
#summen af gears high (15), er fundet ved 2+10+3 (fra tabel 3)

#skift ud ved kig på tabel man får givet


#disse er fået givet fra spg:

# Split A: 3 gears vs. 4 or 5 gears
split_A_counts = [(13, 2), (4, 13)]

# Split B: 3 or 4 gears vs. 5 gears
split_B_counts = [(15, 12), (2, 3)]

# Split C: 3 gears vs. 4 gears vs. 5 gears
split_C_counts = [(13, 2), (2, 10), (2, 3)]


# Calculate purity gains
purity_gain_A = calculate_purity_gain(total_samples, root_counts, split_A_counts)
purity_gain_B = calculate_purity_gain(total_samples, root_counts, split_B_counts)
purity_gain_C = calculate_purity_gain(total_samples, root_counts, split_C_counts)

print(f"Purity Gain for Split A: {purity_gain_A:.4f}")
print(f"Purity Gain for Split B: {purity_gain_B:.4f}")
print(f"Purity Gain for Split C: {purity_gain_C:.4f}")
