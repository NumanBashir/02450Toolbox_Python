import numpy as np

def classification_error(p):
    """
    Calculate the classification error.
    
    Parameters:
    p (float): Probability of the majority class.
    
    Returns:
    float: Classification error.
    """
    return 1 - p

def purity_gain(n, splits):
    """
    Calculate the purity gain for a given split.
    
    Parameters:
    n (int): Total number of observations.
    splits (list of tuples): Each tuple contains the number of observations 
                             (y=0, y=1) in a split.
                             
    Returns:
    float: Purity gain for the split.
    """
    # Calculate initial impurity I0 for the entire dataset
    total_y0 = sum(split[0] for split in splits)
    total_y1 = sum(split[1] for split in splits)
    total_p_y0 = total_y0 / n
    total_p_y1 = total_y1 / n
    I0 = classification_error(max(total_p_y0, total_p_y1))
    
    weighted_impurity = 0
    for split in splits:
        n_split = sum(split)
        if n_split == 0:
            continue
        p_y0 = split[0] / n_split
        p_y1 = split[1] / n_split
        split_impurity = classification_error(max(p_y0, p_y1))
        weighted_impurity += (n_split / n) * split_impurity
    
    return I0 - weighted_impurity

# Total number of observations
n = 200

# Splits as given in the table
splits_1 = [(45, 1), (47, 66), (8, 33)]
splits_2 = [(76, 20), (16, 47), (8, 33)]
splits_3 = [(25, 0), (55, 23), (20, 77)]

# Calculate purity gain for each split
purity_gain_1 = purity_gain(n, splits_1)
purity_gain_2 = purity_gain(n, splits_2)
purity_gain_3 = purity_gain(n, splits_3)

print(f"Purity gain for split 1: {purity_gain_1:.4f}")
print(f"Purity gain for split 2: {purity_gain_2:.4f}")
print(f"Purity gain for split 3: {purity_gain_3:.4f}")

# Determine the best split
best_split = max(purity_gain_1, purity_gain_2, purity_gain_3)
if best_split == purity_gain_1:
    print("Hunt's algorithm will select split 1.")
elif best_split == purity_gain_2:
    print("Hunt's algorithm will select split 2.")
else:
    print("Hunt's algorithm will select split 3.")
