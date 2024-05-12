import numpy as np # type: ignore

def calculate_gini(class_counts):
    """Calculate the Gini impurity for a given node based on class counts."""
    total = sum(class_counts)
    if total == 0:
        return 0
    sum_squared_probs = sum((count / total) ** 2 for count in class_counts)
    return 1 - sum_squared_probs

def calculate_purity_gain_gini(I0, splits, n):
    """Calculate the purity gain from a split using the Gini impurity."""
    weighted_impurity = sum((sum(counts) / n) * calculate_gini(counts) for counts in splits)
    return I0 - weighted_impurity

# Total number of observations
n = 18 + 18 + 18  # Total observations

# Initial impurity (Gini impurity) for the root node
initial_counts = [18, 18, 18]  # Even distribution among the classes
I0 = calculate_gini(initial_counts)

# Define the splits as lists of class counts for each split
split1 = [(6, 9, 3), (4, 6, 10), (8, 3, 5)]

# Calculate purity gains for each split using Gini impurity
purity_gain = calculate_purity_gain_gini(I0, split1, n)

print("Purity gain for the split based on Gini impurity:", purity_gain)
