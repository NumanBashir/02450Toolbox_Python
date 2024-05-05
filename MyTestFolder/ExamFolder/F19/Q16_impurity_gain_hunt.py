import numpy as np

def calculate_purity_gain(y_values, split_rule):
    """
    Calculate the purity gain for a given split based on a decision rule.
    
    Parameters:
    y_values (list): The list of target values in the dataset.
    split_rule (float): The threshold for the decision rule (e.g., x > split_rule).

    Returns:
    float: The purity gain of the split.
    """
    # Separate the y_values based on the split rule
    group1 = [y for i, y in enumerate(y_values) if x_values[i] <= split_rule]
    group2 = [y for i, y in enumerate(y_values) if x_values[i] > split_rule]

    # Calculate the overall impurity before the split
    I_v0 = np.var(y_values, ddof=1)

    # Calculate the impurity of each group
    I_v1 = np.var(group1, ddof=1) if group1 else 0  # Avoid division by zero if a group is empty
    I_v2 = np.var(group2, ddof=1) if group2 else 0

    # Calculate the weighted average impurity after the split
    N_v0 = len(y_values)
    N_v1 = len(group1)
    N_v2 = len(group2)
    weighted_impurity = (N_v1 / N_v0) * I_v1 + (N_v2 / N_v0) * I_v2

    # The purity gain is the initial impurity minus the weighted average impurity after the split
    purity_gain = I_v0 - weighted_impurity

    return purity_gain

# Given x and y values from the dataset
x_values = [-1.76, -0, 0.06, 0.08, 0.65, 1.3]  # x values from the dataset
y_values = [12, 6, 8, 10, 4, 2]  # y values from the dataset
split_rule = 0.365  # The first split rule

# Calculate the purity gain
purity_gain = calculate_purity_gain(y_values, split_rule)
print(f"The purity gain for the split rule x > {split_rule} is: {purity_gain:.2f}")
