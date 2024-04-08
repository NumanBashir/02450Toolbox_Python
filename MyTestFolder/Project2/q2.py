# Definitions for the calculation
parent_total = 120
left_total = parent_total - 5
right_total = 5
parent_majority_class = 33

# Parent node impurity
parent_impurity = 1 - parent_majority_class / parent_total

# Left child node impurity (x7 != 2)
left_majority_class = 33  # assuming the majority class remains the same
left_impurity = 1 - left_majority_class / left_total

# Right child node impurity (x7 = 2), which is pure and hence impurity is 0
right_impurity = 0

# The weights for the child nodes are proportional to the number of samples in each node
weighted_left_impurity = (left_total / parent_total) * left_impurity
weighted_right_impurity = (right_total / parent_total) * right_impurity

# Impurity gain is the parent impurity minus the weighted sum of the child nodes' impurities
impurity_gain = parent_impurity - (weighted_left_impurity + weighted_right_impurity)

parent_impurity, left_impurity, right_impurity, weighted_left_impurity, weighted_right_impurity, impurity_gain

print(impurity_gain)