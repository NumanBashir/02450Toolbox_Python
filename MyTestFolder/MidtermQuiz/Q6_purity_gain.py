# Define a function to calculate impurity based on given fractions
def impurity(fractions):
    return 1 - max(fractions, default=0)

# Calculate the impurity of the parent node
parent_fractions = [5/25, 10/25, 10/25]  # fractions for Setosa, Virginica, and Versicolor respectively
parent_impurity = impurity(parent_fractions)

# Calculate impurity of the child nodes after the split
left_fractions = [0, 8/10, 2/10]  # fractions for Setosa, Virginica, and Versicolor in the left node
right_fractions = [5/15, 2/15, 8/15]  # fractions for Setosa, Virginica, and Versicolor in the right node
left_impurity = impurity(left_fractions)
right_impurity = impurity(right_fractions)

# Number of samples in each child node
N_left = 10  # Total number in the left node
N_right = 15  # Total number in the right node
N_total = N_left + N_right  # Total number in the parent node

# Calculate the purity gain
purity_gain = parent_impurity - (N_left / N_total * left_impurity + N_right / N_total * right_impurity)
print(purity_gain)
