# Number of observations for each class
n_adelie = 146
n_gentoo = 119
n_chinstrap = 68

# Total number of observations
n_total = n_adelie + n_gentoo + n_chinstrap

# Classification error for the parent node
class_error_parent = 1 - max(n_adelie, n_gentoo, n_chinstrap) / n_total

# Number of observations in the left child node
n_left = n_adelie + n_gentoo

# Classification error for the left child node
class_error_left = 1 - max(n_adelie, n_gentoo) / n_left

# Number of observations in the right child node
n_right = n_chinstrap

# Classification error for the right child node
class_error_right = 1 - n_chinstrap / n_right

# Purity gain for the split
purity_gain = class_error_parent - (n_left / n_total * class_error_left + n_right / n_total * class_error_right)

print(f"The purity gain for the split is {purity_gain}")