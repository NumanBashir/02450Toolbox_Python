# Calculating the impurities and purity gain
n_safe = 32
n_unsafe = 24
total = n_safe + n_unsafe

# Impurity before the split
p_safe_before = n_safe / total
I_root = 1 - p_safe_before

# Node 1
n1_safe = 23
n1_unsafe = 8
n1_total = n1_safe + n1_unsafe
p1_safe = n1_safe / n1_total
I1 = 1 - p1_safe

# Node 2
n2_safe = 9
n2_unsafe = 16
n2_total = n2_safe + n2_unsafe
p2_unsafe = n2_unsafe / n2_total
I2 = 1 - p2_unsafe

# Weighted impurity after the split
I_split = (n1_total / total * I1) + (n2_total / total * I2)

# Purity gain
delta = I_root - I_split
I_root, I1, I2, I_split, delta

print(delta)