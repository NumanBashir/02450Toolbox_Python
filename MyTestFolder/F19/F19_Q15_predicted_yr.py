# Given data from the table
xT = [-1.76, -0, 0.06, 0.08, 0.65, 1.3]
yT = [12, 6, 8, 10, 4, 2]

# To predict for x = 0.5, we follow the decision tree rules:
# First rule: x > 0.365
# Second rule (if the first rule is false): x > -0.88

# Predicted value for x = 0.5 following the decision tree
# Since 0.5 > 0.365, we follow the right branch, and since there's no further split, 
# we take the average of y values in the rightmost leaf.

# Identifying the indices of the samples in the rightmost leaf
indices = [i for i, x in enumerate(xT) if x > 0.365]

# Calculating the average of y values corresponding to these indices
predicted_y = sum(yT[i] for i in indices) / len(indices)

print(predicted_y)
