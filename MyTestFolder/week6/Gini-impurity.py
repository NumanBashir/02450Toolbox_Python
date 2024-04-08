## Fall 2013 question 14


# Given values from the image

# N(Parent)
total_subjects = 33+28+30+29
# Total survived
total_survived = 81

# Total not survived
total_not_survived = 225

# Subjects with positive axillary nodes
# N(left)
positive_nodes = 170

# Survived in N(left)
positive_survived = 62

positive_not_survived = positive_nodes - positive_survived

# Subjects without positive axillary nodes
# N(right)
negative_nodes = 136

# Survived in N(right)
negative_survived = 19

negative_not_survived = negative_nodes - negative_survived

# Function to calculate Gini impurity
def gini_impurity(survived, not_survived):
    total = survived + not_survived
    p_survived = survived / total if total else 0
    p_not_survived = not_survived / total if total else 0
    return 1 - p_survived**2 - p_not_survived**2

# Calculate Gini impurity for the entire dataset
I_t = gini_impurity(total_survived, total_not_survived)

# Calculate Gini impurity for subjects with positive axillary nodes
I_positive = gini_impurity(positive_survived, positive_not_survived)

# Calculate Gini impurity for subjects without positive axillary nodes
I_negative = gini_impurity(negative_survived, negative_not_survived)

# Calculate the gain
Delta = I_t - (positive_nodes / total_subjects) * I_positive - (negative_nodes / total_subjects) * I_negative

I_t, I_positive, I_negative, Delta

print("The gain is")
print(Delta)