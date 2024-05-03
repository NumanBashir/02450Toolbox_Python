from itertools import combinations

def matrix_to_frequent_itemsets(matrix, items):
    """
    Converts a matrix indicating the presence of itemsets into a list of sets.
    Each row in the matrix corresponds to an itemset, with 1s marking the items included.
    """
    frequent_itemsets = []
    for row in matrix:
        itemset = {items[i] for i, val in enumerate(row) if val == 1}
        if len(itemset) == 2:  # We only want itemsets of size 2 for L2
            frequent_itemsets.append(itemset)
    return frequent_itemsets

def all_subsets_frequent(candidate, L2):
    """
    Checks if all non-empty subsets of a candidate are in the list of frequent itemsets L2.
    """
    candidate_subsets = list(combinations(candidate, 2))
    return all(set(subset) in L2 for subset in candidate_subsets)

# L2 matrix input, rows represent different itemsets with presence marked by 1
#TODO: Update the matrix based on the task
L2_matrix = [
    [1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 1]
]

# Corresponding items for indices
#TODO, add amount of f's based on the task
items = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

# Convert the L2 matrix to a list of frequent itemsets
L2_frequent = matrix_to_frequent_itemsets(L2_matrix, items)

# Define the options for L3 candidates
#TODO: add the options
options = {
    'A': ['f2', 'f3', 'f4'],
    'B': ['f1', 'f2', 'f6'],
    'C': ['f2', 'f3', 'f6'],
    'D': ['f1', 'f3', 'f4']
}

# Determine which itemsets do not require their support to be evaluated
not_evaluated = []
for option, itemset in options.items():
    if not all_subsets_frequent(itemset, L2_frequent):
        not_evaluated.append(option)

# Determine which itemsets do not require their support to be evaluated and print the outcome
for option, itemset in options.items():
    if all_subsets_frequent(itemset, L2_frequent):
        print(f"Option {option} ({', '.join(itemset)}) does have to evaluate the support of.")
    else:
        print(f"Option {option} ({', '.join(itemset)}) does not have to evaluate the support of.")