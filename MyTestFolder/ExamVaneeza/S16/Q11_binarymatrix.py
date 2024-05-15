import numpy as np
from itertools import combinations

# Data fra tabellen (binary matrix)
data = np.array([
    [0, 1, 1, 0, 1],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# Antal transaktioner
n_transactions = data.shape[0]

# Funktion til at beregne support for et givet itemset
def calculate_support(data, itemset):
    itemset = list(itemset)
    count = np.sum(np.all(data[:, itemset] == 1, axis=1))
    return count / n_transactions

# Funktion til at finde alle itemsets med support større end threshold
def find_frequent_itemsets(data, threshold=0.32): # Skift talled "threshold" ud med det du får givet. vi tjekker om større nd 0.32
    n_items = data.shape[1]
    frequent_itemsets = []
    
    # Tjek alle individuelle items
    for item in range(n_items):
        support = calculate_support(data, [item])
        if support > threshold:
            frequent_itemsets.append(({item}, support))
    
    # Tjek alle kombinationer af items
    for size in range(2, n_items + 1):
        for itemset in combinations(range(n_items), size):
            support = calculate_support(data, itemset)
            if support > threshold:
                frequent_itemsets.append((set(itemset), support))
    
    return frequent_itemsets

# Find alle itemsets med support større end 0.32
threshold = 0.32 # husk skifte ud
frequent_itemsets = find_frequent_itemsets(data, threshold)

# Udskriv resultaterne
for itemset, support in frequent_itemsets:
    print(f"Itemset: {itemset}, Support: {support:.2f}")

# Kortlæg items til deres navne
item_names = ['f1', 'f2', 'f3', 'f4', 'f5']
frequent_itemsets_named = [{item_names[item] for item in itemset} for itemset, _ in frequent_itemsets]

print("\nFrequent itemsets with support greater than 0.32:")
print(frequent_itemsets_named)
