import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Sample dataset: List of transactions (each transaction is a list of items bought)
dataset = [['Milk', 'Bread', 'Butter'],
           ['Beer', 'Bread'],
           ['Milk', 'Bread', 'Butter', 'Beer'],
           ['Milk', 'Bread', 'Beer']]

# Encoding dataset for mlxtend
encoder = TransactionEncoder()
encoded_array = encoder.fit(dataset).transform(dataset)
df = pd.DataFrame(encoded_array, columns=encoder.columns_)

# Finding frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Generating association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Displaying generated rules
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
