def max_hidden_units(total_budget, outer_folds, inner_folds, models_per_spec):
    max_H = total_budget // (outer_folds * inner_folds * models_per_spec + 1)
    return max_H

# Parameters based on the problem description
total_budget = 1000
outer_folds = 5
inner_folds = 10
models_per_spec = 3  # three models for each specification of hidden units

# Calculate the maximum number of hidden units
max_H = max_hidden_units(total_budget, outer_folds, inner_folds, models_per_spec)
print(f"The largest value of H for which no more than 1000 models will be trained is: {max_H}")
