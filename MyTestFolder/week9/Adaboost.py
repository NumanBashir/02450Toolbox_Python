# Given values
import numpy as np


initial_weight = 1/6  # because we have 8 observations (1 / N) N --> Number of observations
misclassified = 2  # 2 points are misclassified
correctly_classified = 6 - misclassified

# Weighted error calculation
epsilon = misclassified * initial_weight

# Alpha calculation
alpha = 0.5 * np.log((1 - epsilon) / epsilon)

# Updating weights
# For correctly classified
updated_weights_correct = initial_weight * np.exp(-alpha)

# For misclassified
updated_weights_incorrect = initial_weight * np.exp(alpha)

# Now we normalize the weights so that they sum up to 1
sum_weights = (correctly_classified * updated_weights_correct) + \
              (misclassified * updated_weights_incorrect)

# Normalized weights
normalized_weights_correct = updated_weights_correct / sum_weights
normalized_weights_incorrect = updated_weights_incorrect / sum_weights

# Create the array with updated weights for all points
new_weights = np.array([normalized_weights_correct if i < correctly_classified
                        else normalized_weights_incorrect
                        for i in range(6)])

epsilon, alpha, new_weights

print(new_weights)

# Rækkefølgen er underordnet fordi vi ikke kender rækkefølgen af punkterne, bare kig på tallene