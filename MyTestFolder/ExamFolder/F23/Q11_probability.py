def calculate_probability(table, priors, x1_value):
    """
    Calculate the conditional probability p(y = High | x1 = x1_value)

    :param table: A dictionary of the form {('x1', 'x3'): {'Low': probability, 'High': probability}, ...}
    :param priors: A dictionary of prior probabilities {'Low': probability, 'High': probability}
    :param x1_value: The value of x1 to condition on
    :return: The conditional probability p(y = High | x1 = x1_value)
    """
    # Calculate the class conditionals for x1
    p_x1_y = {y: sum(prob[y] for (x1, _), prob in table.items() if x1 == x1_value) for y in priors}

    # Calculate the marginal probability of x1
    p_x1 = sum(p_x1_y[y] * priors[y] for y in priors)

    # Calculate the conditional probability p(y = High | x1 = x1_value)
    p_y_given_x1 = p_x1_y['High'] * priors['High'] / p_x1

    return p_y_given_x1

# Example usage
table = {
    (0, 0): {'Low': 0.04, 'High': 0.25},
    (0, 1): {'Low': 0.03, 'High': 0.68},
    (1, 0): {'Low': 0.66, 'High': 0.02},
    (1, 1): {'Low': 0.27, 'High': 0.05}
}
#ændrer matricen 

priors = {'Low': 0.53, 'High': 0.47}
x1_value = 0
#could be another x1 så kig efter hvad de spørger om 

# Calculate the probability
probability = calculate_probability(table, priors, x1_value)
print(f"p(y = High | x1 = {x1_value}) = {probability:.2f}")

