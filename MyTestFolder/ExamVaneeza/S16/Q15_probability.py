def calculate_conditional_probability(y, probabilities):
    """
    Calculate p(g2=1 | y=1) given the conditional probabilities.

    Parameters:
    y (float): Probability of y=1
    probabilities (dict): A dictionary containing conditional probabilities
                          with keys as tuples (g1, g2, y).

    Returns:
    float: The calculated probability p(g2=1 | y=1)
    """
    p_g2_1_y_1 = probabilities[(0, 1, 1)] + probabilities[(1, 1, 1)]
    return p_g2_1_y_1

# Example probabilities provided in the question
probabilities = {
    (0, 0, 0): 0.23,
    (0, 1, 0): 0.40,
    (1, 0, 0): 0.28,
    (1, 1, 0): 0.09,
    (0, 0, 1): 0.01,
    (0, 1, 1): 0.03,
    (1, 0, 1): 0.46,
    (1, 1, 1): 0.50
}

# Probability of y = 1
p_y_1 = 0.5

# Calculate the desired probability
p_g2_1_given_y_1 = calculate_conditional_probability(p_y_1, probabilities)
print(f"p(g2=1 | y=1) = {p_g2_1_given_y_1:.2f}")
