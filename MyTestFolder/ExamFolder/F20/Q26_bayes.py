# Probabilities
p_penguin = 0.01
p_not_penguin = 1 - p_penguin
p_classified_as_penguin_given_penguin = 0.97
p_classified_as_penguin_given_not_penguin = 0.03

# Probability that an image is classified as a penguin
p_classified_as_penguin = p_classified_as_penguin_given_penguin * p_penguin + p_classified_as_penguin_given_not_penguin * p_not_penguin

# Probability that an image contains a penguin given that it is classified as a penguin
p_penguin_given_classified_as_penguin = p_classified_as_penguin_given_penguin * p_penguin / p_classified_as_penguin

print(f"The probability that a random image from this dataset contains a penguin given that it is classified as a penguin is {p_penguin_given_classified_as_penguin}")