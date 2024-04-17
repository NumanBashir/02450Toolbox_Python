# Provided probabilities
probabilities = {
    (0, 0): [0.19, 0.3, 0.19],
    (0, 1): [0.22, 0.3, 0.26],
    (1, 0): [0.25, 0.2, 0.35],
    (1, 1): [0.34, 0.2, 0.2]
}

# Prior probabilities for each copyist
priors = [0.316, 0.356, 0.328]

# Calculate the probability that an observation was authored by copyist 1
# given that x2 = 1 and x10 = 0
# Bayes Theorem application
prob_x_given_y = probabilities[(1, 0)]  # Probabilities for x2=1 and x10=0
prob_y_given_x = [prob_x_given_y[i] * priors[i] for i in range(len(priors))]  # P(x|y)*P(y)
prob_x = sum(prob_y_given_x)  # Total probability of x
prob_y1_given_x = prob_y_given_x[0] / prob_x  # P(y=1|x) = P(x|y=1)*P(y=1) / P(x)

print(prob_y1_given_x)
