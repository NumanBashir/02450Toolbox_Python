# Given data
prob_well_rested = 0.8
prob_not_well_rested = 0.2  # which is 1 - prob_well_rested
prob_pass_if_well_rested = 0.9
prob_pass_if_not_well_rested = 0.4

# Total probability of passing the exam
prob_pass = (prob_well_rested * prob_pass_if_well_rested) + (prob_not_well_rested * prob_pass_if_not_well_rested)

# Applying Bayes' theorem to find the probability of being well rested given that a person has passed the exam
prob_well_rested_given_pass = (prob_well_rested * prob_pass_if_well_rested) / prob_pass
print(prob_well_rested_given_pass)
