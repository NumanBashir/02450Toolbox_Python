# Given values
p_x2_x10_given_y1 = 0.25
p_y1 = 0.316
p_x2_x10_given_y2 = 0.20
p_y2 = 0.356
p_x2_x10_given_y3 = 0.35
p_y3 = 0.328

# Total probability P(x2=1, x10=0)
p_x2_x10 = (p_x2_x10_given_y1 * p_y1) + (p_x2_x10_given_y2 * p_y2) + (p_x2_x10_given_y3 * p_y3)

# Bayes' theorem to find P(y=1 | x2=1, x10=0)
p_y1_given_x2_x10 = (p_x2_x10_given_y1 * p_y1) / p_x2_x10
print(p_y1_given_x2_x10)
