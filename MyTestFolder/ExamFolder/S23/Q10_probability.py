# Given probabilities
p_x2_1_x4_0_machine = 0.08
p_x2_1_x4_1_machine = 0.10
p_x2_1_x4_0_natural = 0.16
p_x2_1_x4_1_natural = 0.16

# Calculate p(x2 = 1)
p_x2_1 = p_x2_1_x4_0_machine + p_x2_1_x4_0_natural + p_x2_1_x4_1_machine + p_x2_1_x4_1_natural

# Calculate p(y = Machine | x2 = 1)
p_y_machine_given_x2_1 = (p_x2_1_x4_0_machine + p_x2_1_x4_1_machine) / p_x2_1

print(f"p(y = Machine | x2 = 1) = {p_y_machine_given_x2_1:.3f}")