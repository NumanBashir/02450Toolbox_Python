# Given probabilities
P_four_doors = 0.5
P_heavy_given_four = 0.8
P_heavy_given_two = 0.2
P_heavy_and_red = 0.1

# The probability that a car has two doors is the complement of it having four doors
P_two_doors = 1 - P_four_doors

# Calculate the total probability that a car is heavy
P_heavy = (P_heavy_given_four * P_four_doors) + (P_heavy_given_two * P_two_doors)

# Calculate the probability that a car is heavy and blue
# Since a car can only be red or blue, the probability of a heavy car being blue is the complement of it being red
P_heavy_and_blue = P_heavy - P_heavy_and_red

# Finally, calculate the probability that a car is blue given that it is heavy
P_blue_given_heavy = P_heavy_and_blue / P_heavy

print(P_blue_given_heavy)
