# Given data
dist_o5_o4 = 0.1  # Distance from o5 to o4
dist_o5_o6 = 0.8  # Distance from o5 to o6
K = 2  # Number of nearest neighbors

# Calculate the average distance for o5 to its two nearest neighbors
average_distance_o5 = (dist_o5_o4 + dist_o5_o6) / K

# Calculate the density at o5
density_o5 = 1 / average_distance_o5
print(density_o5)


### N
