# Given data
total_observations = 8760
low_demand_observations = 3285
medium_demand_observations = 2190
high_demand_observations = 3285

low_humidity_low_demand = 1327
low_humidity_medium_demand = 1718
low_humidity_high_demand = 2344

# Calculate probabilities
P_high_demand = high_demand_observations / total_observations
P_low_humidity_given_high_demand = low_humidity_high_demand / high_demand_observations
P_high_humidity_given_high_demand = 1 - P_low_humidity_given_high_demand

P_low_humidity = (low_humidity_low_demand + low_humidity_medium_demand + low_humidity_high_demand) / total_observations
P_high_humidity = 1 - P_low_humidity

# Bayes' theorem
P_high_demand_given_high_humidity = (P_high_humidity_given_high_demand * P_high_demand) / P_high_humidity

# Print the result
print(P_high_demand_given_high_humidity)
