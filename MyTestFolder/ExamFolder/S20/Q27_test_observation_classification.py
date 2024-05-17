import numpy as np

# Predictions for each round
y1_test = np.array([0, 1, 0, 0])
y2_test = np.array([0, 1, 1, 1])

# Error rates for each round
error_rates = np.array([0.417, 0.243, 0.307, 0.534])

# Calculate the weights (alphas) for each round
alphas = 0.5 * np.log((1 - error_rates) / error_rates)

# Calculate the final weighted sum for each test point
final_prediction_y1 = np.sum(alphas * y1_test)
final_prediction_y2 = np.sum(alphas * y2_test)

# Determine the final classification based on the sign of the weighted sum
final_class_y1 = np.sign(final_prediction_y1)
final_class_y2 = np.sign(final_prediction_y2)

# Map the final classifications to 0 or 1 (considering AdaBoost typically uses {-1, 1} labels)
final_class_y1 = 1 if final_class_y1 >= 0 else 0
final_class_y2 = 1 if final_class_y2 >= 0 else 0

print(f"Final classification for y1_test: {final_class_y1}")
print(f"Final classification for y2_test: {final_class_y2}")
