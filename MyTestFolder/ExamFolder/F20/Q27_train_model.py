# Constants
outer_folds_used = 2
loocv_trainings_per_fold = 221
models_per_lambda = 4
lambda_values = 4  # Number of different lambda values tested
final_optimal_lambda_training = 1

# Calculate total training uses
total_uses = outer_folds_used * (loocv_trainings_per_fold * models_per_lambda) + final_optimal_lambda_training

print(f"Total times each observation is used for training: {total_uses}")
