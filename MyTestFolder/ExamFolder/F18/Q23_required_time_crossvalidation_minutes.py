def calculate_total_training_time(S, K2, K1, single_model_time):
    # Calculate the total number of models
    total_models = K1 * (K2 * S + 1)
    
    # Calculate total training time
    total_training_time = total_models * single_model_time
    
    return total_training_time

# Given values
S = 4
K2 = 4
K1 = 3
Train_single_model_time = 20

# Calculate total training time
total_time = calculate_total_training_time(S, K2, K1, Train_single_model_time)
print(f"The total training time for all models is {total_time} minutes.")
