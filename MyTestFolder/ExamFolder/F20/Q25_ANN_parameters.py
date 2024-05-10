# Define the network structure
M = 4  # Number of input features
n_h = 6  # Number of hidden units
C = 3  # Number of classes TODO CHANGE THIS TO MATCH THE QUESTION OR TRY EACH NUMBER UNTIL CORRECT ANSWER

# Calculate the parameters for the hidden layer
hidden_layer_params = (M + 1) * n_h

# Calculate the parameters for the output layer
output_layer_params = (n_h + 1) * C

# Calculate the total number of parameters
total_params = hidden_layer_params + output_layer_params

print(f"Total number of parameters in the network: {total_params}")
