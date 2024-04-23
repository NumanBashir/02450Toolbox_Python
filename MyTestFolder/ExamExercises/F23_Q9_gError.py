import numpy as np

# Validation errors for each model on the inner folds and the test error for each outer fold
data = {
    'outer_fold_1': {
        'inner_folds': {
            'Model 1': [0.12, 0.21, 0.22, 0.23, 0.15],
            'Model 2': [0.30, 0.11, 0.15, 0.30, 0.28],
            'Model 3': [0.21, 0.14, 0.26, 0.17, 0.26]
        },
        'test_error': {
            'Model 1': 0.24,
            'Model 2': 0.17,
            'Model 3': 0.22
        }
    },
    'outer_fold_2': {
        'inner_folds': {
            'Model 1': [0.28, 0.18, 0.19, 0.27, 0.12],
            'Model 2': [0.16, 0.20, 0.27, 0.30, 0.25],
            'Model 3': [0.13, 0.16, 0.21, 0.17, 0.13]
        },
        'test_error': {
            'Model 1': 0.19,
            'Model 2': 0.16,
            'Model 3': 0.25
        }
    }
}

# Function to calculate the generalization error
def calculate_generalization_error(data):
    generalization_error = {}
    
    # Calculate the generalization error for each model
    for model in ['Model 1', 'Model 2', 'Model 3']:
        error_sum = 0
        fold_count = 0
        
        for fold in data.values():
            # Calculate the average validation error for the inner folds
            avg_validation_error = np.mean(fold['inner_folds'][model])
            # Get the test error for the outer fold
            test_error = fold['test_error'][model]
            # Add the difference between test error and validation error to the sum
            error_sum += (test_error - avg_validation_error)
            fold_count += 1
        
        # Calculate the average of the differences
        generalization_error[model] = error_sum / fold_count
    
    return generalization_error

# Calculate the generalization error for each model
generalization_errors = calculate_generalization_error(data)
print(generalization_errors)
