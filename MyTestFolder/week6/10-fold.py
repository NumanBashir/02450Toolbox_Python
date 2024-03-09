# Parameters
N = 1000  # Total number of observations
lambda_values = 6  # Number of lambda values
D_CV_ratio = 0.8  # Ratio of D_CV
D_validation_ratio = 0.2  # Ratio of D_validation
K_fold = 10  # Number of folds in cross-validation

def calculate_total_time(N, lambda_values, D_CV_ratio, D_validation_ratio, K_fold):
    # Calculate the sizes of D_CV and D_validation
    D_CV_size = int(N * D_CV_ratio)
    D_validation_size = int(N * D_validation_ratio)

    # Calculate the time taken for training and testing for a single lambda
    train_time_single_lambda = D_CV_size**2
    test_time_single_lambda = (D_validation_size**2) / 2

    # Calculate the time taken for one fold of cross-validation
    CV_train_size = int(D_CV_size * (1 - 1/K_fold))
    CV_test_size = int(D_CV_size * (1/K_fold))
    CV_time_single_fold = CV_train_size**2 + (CV_test_size**2) / 2

    # Total time for 10 folds of cross-validation
    CV_time_total = CV_time_single_fold * K_fold

    # Total time for the whole process for a single lambda, including final train and test
    total_time_single_lambda = (train_time_single_lambda + test_time_single_lambda + CV_time_total)

    # Total time for all lambda values
    total_time_all_lambda = total_time_single_lambda * lambda_values

    return total_time_all_lambda

# Call the function with the given parameters
total_time = calculate_total_time(N, lambda_values, D_CV_ratio, D_validation_ratio, K_fold)
print(f"The total time for the entire procedure is {total_time} units.")
