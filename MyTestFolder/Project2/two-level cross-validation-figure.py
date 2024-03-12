import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset, assuming it's correctly named and located
data = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', delimiter=';')

# Identify non-numeric columns for one-hot encoding
non_numeric_columns = data.select_dtypes(include=['object']).columns.tolist()

# Apply one-hot encoding if non-numeric columns exist
if non_numeric_columns:
    ct = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), non_numeric_columns)],
                           remainder='passthrough')
    X = ct.fit_transform(data.iloc[:, :-1])
else:
    X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Define the number of outer and inner folds
K1 = 10
K2 = 5

# Initialize KFold for the outer loop
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)

# Initialize results storage
results = {
    'Outer Fold': [],
    'Optimal h': [],
    'Optimal lambda': [],
    'ANN Test Error': [],
    'Baseline Test Error': []
}

# Outer cross-validation loop
for i, (train_val_indices, test_indices) in enumerate(outer_cv.split(X), 1):
    X_train_val, X_test = X[train_val_indices], X[test_indices]
    y_train_val, y_test = y[train_val_indices], y[test_indices]

    # Inner cross-validation for hyperparameter tuning
    inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)

    # Define hyperparameter space for h and lambda
    h_values = [1, 2, 4, 6, 8, 10]  # Range for the number of hidden units
    lambda_values = [0.001, 0.01, 0.1, 1]  # Regularization strengths

    best_h = None
    best_lambda = None
    lowest_error = np.inf

    for h in h_values:
        for lambda_val in lambda_values:
            inner_errors = []

            for train_indices, val_indices in inner_cv.split(X_train_val):
                X_train, X_val = X_train_val[train_indices], X_train_val[val_indices]
                y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]

                # Create a pipeline with scaling and MLPRegressor
                scaler = StandardScaler()
                mlp = MLPRegressor(hidden_layer_sizes=(h,), alpha=lambda_val,
                                   max_iter=3000, learning_rate_init=0.005, momentum=0.9,
                                   solver='adam', early_stopping=True, n_iter_no_change=20, random_state=42)
                pipeline = Pipeline(steps=[('scaler', scaler), ('mlp', mlp)])

                # Train the model
                pipeline.fit(X_train, y_train)

                # Predict on validation set and store error
                y_pred_val = pipeline.predict(X_val)
                error = mean_squared_error(y_val, y_pred_val)
                inner_errors.append(error)

            average_error = np.mean(inner_errors)
            if average_error < lowest_error:
                lowest_error = average_error
                best_h = h
                best_lambda = lambda_val

    # Retrain model with best hyperparameters on full training/validation data
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(best_h,), alpha=best_lambda,
                             max_iter=3000, learning_rate_init=0.005, momentum=0.9,
                             solver='adam', early_stopping=True, n_iter_no_change=20, random_state=42))
    ])
    best_pipeline.fit(X_train_val, y_train_val)

    # Evaluate on test set
    y_pred_test = best_pipeline.predict(X_test)
    ann_test_error = mean_squared_error(y_test, y_pred_test)

    # Baseline model (mean prediction)
    baseline_pred = np.full(y_test.shape, y_train_val.mean())
    baseline_test_error = mean_squared_error(y_test, baseline_pred)

    # Store results
    results['Outer Fold'].append(i)
    results['Optimal h'].append(best_h)
    results['Optimal lambda'].append(best_lambda)
    results['ANN Test Error'].append(ann_test_error)
    results['Baseline Test Error'].append(baseline_test_error)

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('cross_validation_results.csv', index=False)

print("Cross-validation results:")
print(results_df)
