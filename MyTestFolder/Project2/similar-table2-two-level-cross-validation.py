import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')

# Define the features and the target variable
target_column = 'schoolsup'  
features = ['age', 'address', 'Pstatus', 'Medu', 'Fedu']
X = df[features]
y = df[target_column].apply(lambda x: 1 if x == 'yes' else 0)  # Binary encoding for 'yes'/'no'

# Specify the categorical features that need encoding
categorical_features = ['address', 'Pstatus']

# Create preprocessing steps for the categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply the transformations to the appropriate columns in the DataFrame
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# Parameters grid for logistic regression
param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}

# Set up the nested cross-validation
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results of cross-validation
test_error_rates = []
best_params_list = []

# Perform the outer cross-validation
for train_index, test_index in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Grid search within the inner cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Store the best parameters and compute the test error rate
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    
    # Predict on the test set using the best model
    y_pred = grid_search.predict(X_test)
    test_error_rate = 1 - accuracy_score(y_test, y_pred)
    test_error_rates.append(test_error_rate)

# Construct a DataFrame to display the results
results_df = pd.DataFrame({
    'Outer Fold': range(1, 11),
    'Best Hyperparameter C': [params['classifier__C'] for params in best_params_list],
    'Test Error Rate': test_error_rates
})

# Print and save the results
print(results_df)
results_df.to_csv('nested_cv_results.csv', index=False)

# Compute the average test error rate across all outer folds
average_test_error_rate = sum(test_error_rates) / len(test_error_rates)
print(f"Average Test Error Rate: {average_test_error_rate:.4f}")
