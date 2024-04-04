import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')

# Define your features and target variable
target_column = 'schoolsup'  # Update with the actual name of your target column
X = df.drop(target_column, axis=1)
y = df[target_column].apply(lambda x: 1 if x == 'yes' else 0)  # Binary encoding for 'yes'/'no'

# Define categorical features for one-hot encoding
categorical_features = ['age', 'address', 'Pstatus', 'Medu', 'Fedu']  # Update with your actual categorical features

# Preprocessing pipeline for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Models setup
knn = KNeighborsClassifier()
logistic = LogisticRegression(solver='liblinear')
baseline = DummyClassifier(strategy='most_frequent')

# Parameters grid
knn_params = {'knn__n_neighbors': range(1, 31)}
logistic_params = {'logistic__C': np.logspace(-4, 4, 10)}

# Pipelines for models
knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('knn', knn)])
logistic_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('logistic', logistic)])
baseline_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('baseline', baseline)])

# Cross-validation setup
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Storage for results
results = []

# Outer CV
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    # Train-test split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # KNN grid search within an inner CV
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_gs = GridSearchCV(knn_pipeline, knn_params, cv=inner_cv, scoring='accuracy')
    knn_gs.fit(X_train, y_train)

    # Logistic regression grid search within an inner CV
    logistic_gs = GridSearchCV(logistic_pipeline, logistic_params, cv=inner_cv, scoring='accuracy')
    logistic_gs.fit(X_train, y_train)

    # Baseline model fit
    baseline_pipeline.fit(X_train, y_train)

    # Get the error rate for each model
    knn_error_rate = 1 - accuracy_score(y_test, knn_gs.predict(X_test))
    logistic_error_rate = 1 - accuracy_score(y_test, logistic_gs.predict(X_test))
    baseline_error_rate = 1 - accuracy_score(y_test, baseline_pipeline.predict(X_test))

    # Append results
    results.append({
        'Outer fold': fold_idx,
        'KNN Best K': knn_gs.best_params_['knn__n_neighbors'],
        'KNN Test Error Rate': knn_error_rate,
        'Logistic Regression Best C': logistic_gs.best_params_['logistic__C'],
        'Logistic Regression Test Error Rate': logistic_error_rate,
        'Baseline Test Error Rate': baseline_error_rate
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Calculate the mean error rate across all folds
mean_knn_error = results_df['KNN Test Error Rate'].mean()
mean_logistic_error = results_df['Logistic Regression Test Error Rate'].mean()
mean_baseline_error = results_df['Baseline Test Error Rate'].mean()

# Add a row with the mean error rate to the DataFrame
mean_error_rates = pd.DataFrame([['Mean', '', mean_knn_error, '', mean_logistic_error, mean_baseline_error]],
                                columns=results_df.columns)
results_df = pd.concat([results_df, mean_error_rates], ignore_index=True)

# Print and save the results
print(results_df)
results_df.to_csv('cross_validation_results.csv', index=False)  # Update the path
