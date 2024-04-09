import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')

def error_rate(y_true, y_pred):
    """Calculate the error rate as the fraction of misclassified samples."""
    return np.sum(y_true != y_pred) / len(y_true)

# Define the column transformer
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove 'G3' from both column lists if present
categorical_columns = [col for col in categorical_columns if col != 'G3']
numeric_columns = [col for col in numeric_columns if col != 'G3']

column_transformer = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_columns),
    ('num', StandardScaler(), numeric_columns)
])

# Define feature matrix X and target vector y
X = df.drop('G3', axis=1)
y = df['G3'].astype(int) 


# Initialize the KFold cross-validation settings
K1, K2 = 10, 10  # Number of folds for outer and inner cross-validation
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)

# Initialize the results list
results = []

# Perform two-level cross-validation
for fold_num, (train_index, test_index) in enumerate(outer_cv.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Define the pipeline for the KNN model with preprocessing
    knn_pipeline = Pipeline(steps=[
        ('preprocess', column_transformer),
        ('knn', KNeighborsClassifier())
    ])
    
    # Inner cross-validation for KNN parameter tuning
    best_k = None
    best_error_rate = np.inf

    for k in range(1, 21):  # Trying k values from 1 to 20
        knn_pipeline.set_params(knn__n_neighbors=k)
        fold_errors = []

        for inner_train_index, inner_test_index in inner_cv.split(X_train):
            X_inner_train, X_inner_test = X_train.iloc[inner_train_index], X_train.iloc[inner_test_index]
            y_inner_train, y_inner_test = y_train.iloc[inner_train_index], y_train.iloc[inner_test_index]

            knn_pipeline.fit(X_inner_train, y_inner_train)
            y_inner_pred = knn_pipeline.predict(X_inner_test)
            fold_errors.append(error_rate(y_inner_test, y_inner_pred))

        average_error = np.mean(fold_errors)
        if average_error < best_error_rate:
            best_error_rate = average_error
            best_k = k

    # Retrain KNN with best parameters on the whole training set
    knn_pipeline.set_params(knn__n_neighbors=best_k)
    knn_pipeline.fit(X_train, y_train)
    y_pred_knn = knn_pipeline.predict(X_test)

    # Logistic Regression Model with preprocessing
    logistic_pipeline = Pipeline(steps=[
        ('preprocess', column_transformer),
        ('logistic', LogisticRegression(random_state=42))
    ])
    
    # Train the logistic regression model
    logistic_pipeline.fit(X_train, y_train)
    y_pred_logistic = logistic_pipeline.predict(X_test)

    # Baseline Model with preprocessing
    baseline_pipeline = Pipeline(steps=[
        ('preprocess', column_transformer),
        ('baseline', DummyClassifier(strategy='most_frequent', random_state=42))
    ])
    
    # Train the baseline model
    baseline_pipeline.fit(X_train, y_train)
    y_pred_baseline = baseline_pipeline.predict(X_test)

    # Append the results for this fold
    results.append({
        'Fold': fold_num,
        'Error_Logistic': error_rate(y_test, y_pred_logistic),
        'Error_KNN': error_rate(y_test, y_pred_knn),
        'k': best_k,
        'Error_Baseline': error_rate(y_test, y_pred_baseline)
    })

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)
print(results_df)
