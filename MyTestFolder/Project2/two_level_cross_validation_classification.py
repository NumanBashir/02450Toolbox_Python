import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')  # Update the path as needed

def error_rate(y_true, y_pred):
    """Calculate the error rate as the fraction of misclassified samples."""
    return np.sum(y_true != y_pred) / len(y_true)

# Transforming variables to categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

categorical_columns = [col for col in categorical_columns if col != 'G3']
numeric_columns = [col for col in numeric_columns if col != 'G3']

# Define the column transformer
column_transformer = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_columns),
    ('num', StandardScaler(), numeric_columns)
])

# Define feature matrix X and target vector y
X = df.drop('G3', axis=1)  
y = df['G3'].astype(int)  # Convert to int for classification

# Initialize cross-validation
K1, K2 = 10, 10  # Folds
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)

results = []

# Perform two-level cross-validation
for fold_num, (train_index, test_index) in enumerate(outer_cv.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Pipelines for each model
    knn_pipeline = Pipeline([
        ('preprocess', column_transformer),
        ('knn', KNeighborsClassifier())
    ])

    logistic_pipeline = Pipeline([
        ('preprocess', column_transformer),
        ('logistic', LogisticRegression(random_state=42))
    ])

    baseline_pipeline = Pipeline([
        ('preprocess', column_transformer),
        ('baseline', DummyClassifier(strategy='most_frequent', random_state=42))
    ])

    # Inner cross-validation for KNN parameter tuning
    best_k, best_error_rate = None, np.inf

    for k in range(1, 21):
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
            best_error_rate, best_k = average_error, k

    # Retrain models with best parameters or settings
    knn_pipeline.set_params(knn__n_neighbors=best_k).fit(X_train, y_train)
    logistic_pipeline.fit(X_train, y_train)
    baseline_pipeline.fit(X_train, y_train)

    # Evaluate on the outer test set
    y_pred_knn = knn_pipeline.predict(X_test)
    y_pred_logistic = logistic_pipeline.predict(X_test)
    y_pred_baseline = baseline_pipeline.predict(X_test)

    results.append({
        'Fold': fold_num,
        'k': best_k,
        'Error_Logistic': error_rate(y_test, y_pred_logistic),
        'Error_KNN': error_rate(y_test, y_pred_knn),
        'Error_Baseline': error_rate(y_test, y_pred_baseline)
    })

results_df = pd.DataFrame(results)
print(results_df)
