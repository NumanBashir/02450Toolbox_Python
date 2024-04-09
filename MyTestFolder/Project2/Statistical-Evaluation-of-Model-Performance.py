import numpy as np
import pandas as pd
from scipy.stats import t, sem
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')

# Identify categorical and numerical columns (adjust as necessary)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
target_column = 'G3'  


if target_column in numeric_columns:
    numeric_columns.remove(target_column)

# Define the column transformer for preprocessing
column_transformer = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_columns),
    ('num', StandardScaler(), numeric_columns)
])

# Define feature matrix X and target vector y
X = df.drop(target_column, axis=1)
y = df[target_column].astype(int)

# Define error rate function
def error_rate(y_true, y_pred):
    return np.sum(y_true != y_pred) / len(y_true)

# Custom scorer for cross-validation
error_rate_scorer = make_scorer(error_rate, greater_is_better=False)

# Initialize the KFold cross-validation settings
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize models
knn_pipeline = Pipeline(steps=[('preprocess', column_transformer),
                               ('knn', KNeighborsClassifier())])
logistic_pipeline = Pipeline(steps=[('preprocess', column_transformer),
                                    ('logistic', LogisticRegression(max_iter=10000))]) # Increased max_iter for convergence
baseline_pipeline = Pipeline(steps=[('preprocess', column_transformer),
                                    ('baseline', DummyClassifier(strategy='most_frequent'))])

# Perform cross-validation and compute error rates
error_knn = cross_val_score(knn_pipeline, X, y, cv=kf, scoring=error_rate_scorer)
error_logistic = cross_val_score(logistic_pipeline, X, y, cv=kf, scoring=error_rate_scorer)
error_baseline = cross_val_score(baseline_pipeline, X, y, cv=kf, scoring=error_rate_scorer)

# Compute differences in error rates for each fold between pairs of models
diff_knn_logistic = error_knn - error_logistic
diff_knn_baseline = error_knn - error_baseline
diff_logistic_baseline = error_logistic - error_baseline

# Function to compute correlated t-test statistics
def compute_correlated_t_test(diff_errors):
    mean_diff = np.mean(diff_errors)
    sem_diff = sem(diff_errors)
    t_statistic = mean_diff / sem_diff
    df = len(diff_errors) - 1
    p_value = t.sf(np.abs(t_statistic), df) * 2
    confidence_interval = t.interval(0.95, df, mean_diff, sem_diff)
    return mean_diff, t_statistic, p_value, confidence_interval

# Compute statistics for each model comparison
stats_knn_logistic = compute_correlated_t_test(diff_knn_logistic)
stats_knn_baseline = compute_correlated_t_test(diff_knn_baseline)
stats_logistic_baseline = compute_correlated_t_test(diff_logistic_baseline)

# Output the results
print('KNN vs Logistic:', stats_knn_logistic)
print('KNN vs Baseline:', stats_knn_baseline)
print('Logistic vs Baseline:', stats_logistic_baseline)
