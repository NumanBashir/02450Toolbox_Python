import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Assuming df is your DataFrame and 'G3' is your target variable
# Load the dataset
df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')

# Preprocess or discretize 'G3' if necessary
# For example, if 'G3' is continuous, you might categorize it as passing or failing based on a threshold
# Here, I'll assume 'G3' is already suitable for classification or has been preprocessed accordingly

y = (df['G3'] >= 10).astype(int)
X = df.drop('G3', axis=1)

# Define categorical and numeric columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

# Setup the column transformer for preprocessing
column_transformer = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_columns),
    ('num', StandardScaler(), numeric_columns)
])

# Setup the logistic regression model with cross-validation to select lambda (C)
logistic_cv = LogisticRegressionCV(
    Cs=10,  # Number of C values to try
    cv=5,   # 5-fold cross-validation
    penalty='l2',  # Using L2 regularization
    scoring='accuracy',  # Optimization criterion
    max_iter=10000,  # Ensuring convergence
    random_state=42  # For reproducibility
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocess', column_transformer),
    ('logistic_cv', logistic_cv)
])

# Fit the model
pipeline.fit(X, y)

# The best value of C (inverse of lambda)
best_C = logistic_cv.C_[0]
print(f"Best C (inverse of lambda): {best_C}")

# Feature importance (coefficients)
feature_importance = pipeline.named_steps['logistic_cv'].coef_
print("Feature importance in logistic regression:", feature_importance)

