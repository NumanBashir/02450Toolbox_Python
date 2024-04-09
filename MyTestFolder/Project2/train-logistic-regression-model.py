import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')

y = (df['G3'] >= 10).astype(int)
X = df.drop(columns=['G3'])

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

column_transformer = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_columns),
    ('num', StandardScaler(), numeric_columns)
])

# Setup the logistic regression model with cross-validation
logistic_cv = LogisticRegressionCV(
    Cs=10,  
    cv=5,   
    penalty='l2',  
    scoring='accuracy',  
    max_iter=10000,  
    random_state=42  
)

# Create a pipeline with preprocessing and the logistic regression model
pipeline = Pipeline([
    ('preprocess', column_transformer),
    ('logistic_cv', logistic_cv)
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model on the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Extract and print the best value of C found by cross-validation
best_C = logistic_cv.C_[0]
print(f"Best C (inverse of lambda): {best_C}")

# Extract and print the feature importance (coefficients)
feature_importance = pipeline.named_steps['logistic_cv'].coef_
print("Feature importance in logistic regression:")
print(feature_importance.flatten())
