import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')

categorical_features = df.select_dtypes(include=['object']).columns

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('G3')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

#ridge regression with cross-validation with parameters from 10^-4 to 10^4
lambdas = np.logspace(-4, 4, 100)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train_errors, val_errors, coefs = [], [], []

X = df.drop('G3', axis=1)
y = df['G3']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#pipeline with preprocessing and ridge regression
for lambda_ in lambdas:
    ridge = Ridge(alpha=lambda_)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ridge', ridge)])
    
    val_score = cross_val_score(pipeline, X_train, y_train, cv=kf, 
                                scoring='neg_mean_squared_error')
    val_errors.append(np.mean(-val_score))
    
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    
    coefs.append(pipeline.named_steps['ridge'].coef_)

train_errors = np.array(train_errors)
val_errors = np.array(val_errors)
coefs = np.array(coefs)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for coef in coefs.T:
    plt.semilogx(lambdas, coef)
plt.title('Ridge coefficients as a function of the regularization')
plt.xlabel('λ (Regularization strength)')
plt.ylabel('Coefficient value')

plt.subplot(1, 2, 2)
plt.semilogx(lambdas, train_errors, label='Training error')
plt.semilogx(lambdas, val_errors, label='Validation error')
plt.title('Training/Validation error as a function of the regularization')
plt.xlabel('λ (Regularization strength)')
plt.ylabel('Mean squared error')
plt.legend()

plt.tight_layout()
plt.show()
