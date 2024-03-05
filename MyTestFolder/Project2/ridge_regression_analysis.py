import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('MyTestFolder/Project2/student-mat-selected.csv', sep=';')

X = df[['studytime', 'absences', 'age', 'Medu', 'Fedu', 'Dalc', 'Walc', 'health']].values
y = df['G3'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#K-fold cross-validation and range of lambda values
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
lambdas = np.logspace(-4, 4, 100)

generalization_errors = []

for lambda_ in lambdas:
    model = Ridge(alpha=lambda_)
    mse_scores = cross_val_score(model, X_scaled, y, cv=k_fold, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-mse_scores)
    generalization_errors.append(rmse_scores.mean())

plt.figure(figsize=(8, 6))
plt.semilogx(lambdas, generalization_errors, '-o')
plt.title('Generalization Error as a Function of λ')
plt.xlabel('λ (Regularization Strength)')
plt.ylabel('RMSE (Generalization Error)')
plt.show()
