import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('MyTestFolder/Project1/student-mat-selected.csv', sep=';')

X = df[['G1', 'G2', 'age', 'absences']]  # Predictors
y = df['G3']  # Target variable

#standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#predict testing set
y_pred = model.predict(X_test)

plt.scatter(X_test[:, 1], y_test, color='black', label='Actual G3')
plt.scatter(X_test[:, 1], y_pred, color='blue', alpha=0.5, label='Predicted G3')
plt.title('Actual vs Predicted G3')
plt.xlabel('G2 (Standardized)')
plt.ylabel('G3')
plt.legend()
plt.show()

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
