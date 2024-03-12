import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
df = pd.read_csv('MyTestFolder/Project1/student-mat-selected.csv', sep=';')

# Select features and target
X = df.drop('G3', axis=1)
y = df['G3']

# Preprocessing: Standardize numeric features and One-Hot Encode categorical features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Baseline model: Linear regression with constant features
class BaselineModel:
    def fit(self, X, y):
        self.mean = np.mean(y)
        
    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean)

# ANN model with adjusted parameters for better convergence
ann_model = MLPRegressor(max_iter=5000, learning_rate_init=0.001, solver='adam', random_state=42, early_stopping=True)

# Models for comparison
models = {
    'Baseline': BaselineModel(),
    'Ridge': Ridge(),
    'ANN': ann_model
}

# Set up two-level cross-validation
K1, K2 = 10, 10
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)

# Parameters to try for Ridge
ridge_alphas = np.logspace(-4, 4, 10)

# Two-level cross-validation
for name, model in models.items():
    scores = []
    
    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if name == 'Ridge':
            # Inner CV for Ridge
            best_score, best_alpha = -np.inf, None
            for alpha in ridge_alphas:
                model.alpha = alpha
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                score = np.mean(cross_val_score(pipeline, X_train, y_train, cv=inner_cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score, best_alpha = score, alpha
            model.alpha = best_alpha
        
        # No inner CV for ANN and Baseline, direct training
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        scores.append(mean_squared_error(y_test, y_pred))
    
    print(f'{name}: Mean MSE = {np.mean(scores)}')
