import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('MyTestFolder/Project1/student-mat-selected.csv', sep=';')

X = df.drop('G3', axis=1)
y = df['G3']

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

#linear regression
class BaselineModel:
    def fit(self, X, y):
        self.mean = np.mean(y)
        
    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean)

# ANN model
ann_model = MLPRegressor(max_iter=5000, learning_rate_init=0.001, solver='adam', random_state=42, early_stopping=True)

# models for comparison
models = {
    'Baseline': BaselineModel(),
    'Ridge': Ridge(),
    'ANN': ann_model
}

K1, K2 = 10, 10
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)

#try for Ridge
ridge_alphas = np.logspace(-4, 4, 10)
hidden_units_range = [10, 20, 50]  # Adjust based on test runs

# two-level cross-validation
for name, model in models.items():
    scores = []
    
    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if name == 'Ridge':
            best_score, best_alpha = -np.inf, None
            for alpha in ridge_alphas:
                model.alpha = alpha
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                score = np.mean(cross_val_score(pipeline, X_train, y_train, cv=inner_cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score, best_alpha = score, alpha
            model.alpha = best_alpha
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        elif name == 'ANN':
            best_score, best_h = -np.inf, None
            for h in hidden_units_range:
                ann_model = MLPRegressor(hidden_layer_sizes=(h,),
                                         max_iter=5000,
                                         learning_rate_init=0.001,
                                         solver='adam',
                                         random_state=42,
                                         early_stopping=True)
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', ann_model)])
                score = np.mean(cross_val_score(pipeline, X_train, y_train, cv=inner_cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score, best_h = score, h
            ann_model.set_params(hidden_layer_sizes=(best_h,))
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', ann_model)])
        
        else:  
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        scores.append(mean_squared_error(y_test, y_pred))

    print(f'{name}: Mean MSE = {np.mean(scores)}')
