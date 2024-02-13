import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('MyTestFolder/Project1/student-mat-selected.csv', sep=';')

# Convert 'G3' into a binary variable (e.g., pass/fail)
df['pass'] = df['G3'].apply(lambda x: 1 if x > 10 else 0)

# Select predictor variables and the target variable
X = df.drop(['G3', 'pass'], axis=1)  # Drop 'G3' and the newly created 'pass' column
y = df['pass']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining preprocessing for numeric and nominal features
numeric_features = ['age', 'absences']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

nominal_features = ['school', 'sex']  # Example nominal features
nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('nom', nominal_transformer, nominal_features)
    ])

# Creating a preprocessing and training pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

# Training the model
clf.fit(X_train, y_train)

# Predicting the test set
y_pred = clf.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
