import pandas as pd
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('MyTestFolder/Project1/student-mat-selected.csv', sep=';')

numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()

data_standardized = scaler.fit_transform(data[numeric_columns])

data_standardized_df = pd.DataFrame(data_standardized, columns=numeric_columns)

print(data_standardized_df.head())
