import pandas as pd

file_path = 'MyTestFolder/Project1/student-mat-selected.csv'
data = pd.read_csv(file_path, sep=';') 

missing_values = data.isnull().sum()

print("Missing values by column:")
print(missing_values)
