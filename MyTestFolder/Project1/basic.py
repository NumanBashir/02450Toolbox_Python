import pandas as pd

df = pd.read_csv('MyTestFolder/Project1/student-mat-selected.csv', sep=';')

print(df.describe())