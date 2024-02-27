import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'MyTestFolder/Project1/student-mat-selected.csv'
data = pd.read_csv(file_path, sep=';')

#correlation matrix
numeric_data = data.select_dtypes(include=np.number)
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 8))
plt.title('Correlation Matrix')

plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')

#add correlation values in the boxes
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='white')

plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()