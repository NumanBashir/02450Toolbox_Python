import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('MyTestFolder/Project1/student-mat-selected.csv', sep=';')
columns_of_interest = list(data.columns)[1:]  # Exclude the first column

data[columns_of_interest].boxplot()
plt.title('Boxplots')
plt.ylabel('Values')
plt.show()