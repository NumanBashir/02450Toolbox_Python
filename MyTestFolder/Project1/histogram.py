import matplotlib.pyplot as plt
import pandas as pd

# Read the data from CSV
data = pd.read_csv('MyTestFolder/Project1/student-mat-selected.csv', sep=';')
attributes = list(data.columns[1:])

# Iterate over each attribute and plot the histogram
for attribute in attributes:
    plt.figure(figsize=(8, 6))
    plt.hist(data[attribute], bins='auto', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of {attribute}")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.show()