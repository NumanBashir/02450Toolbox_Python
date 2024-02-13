import pandas as pd

# Path to the file you've uploaded
file_path = 'MyTestFolder/Project1/processed.cleveland.data'

# Load the dataset
# You might need to specify the separator if it's not a comma
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())
