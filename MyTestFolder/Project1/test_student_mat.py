import pandas as pd
# Followed by the rest of your script...

# Load the dataset
file_path = 'MyTestFolder/StudentPerformance/student-mat.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()


