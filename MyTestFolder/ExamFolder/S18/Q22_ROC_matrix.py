import numpy as np 
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt 

# Defining the features based on your description
features = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  
    [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],  
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],  
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],  
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],  
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],  
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]   
])

# TODO CHANGE COLUMN FROM FOR THE ASKED IN QUESTION
x5L = features[:, 8]

# True labels (1 = safe, 0 = unsafe) from each color observation
y_true = np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 1])

# Calculate the ROC curve and AUC
fpr, tpr, _ = roc_curve(y_true, x5L)
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Airline Safety')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
