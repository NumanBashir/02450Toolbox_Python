import numpy as np 
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt 

# Data from Table 5
table5 = np.array([
    [1, 1, 1, 1, 0, 1],  
    [0, 0, 0, 0, 0, 0],  
    [1, 1, 0, 1, 0, 0],  
    [0, 1, 1, 0, 1, 0],  
    [1, 1, 1, 1, 1, 1],  
    [0, 0, 0, 0, 0, 0],  
    [1, 1, 0, 1, 0, 0],  
    [0, 1, 1, 0, 1, 0],  
    [1, 1, 1, 1, 0, 1],  
    [0, 1, 1, 0, 1, 0],   
    [0, 0, 0, 0, 0, 0],   
    [1, 1, 0, 1, 0, 0],   
    [0, 1, 1, 0, 1, 0],   
    [0, 1, 1, 0, 1, 0],   
])

# Extract x1 and y
x1 = table5[:, 0]
y = table5[:, 5]

# Calculate the ROC curve and AUC
fpr, tpr, _ = roc_curve(y, x1)
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for x1')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Print the AUC value
print(f"AUC: {roc_auc:.3f}")
