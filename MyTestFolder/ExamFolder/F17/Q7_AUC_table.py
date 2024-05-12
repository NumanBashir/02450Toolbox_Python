import numpy as np # type: ignore
from sklearn.metrics import roc_curve, auc # type: ignore
import matplotlib.pyplot as plt # type: ignore

#TODO if table use this 
# Heights for each player based on your provided data
table = np.array([5.7, 6.0, 6.2, 6.3, 6.4, 6.6, 6.7, 6.9, 7.0, 7.4])
#TODO: Label is used if you have to fliter it in classes forexample red or black
labels = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])  # Correct labels as you described

#TODO use this if matrix, remember to read the task and see which one of the fields it is asking for
#TODO if similar to exam16f Q18

x1 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0])  
y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])   
# Compute ROC curve and AUC


# Data for gears and mpg classification
#TODO: if table similar to exam17S Q11

gears = np.array([3]*15 + [4]*12 + [5]*5)
mpg = np.array([0]*13 + [1]*2 + [0]*2 + [1]*10 + [0]*2 + [1]*3)

# # Simple numerical encoding for gears (although one-hot encoding might be more appropriate in other contexts)
# numerical_gears = np.array([1 if gear == 4 else (2 if gear == 5 else 0) for gear in gears])


#TODO: Update roc_curve based on the task
fpr, tpr, _ = roc_curve(labels, table)
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Player Height vs FG%')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
