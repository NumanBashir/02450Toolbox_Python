import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# TODO CHANGE THESE NUMBERS FROM DATA
x5 = np.array([3, 4, 5])
low_mpg = np.array([13, 2, 2])
high_mpg = np.array([2, 10, 3])

# Create labels and scores
labels = np.concatenate([np.zeros(low_mpg[i]) for i in range(len(x5))] + 
                        [np.ones(high_mpg[i]) for i in range(len(x5))])
scores = np.concatenate([np.full(low_mpg[i], x5[i]) for i in range(len(x5))] + 
                        [np.full(high_mpg[i], x5[i]) for i in range(len(x5))])

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(labels, scores)  # Using scores to treat higher gears as more likely to be high mpg
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for x5 (gear)')
plt.legend(loc="lower right")
plt.show()

print(f"AUC: {roc_auc:.3f}")
