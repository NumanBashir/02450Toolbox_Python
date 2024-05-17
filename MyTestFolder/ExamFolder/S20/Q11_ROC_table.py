import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# True class labels
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1])

# Predicted probabilities of the positive class
y_scores = np.array([0, 0.04, 0.06, 0.08, 0.1, 0.11, 0.125, 0.62, 0.7, 0.88, 0.9])

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Compute the AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
