import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# True class labels
y_true = np.array([1, 0, 0, 1, 1, 0, 1])

# Predicted probabilities of the positive class
y_scores = np.array([0.01, 0.05, 0.14, 0.3, 0.31, 0.36, 0.91])

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
