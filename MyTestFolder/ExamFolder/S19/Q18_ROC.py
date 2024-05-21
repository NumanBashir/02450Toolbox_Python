from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Suppose y_true and y_scores are known
y_true = [1, 1, 0, 1, 1, 1, 0]  # Actual classes
y_scores = [0.14, 0.15, 0.27, 0.61, 0.71, 0.75, 0.81]  # Predicted probabilities

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
