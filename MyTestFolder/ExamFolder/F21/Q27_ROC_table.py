import numpy as np
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt 

#TODO if table use this 
# Heights for each player based on your provided data
table = np.array([0, 0.04, 0.1, 0.18, 0.39, 0.52, 0.53, 0.72, 0.82])

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Define the predicted values and actual labels for each prediction option
predictions = {
    'A': {'y_hat': [0, 0.04, 0.1, 0.18, 0.39, 0.52, 0.53, 0.72, 0.82], 'y': [1, 0, 0, 1, 1, 0, 0, 0, 1]},
    'B': {'y_hat': [0, 0.04, 0.1, 0.18, 0.39, 0.52, 0.53, 0.72, 0.82], 'y': [1, 0, 0, 0, 1, 0, 1, 0, 1]},
    'C': {'y_hat': [0, 0.04, 0.1, 0.18, 0.39, 0.52, 0.53, 0.72, 0.82], 'y': [1, 1, 1, 0, 0, 0, 1, 0, 1]},
    'D': {'y_hat': [0, 0.04, 0.1, 0.18, 0.39, 0.52, 0.53, 0.72, 0.82], 'y': [0, 1, 0, 0, 1, 0, 0, 1, 1]}
}

# Plotting setup
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axes = axes.flatten()

for i, (key, value) in enumerate(predictions.items()):
    y_hat = np.array(value['y_hat'])
    y = np.array(value['y'])
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y, y_hat)
    
    # Plot ROC curve
    ax = axes[i]
    ax.plot(fpr, tpr, label=f'ROC Curve {key}')
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for Prediction {key}')
    ax.legend(loc="lower right")

plt.tight_layout()
plt.show()

