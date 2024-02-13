import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generating a synthetic dataset
np.random.seed(42)  # For reproducibility
n_normal_transactions = 500
n_fraud_transactions = 50

# Normal transactions - centered around medium amounts during regular hours
normal_transaction_amounts = np.random.normal(100, 20, n_normal_transactions)
normal_transaction_hours = np.random.normal(14, 5, n_normal_transactions)

# Fraudulent transactions - higher amounts, odd hours
fraud_transaction_amounts = np.random.normal(250, 50, n_fraud_transactions)
fraud_transaction_hours = np.concatenate([np.random.normal(3, 1, n_fraud_transactions // 2),
                                          np.random.normal(23, 1, n_fraud_transactions // 2)])

# Combining the data
transaction_amounts = np.concatenate([normal_transaction_amounts, fraud_transaction_amounts])
transaction_hours = np.concatenate([normal_transaction_hours, fraud_transaction_hours])
X = np.column_stack([transaction_amounts, transaction_hours])

# Applying Isolation Forest for anomaly detection
model = IsolationForest(contamination=float(n_fraud_transactions) / (n_normal_transactions + n_fraud_transactions))
model.fit(X)

# Predictions
scores_pred = model.decision_function(X)
y_pred = model.predict(X)

# Visualizing the results
plt.scatter(transaction_amounts, transaction_hours, c=y_pred, edgecolor='k', cmap='coolwarm')
plt.xlabel('Transaction Amount')
plt.ylabel('Transaction Hour')
plt.title('Credit Card Fraud Detection')
plt.colorbar(label='Anomaly Score')
plt.show()
