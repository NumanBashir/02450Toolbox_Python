import numpy as np
from scipy.stats import norm

# Given observations
X = np.array([-0.82, 0.0, 2.5])

# Values of lambda
lambdas = [0.15, 0.21, 0.49, 1.15]

# Function to calculate LOO log likelihood
def loo_log_likelihood(X, lam):
    N = len(X)
    log_likelihoods = np.zeros(N)
    
    for i in range(N):
        x_i = X[i]
        other_x = np.delete(X, i)
        density = np.mean(norm.pdf(x_i, loc=other_x, scale=lam))
        log_likelihoods[i] = np.log(density)
    
    return log_likelihoods

# Calculate log likelihoods for each lambda
results = {}
for lam in lambdas:
    log_likelihoods = loo_log_likelihood(X, lam)
    results[lam] = log_likelihoods
    print(f"Lambda = {lam}: Log Likelihoods = {log_likelihoods}")

# Determine which lambda matches the given log likelihoods
given_log_likelihoods = np.array([-2.3, -2.3, -13.91])

for lam, log_likelihoods in results.items():
    if np.allclose(log_likelihoods, given_log_likelihoods, atol=0.1):
        print(f"Lambda {lam} matches the given log likelihoods.")
