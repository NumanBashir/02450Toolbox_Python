import numpy as np

# Given matrices V and S

# For future questions make sure to check which coordinates they want to change. In this the numbers were v1.3, v2.5 and s2.2

# TODO CHANGE THESE VALUES ACCORDING TO THE QUESTION OPTIONS
v1 = -0.3413
v2 = -0.6508
V = np.array([
    [-0.5939, 0.2906, v1, 0.0621, 0.6652],
    [-0.6521, 0.0759, 0.0004, 0.3813, v2],
    [0.2028, -0.5105, -0.7036, 0.4508, 0.0010],
    [-0.3696, -0.5414, -0.1781, -0.7244, -0.1173],
    [-0.2102, -0.5967, 0.5973, 0.3503, 0.3467]
])

s1 = 26.9387
S = np.diag([30.3832, s1, 22.7730, 19.7263, 16.0724])

# The original matrix X can be reconstructed by U * S * V^T
# However, U is not provided, and normally U * S * V^T would give us X, not X_tilde.
# Since we're only given V and S, and assuming V contains right-singular vectors,
# we'll calculate the square of the Frobenius norm of S, which should give us the sum
# of the squares of the singular values (since the square of the Frobenius norm of X
# is the sum of the squares of its singular values).

frobenius_norm_squared = np.sum(S**2)
print(frobenius_norm_squared)
