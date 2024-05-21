import numpy as np

# Given data
n_y1 = 263
n_y2 = 359
n_y3 = 358

# Splits for x4 <= 0.43 and x4 <= 0.55
R_0_43 = np.array([[143, 120], [137, 222], [54, 304]])
R_0_55 = np.array([[178, 85], [180, 179], [75, 283]])

# Total number of observations
N_r = np.sum(R_0_43)

# Function to calculate impurity
def impurity(R):
    N_k = np.sum(R, axis=0)
    p_i = R / N_k
    I_k = 1 - np.max(p_i, axis=0)
    return I_k, N_k

# Calculate impurity for root node
I_0 = 1 - np.max([n_y1 / N_r, n_y2 / N_r, n_y3 / N_r])

# Calculate impurity for each split
I_v1_43, N_v1_43 = impurity(R_0_43)
I_v2_55, N_v2_55 = impurity(R_0_55)

# Calculate impurity gain for each split
delta_0_43 = I_0 - np.sum(N_v1_43 / N_r * I_v1_43)
delta_0_55 = I_0 - np.sum(N_v2_55 / N_r * I_v2_55)

print(f"Impurity gain for split x4 <= 0.43: Δ ≈ {delta_0_43:.4f}")
print(f"Impurity gain for split x4 <= 0.55: Δ ≈ {delta_0_55:.4f}")
