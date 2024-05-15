import numpy as np

# Data fra tabellen (binary matrix)
data = np.array([
    [0, 1, 1, 0, 1],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# Funktion til at beregne SMC
def smc(A, B):
    n = len(A)
    n_11 = np.sum((A == 1) & (B == 1))
    n_00 = np.sum((A == 0) & (B == 0))
    return (n_11 + n_00) / n

# Funktion til at beregne Jaccard-similaritet
def jaccard(A, B):
    n_11 = np.sum((A == 1) & (B == 1))
    n_01 = np.sum((A == 0) & (B == 1))
    n_10 = np.sum((A == 1) & (B == 0))
    return n_11 / (n_11 + n_01 + n_10)

# Funktion til at beregne cosinus-similaritet
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

# Beregn SMC, Jaccard og cosinus-similaritet for de specifikke par
o1 = data[0]
o2 = data[1]
o3 = data[2]

smc_o1_o2 = smc(o1, o2)
jaccard_o1_o3 = jaccard(o1, o3)
cos_o1_o2 = cosine_similarity(o1, o2)
cos_o1_o3 = cosine_similarity(o1, o3)

print(f"SMC(o1, o2) = {smc_o1_o2:.6f}")
print(f"J(o1, o3) = {jaccard_o1_o3:.6f}")
print(f"COS(o1, o2) = {cos_o1_o2:.6f}")
print(f"COS(o1, o3) = {cos_o1_o3:.6f}")

# Sammenlign værdierne for at finde den korrekte udsagn
if cos_o1_o2 > smc_o1_o2:
    print("A. COS(o1, o2) > SMC(o1, o2) is true.")
if cos_o1_o2 > cos_o1_o3:
    print("B. COS(o1, o2) > COS(o1, o3) is true.")
if jaccard_o1_o3 > smc_o1_o2:
    print("C. J(o1, o3) > SMC(o1, o2) is true.")
if jaccard_o1_o3 > cos_o1_o3:
    print("D. J(o1, o3) > COS(o1, o3) is true.")
