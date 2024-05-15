import numpy as np

# Data fra tabellen - udskift med dit eget
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

# Kategorier
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Funktioner og målværdi
features = [0, 1, 2]  # f1, f2, f3
target_value = [0, 1, 1]  # f1 = 0, f2 = 1, f3 = 1

#udskift også target value til givet

# Beregn prior sandsynligheder
p_y_1 = np.sum(y == 1) / len(y)
p_y_0 = np.sum(y == 0) / len(y)

# Beregn betingede sandsynligheder for hver funktion givet y = 1 og y = 0
p_f_given_y_1 = np.prod([np.sum(data[y == 1, f] == target_value[i]) / np.sum(y == 1) for i, f in enumerate(features)])
p_f_given_y_0 = np.prod([np.sum(data[y == 0, f] == target_value[i]) / np.sum(y == 0) for i, f in enumerate(features)])

# Anvend Bayes' sætning til at beregne den betingede sandsynlighed for y = 1
p_y_1_given_f = (p_f_given_y_1 * p_y_1) / (p_f_given_y_1 * p_y_1 + p_f_given_y_0 * p_y_0)

print(f"p_NB(y = 1 | f1 = 0, f2 = 1, f3 = 1) ≈ {p_y_1_given_f:.3f}")
