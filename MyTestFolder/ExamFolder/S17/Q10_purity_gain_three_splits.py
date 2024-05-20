import numpy as np

# Given data
data = {
    "3 gears": {"Low": 13, "High": 2},
    "4 gears": {"Low": 2, "High": 10},
    "5 gears": {"Low": 2, "High": 3},
}

# Total counts
total_low = sum([v["Low"] for v in data.values()])
total_high = sum([v["High"] for v in data.values()])
total_count = total_low + total_high

# Initial impurity
initial_impurity = 1 - max(total_low / total_count, total_high / total_count)

# Split A
low_A1 = data["3 gears"]["Low"]
high_A1 = data["3 gears"]["High"]
count_A1 = low_A1 + high_A1
impurity_A1 = 1 - max(low_A1 / count_A1, high_A1 / count_A1)
weight_A1 = count_A1 / total_count

low_A2 = data["4 gears"]["Low"] + data["5 gears"]["Low"]
high_A2 = data["4 gears"]["High"] + data["5 gears"]["High"]
count_A2 = low_A2 + high_A2
impurity_A2 = 1 - max(low_A2 / count_A2, high_A2 / count_A2)
weight_A2 = count_A2 / total_count

impurity_A = weight_A1 * impurity_A1 + weight_A2 * impurity_A2
purity_gain_A = initial_impurity - impurity_A

# Split B
low_B1 = data["3 gears"]["Low"] + data["4 gears"]["Low"]
high_B1 = data["3 gears"]["High"] + data["4 gears"]["High"]
count_B1 = low_B1 + high_B1
impurity_B1 = 1 - max(low_B1 / count_B1, high_B1 / count_B1)
weight_B1 = count_B1 / total_count

low_B2 = data["5 gears"]["Low"]
high_B2 = data["5 gears"]["High"]
count_B2 = low_B2 + high_B2
impurity_B2 = 1 - max(low_B2 / count_B2, high_B2 / count_B2)
weight_B2 = count_B2 / total_count

impurity_B = weight_B1 * impurity_B1 + weight_B2 * impurity_B2
purity_gain_B = initial_impurity - impurity_B

# Split C
count_C1 = data["3 gears"]["Low"] + data["3 gears"]["High"]
impurity_C1 = 1 - max(data["3 gears"]["Low"] / count_C1, data["3 gears"]["High"] / count_C1)
weight_C1 = count_C1 / total_count

count_C2 = data["4 gears"]["Low"] + data["4 gears"]["High"]
impurity_C2 = 1 - max(data["4 gears"]["Low"] / count_C2, data["4 gears"]["High"] / count_C2)
weight_C2 = count_C2 / total_count

count_C3 = data["5 gears"]["Low"] + data["5 gears"]["High"]
impurity_C3 = 1 - max(data["5 gears"]["Low"] / count_C3, data["5 gears"]["High"] / count_C3)
weight_C3 = count_C3 / total_count

impurity_C = weight_C1 * impurity_C1 + weight_C2 * impurity_C2 + weight_C3 * impurity_C3
purity_gain_C = initial_impurity - impurity_C

print(f"Purity gain for Split A: {purity_gain_A:.4f}")
print(f"Purity gain for Split B: {purity_gain_B:.4f}")
print(f"Purity gain for Split C: {purity_gain_C:.4f}")
