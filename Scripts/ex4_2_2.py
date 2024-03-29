# Exercise 4.2.2

import numpy as np

# requires data from exercise 4.2.1
from ex4_2_1 import *
from matplotlib.pyplot import figure, hist, show, subplot, xlabel, ylim

figure(figsize=(8, 7))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    subplot(int(u), int(v), i + 1)
    hist(X[:, i], color=(0.2, 0.8 - i * 0.2, 0.4))
    xlabel(attributeNames[i])
    ylim(0, N / 2)

show()

print("Ran Exercise 4.2.2")
