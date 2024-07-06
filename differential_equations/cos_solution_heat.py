import numpy as np

import matplotlib.pyplot as plt

samples = 1000

n = 5
a = 0.5

T = 4
X = np.pi

# evaluation space
x = np.linspace(0, X, samples, endpoint=False)
x = np.tile(x, (samples, 1))

t = np.linspace(0, T, samples, endpoint=False)
t = np.tile(t, (samples, 1)).T

# evaluate
x = np.cos(n * (np.pi / X) * x)
t = np.e ** (-a * (n * ((np.pi / X)) ** 2) * t)
f = x * t
plt.imshow(f)
plt.show()
