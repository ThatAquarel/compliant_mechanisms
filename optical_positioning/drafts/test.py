# x = 10
# y = 58
# print(x + y)

import numpy as np

a = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]
)

b = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)
T, _, _, _ = np.linalg.lstsq(a, b)
print(a)
print(b)
print(T.T)
print([3, 3, 0] @ T)
