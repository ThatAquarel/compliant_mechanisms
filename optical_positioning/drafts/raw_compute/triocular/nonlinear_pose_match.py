import numpy as np
import scipy.optimize

points_0 = np.array(
    [
        [0.5, 0, 0, 1],
        [0.5, 0.5, 0, 1],
        [0, 0.5, 0, 1],
        [0, 0, -0.5, 1],
    ],
    dtype=np.float32,
)
points_0

points_1 = np.array(
    [
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, -1.25],
    ],
    dtype=np.float32,
)


def transform_cost(T, x, y):
    T = T.reshape((4, 3))

    x_ = x @ T
    se = (x_ - y) ** 2
    mse = np.mean(se, axis=1)

    return mse


T_i = np.eye(4)[0:4, 0:3]
initial_params = T_i.flatten()
result = scipy.optimize.least_squares(
    transform_cost, initial_params, args=(points_0, points_1)
)

T_r = result.x.reshape((4, 3))

print(T_r)
print(points_0)
print(points_0 @ T_r)
print(points_1)
