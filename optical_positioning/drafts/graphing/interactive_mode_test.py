import matplotlib.pyplot as plt
import numpy as np
import time

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


ax.set_xlim([-0.25, 0.25])
ax.set_ylim([-0.25, 0.25])
ax.set_zlim([-0.25, 0.25])


center_find = np.array(
    [
        [-0.02741716, -0.06619087, +0.008, 1.000],
        [-0.02741716, -0.06619087, +0.063, 1.000],
        [+0.02741716, -0.06619087, +0.063, 1.000],
        [+0.02741716, -0.06619087, +0.008, 1.000],
        [0.000, 0.000, 0.000, 1.000],
    ],
    dtype=np.float32,
)

scatter = ax.scatter(*center_find[:, 0:3].T, marker="^", color="red")

plt.show()
