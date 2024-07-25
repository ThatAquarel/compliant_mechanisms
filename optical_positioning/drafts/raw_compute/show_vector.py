import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("u")
ax.set_ylabel("v")
ax.set_zlabel("w")

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

ax.set_box_aspect([1, 1, 1])
ax.set_title("3D Vector Plot")

while True:
    # Define the origin and the direction of the vector
    origin = np.array([0, 0, 0])
    vector = (
        np.array([1, 0.5, 0.5])
        + [
            np.random.rand(),
        ]
        * 3
    )

    # Plot the vector using quiver
    ax.clear()
    ax.quiver(
        origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color="r"
    )

    plt.pause(0.05)
