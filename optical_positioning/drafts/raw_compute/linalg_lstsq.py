import matplotlib.pyplot as plt
import numpy as np

a = [[0, 0, 0], [0, 1, 0], [1, 0, 0]]
b = [[0, 0, 0], [0, 1, 0], [0, 0, 2]]
T, _, _, _ = np.linalg.lstsq(a, b)

n = np.linspace(0, 1, 10)
x, y, z = np.meshgrid(n, n, 0)

points = np.concatenate([x, y, z], axis=-1).reshape((-1, 3))
points_t = np.dot(points, T)

# Create a new figure
fig = plt.figure()

# Add 3D subplot
ax = fig.add_subplot(111, projection="3d")

# Plot the 3D data points
ax.scatter(x, y, z, c="r", marker="o")
ax.scatter(points_t[:, 0], points_t[:, 1], points_t[:, 2], c="b", marker="o")

# Set labels
ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")

# Show plot
plt.show()
