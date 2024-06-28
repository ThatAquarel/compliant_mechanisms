import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
R = 1  # Radius of the cylinder

# Create a flat grid of points (for simplicity, let's create a 2D grid)
x_flat = np.linspace(0, 2 * np.pi * R, 100)
y_flat = np.linspace(-1, 1, 50)
x_flat, y_flat = np.meshgrid(x_flat, y_flat)

# Project points onto the cylinder
theta = x_flat / R
X_cylinder = R * np.cos(theta)
Y_cylinder = R * np.sin(theta)
Z_cylinder = y_flat

# Plot the points on the cylindrical surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X_cylinder, Y_cylinder, Z_cylinder, cmap="viridis!")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Flat Surface Wrapped onto a Cylinder")

plt.show()
