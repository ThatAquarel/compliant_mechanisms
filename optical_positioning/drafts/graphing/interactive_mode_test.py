import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


ax.set_xlim([-0.25, 0.25])
ax.set_ylim([-0.25, 0.25])
ax.set_zlim([-0.25, 0.25])

scatter = ax.scatter(None, None, None, marker="^", color="red")

for i in range(100):
    scatter._offsets3d = np.random.rand(30).reshape((3, 10)) * 0.5 - 0.25

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.5)

plt.ioff()
