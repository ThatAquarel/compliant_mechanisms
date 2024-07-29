import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Initialize video capture

id = 0
cap = cv2.VideoCapture(id, cv2.CAP_DSHOW)
width = 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width * 2 / 3)

if not cap.isOpened():
    print(f"cannot open camera id: {id}")
    exit()

# Create a figure and axis for matplotlib
fig, ax = plt.subplots()
im = ax.imshow(
    np.zeros((480, 640, 3), dtype=np.uint8)
)  # Initialize with an empty image


def update(frame):
    ret, img = cap.read()  # Capture frame-by-frame
    if ret:
        im.set_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Update the image
    return (im,)


# Create an animation object
ani = animation.FuncAnimation(fig, update, blit=True, interval=50)  # Update every 50ms

# Display the plot
plt.show()

# Release the capture when done
cap.release()
