import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def set_parameters(cap):
    width = 1280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width * 2 / 3)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
set_parameters(cap)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("u")
ax.set_ylabel("v")
ax.set_zlabel("w")

ax.set_box_aspect([1, 1, 1])
ax.set_title("3D Vector Plot")


while True:
    ax.clear()

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    retc, corners = cv2.findChessboardCorners(gray, (6, 9), None)

    if retc:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, (6, 9), corners2, ret)

        h, w = gray.shape
        f = 640
        normalized_corners = (corners2 - [w / 2, h / 2]) / f
        corners_x, corners_y = normalized_corners[:, :, 0], normalized_corners[:, :, 1]
        theta = np.sqrt(corners_x**2 + corners_y**2)
        w = np.sqrt(corners_x**2 + corners_y**2) / np.tan(theta)

        zero = np.zeros(w.shape)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1])
        ax.quiver(
            zero,
            zero,
            zero,
            corners_x,
            corners_y,
            w,
            color="b",
            linewidths=0.1,
        )
        ax.scatter(corners_x, corners_y, w, marker="o")

    if not ret:
        print("Can't receive frame (stream endq?). Exiting ...")
        continue

    plt.pause(0.00001)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
plt.show()
