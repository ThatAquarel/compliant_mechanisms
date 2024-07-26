import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def set_parameters(cap):
    width = 1280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width * 2 / 3)


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
set_parameters(cap)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


# Define the chessboard size
chessboard_size = (9, 6)
square_size = 0.025  # The size of a square in your defined unit (e.g., meters)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (6,6,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

K = np.array(
    [
        [362.47678783, 0.0, 403.38813512],
        [0.0, 362.85757174, 427.99828281],
        [0.0, 0.0, 1.0],
    ]
)

D = np.array([-0.04804361, -0.00403489, -0.00232701, 0.00063726])


def undistort_fisheye(image, K, D):
    DIM = image.shape[:2][::-1]
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, DIM, 1.0, DIM)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        K_new,
        DIM,
        cv2.CV_16SC2,
    )
    undistorted_img = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return undistorted_img, K_new


def find_points(gray, k, d):
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    out_corners = None
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        out_corners = cv2.cornerSubPix(gray, corners, (8, 8), (-1, -1), criteria)

    return ret, out_corners


def draw_axes(img, img_pts):
    img_pts = img_pts.astype(np.int32)
    # Draw the axes
    img = cv2.line(
        img, tuple(img_pts[0].ravel()), tuple(img_pts[1].ravel()), (255, 0, 0), 3
    )  # X-axis in blue
    img = cv2.line(
        img, tuple(img_pts[0].ravel()), tuple(img_pts[2].ravel()), (0, 255, 0), 3
    )  # Y-axis in green
    img = cv2.line(
        img, tuple(img_pts[0].ravel()), tuple(img_pts[3].ravel()), (0, 0, 255), 3
    )  # Z-axis in red
    return img


while True:
    ax.clear()

    ret, frame = cap.read()
    frame = frame[:, 240:1040]

    frame, K_new = undistort_fisheye(frame, K, D)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = find_points(gray, K, D)
    if ret:
        frame = cv2.drawChessboardCorners(frame, chessboard_size, corners, True)
        _, rvec, tvec = cv2.solvePnP(objp, corners, K_new, D)
        r_1, _ = cv2.Rodrigues(rvec)

        ax.scatter(tvec[0], tvec[2], -tvec[1], marker="^", color="red")
        # ax.scatter(*tvec, marker="^", color="red")

        axis_cv = np.float32(
            [[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]
        ).reshape(-1, 3)
        img_pts, _ = cv2.projectPoints(axis_cv, rvec, tvec, K_new, D)
        frame = draw_axes(frame, img_pts)

    ax.scatter(*[0, 0, 0], color="black")
    axis_math = np.float32(
        [[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]
    ).reshape(-1, 3)
    ax.quiver(*[0, 0, 0], *axis_math[1], color="blue")
    ax.quiver(*[0, 0, 0], *axis_math[2], color="green")
    ax.quiver(*[0, 0, 0], *axis_math[3], color="red")
    plt.axis("equal")
    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([-0.25, 0.25])
    ax.set_zlim([-0.25, 0.25])

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
