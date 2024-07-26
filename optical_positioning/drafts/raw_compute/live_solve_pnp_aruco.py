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


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
objp = (
    np.array(
        [
            [0, 6, 0],
            [6, 6, 0],
            [6, 0, 0],
            [0, 0, 0],
            [7, 6, 0],
            [13, 6, 0],
            [13, 0, 0],
            [7, 0, 0],
            [0, 13, 0],
            [6, 13, 0],
            [6, 7, 0],
            [0, 7, 0],
            [-7, 6, 0],
            [-1, 6, 0],
            [-1, 0, 0],
            [-7, 0, 0],
            [0, -1, 0],
            [6, -1, 0],
            [6, -7, 0],
            [0, -7, 0],
        ],
        dtype=np.float32,
    )
    / 100
)


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


def find_points(gray):
    corners, ids, _ = detector.detectMarkers(gray)
    out_corners = None

    if type(ids) != type(None):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = np.array(corners)
        corners_shape = corners.shape
        out_corners = cv2.cornerSubPix(
            gray, corners.reshape((-1, 2)), (3, 3), (-1, -1), criteria
        )
        out_corners = out_corners.reshape(corners_shape)

        return True, out_corners, ids

    return False, out_corners, ids


def draw_axes(img, img_pts):
    img_pts = img_pts.astype(np.int32)
    # Draw the axes
    img = cv2.line(
        img, tuple(img_pts[0].ravel()), tuple(img_pts[1].ravel()), (0, 0, 255), 3
    )  # X-axis in red
    img = cv2.line(
        img, tuple(img_pts[0].ravel()), tuple(img_pts[2].ravel()), (0, 255, 0), 3
    )  # Y-axis in green
    img = cv2.line(
        img, tuple(img_pts[0].ravel()), tuple(img_pts[3].ravel()), (255, 0, 0), 3
    )  # Z-axis in blue
    return img


while True:
    ax.clear()

    ret, frame = cap.read()
    frame = frame[:, 240:1040]

    frame, K_new = undistort_fisheye(frame, K, D)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners, ids = find_points(gray)
    if ret and len(ids) == 5:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        corners = corners[np.argsort(ids.flatten())].reshape((-1, 1, 2))

        _, rvec, tvec = cv2.solvePnP(objp, corners, K_new, D)
        r_1, _ = cv2.Rodrigues(rvec)

        # ax.scatter(tvec[0], tvec[2], -tvec[1], marker="^", color="red")
        # ax.scatter(*tvec, marker="^", color="red")

        T_w_c = np.eye(4)
        T_w_c[0:3, 0:3] = r_1
        T_w_c[0:3, 3:4] = tvec
        T_c_w = np.linalg.inv(T_w_c)
        tvec_c_w = T_c_w[0:3, 3:4]

        # world at (0, 0, 0), camera moving
        # ax.scatter(*tvec_c_w.flatten(), marker="^", color="red")

        # camera at (0, 0, 0), world moving
        T_cv_bl = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
        ax.scatter(*tvec.flatten() @ T_cv_bl, marker="^", color="black")
        ax.quiver(*[0, 0, 0], *([0, 0, 1, 1] @ T_w_c.T)[0:3] @ T_cv_bl, color="black")

        axis_cv = np.float32(
            [[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]
        ).reshape(-1, 3)
        img_pts, _ = cv2.projectPoints(axis_cv, rvec, tvec, K_new, D)
        frame = draw_axes(frame, img_pts)

    axis_math = np.float32(
        [[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]
    ).reshape(-1, 3)
    ax.quiver(*[0, 0, 0], *axis_math[1], color="red")
    ax.quiver(*[0, 0, 0], *axis_math[2], color="green")
    ax.quiver(*[0, 0, 0], *axis_math[3], color="blue")
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
