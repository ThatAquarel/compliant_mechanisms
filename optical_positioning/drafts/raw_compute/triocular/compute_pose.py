import cv2
import numpy as np

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


imgs = [
    cv2.imread(f"optical_positioning/drafts/raw_compute/triocular/cam_{i}.png")
    for i in range(3)
]


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


frames = []
projection_matrix = []
triangulate_corners = []
for i in range(3):
    frame, K_new = undistort_fisheye(imgs[i], K, D)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners, ids = find_points(gray)
    if ret and len(ids) == 5:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        corners = corners[np.argsort(ids.flatten())].reshape((-1, 1, 2))

        _, rvec, tvec = cv2.solvePnP(objp, corners, K_new, D)
        r_1, _ = cv2.Rodrigues(rvec)

        triangulate_corners.append(corners.reshape((-1, 2)))

        proj_mat = np.hstack((r_1, tvec.reshape((3, 1))))
        proj_mat = K @ proj_mat
        projection_matrix.append(proj_mat)

    frames.append(frame)


def compute_obj_pos(proj_mat_0, proj_mat_1, points_0, points_1):
    object_points_homogeneous = cv2.triangulatePoints(
        proj_mat_0,
        proj_mat_1,
        points_0,
        points_1,
    )

    object_points_homogeneous /= object_points_homogeneous[3]
    return object_points_homogeneous[:3]


obj_pos_0 = compute_obj_pos(
    projection_matrix[0],
    projection_matrix[1],
    triangulate_corners[0].T,
    triangulate_corners[1].T,
)

obj_pos_1 = compute_obj_pos(
    projection_matrix[1],
    projection_matrix[2],
    triangulate_corners[1].T,
    triangulate_corners[2].T,
)

obj_pos_2 = compute_obj_pos(
    projection_matrix[0],
    projection_matrix[2],
    triangulate_corners[0].T,
    triangulate_corners[2].T,
)


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.scatter(*obj_pos_0, color="red")
ax.scatter(*obj_pos_1, color="green")
ax.scatter(*obj_pos_2, color="blue")

axis_math = np.float32([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]).reshape(
    -1, 3
)
ax.quiver(*[0, 0, 0], *axis_math[1], color="red")
ax.quiver(*[0, 0, 0], *axis_math[2], color="green")
ax.quiver(*[0, 0, 0], *axis_math[3], color="blue")
plt.axis("equal")
ax.set_xlim([-0.25, 0.25])
ax.set_ylim([-0.25, 0.25])
ax.set_zlim([-0.25, 0.25])

plt.show()
