import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

objp_id = {
    0: np.array(
        [
            [0, 6, 0],
            [6, 6, 0],
            [6, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    / 100,
    1: np.array(
        [
            [7, 6, 0],
            [13, 6, 0],
            [13, 0, 0],
            [7, 0, 0],
        ],
        dtype=np.float32,
    )
    / 100,
    2: np.array(
        [
            [0, 13, 0],
            [6, 13, 0],
            [6, 7, 0],
            [0, 7, 0],
        ],
        dtype=np.float32,
    )
    / 100,
    3: np.array(
        [
            [-7, 6, 0],
            [-1, 6, 0],
            [-1, 0, 0],
            [-7, 0, 0],
        ],
        dtype=np.float32,
    )
    / 100,
    4: np.array(
        [
            [0, -1, 0],
            [6, -1, 0],
            [6, -7, 0],
            [0, -7, 0],
        ],
        dtype=np.float32,
    )
    / 100,
    11: np.array(
        [
            [7, 6.5, 0.5],
            [7, 6.5, 6.5],
            [13, 6.5, 6.5],
            [13, 6.5, 0.5],
        ],
        dtype=np.float32,
    )
    / 100,
    12: np.array(
        [
            [7, 7, 7],
            [7, 13, 7],
            [13, 13, 7],
            [13, 7, 7],
        ],
        dtype=np.float32,
    )
    / 100,
}

K = np.array(
    [
        [362.47678783, 0.0, 403.38813512],
        [0.0, 362.85757174, 427.99828281],
        [0.0, 0.0, 1.0],
    ]
)

D = np.array([-0.04804361, -0.00403489, -0.00232701, 0.00063726])


imgs = [
    cv2.imread(f"optical_positioning/drafts/raw_compute/triocular/3d_cam_{i}.png")
    for i in range(3)
]


def undistort_fisheye(image, K, D):
    DIM = image.shape[:2][::-1]
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, DIM, 1.0, DIM)
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

triangulate_corners_all = []
triangulate_corners_ids = []
for i in range(3):
    frame, K_new = undistort_fisheye(imgs[i], K, D)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners, ids = find_points(gray)
    if ret and len(ids) >= 2:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        axis_corners = []
        axis_objp = []
        triangulate_corners = []
        triangulate_ids = []
        for i, id in enumerate(ids.flatten()):
            if not id in objp_id.keys():
                continue
            id_corners = corners[i]
            id_objp = objp_id[id]
            if id < 5:
                axis_corners.append(id_corners)
                axis_objp.append(id_objp)
            triangulate_corners.append(id_corners)
            triangulate_ids.append(id)
        axis_corners = np.array(axis_corners).reshape((-1, 1, 2))
        axis_objp = np.array(axis_objp).reshape((-1, 3))
        triangulate_corners = np.array(triangulate_corners)

        _, rvec, tvec = cv2.solvePnP(axis_objp, axis_corners, K_new, D)
        r_1, _ = cv2.Rodrigues(rvec)

        triangulate_corners_all.append(triangulate_corners)
        triangulate_corners_ids.append(triangulate_ids)

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


def get_corner_intersection(id_0, id_1, corners_0, corners_1):
    _, ind_0, ind_1 = np.intersect1d(id_0, id_1, return_indices=True)
    return corners_0[ind_0].reshape((-1, 2)).T, corners_1[ind_1].reshape((-1, 2)).T


obj_pos_0 = compute_obj_pos(
    projection_matrix[0],
    projection_matrix[1],
    *get_corner_intersection(
        triangulate_corners_ids[0],
        triangulate_corners_ids[1],
        triangulate_corners_all[0],
        triangulate_corners_all[1],
    ),
)

obj_pos_1 = compute_obj_pos(
    projection_matrix[1],
    projection_matrix[2],
    *get_corner_intersection(
        triangulate_corners_ids[1],
        triangulate_corners_ids[2],
        triangulate_corners_all[1],
        triangulate_corners_all[2],
    ),
)

obj_pos_2 = compute_obj_pos(
    projection_matrix[0],
    projection_matrix[2],
    *get_corner_intersection(
        triangulate_corners_ids[0],
        triangulate_corners_ids[2],
        triangulate_corners_all[0],
        triangulate_corners_all[2],
    ),
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
