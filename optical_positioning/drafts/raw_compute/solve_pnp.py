import cv2
import numpy as np

# Define the chessboard size
chessboard_size = (9, 6)
square_size = 0.025  # The size of a square in your defined unit (e.g., meters)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (6,6,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Define camera intrinsics (example values, replace with your camera calibration results)
K = np.array(
    [
        [362.47678783, 0.0, 403.38813512],
        [0.0, 362.85757174, 427.99828281],
        [0.0, 0.0, 1.0],
    ]
)

D = np.array([-0.04804361, -0.00403489, -0.00232701, 0.00063726])

camera_matrix_1 = K
dist_coeffs_1 = D

camera_matrix_2 = K
dist_coeffs_2 = D

camera_matrix_3 = K
dist_coeffs_3 = D

# Detect chessboard corners for each camera image (example)


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


def find_points(image, k, d):
    img = cv2.imread(image)
    img, k_new = undistort_fisheye(img, k, d)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (8, 8), (-1, -1), criteria)
    return corners, k_new


image_points_1, camera_matrix_1 = find_points(
    "optical_positioning/drafts/raw_compute/img_0.png", camera_matrix_1, dist_coeffs_1
)  # 2D points from camera 1 image
image_points_2, camera_matrix_2 = find_points(
    "optical_positioning/drafts/raw_compute/img_1.png", camera_matrix_2, dist_coeffs_2
)  # 2D points from camera 2 image
image_points_3, camera_matrix_3 = find_points(
    "optical_positioning/drafts/raw_compute/img_2.png", camera_matrix_3, dist_coeffs_3
)  # 2D points from camera 3 image

# Solve PnP for each camera
_, rvec_1, tvec_1 = cv2.solvePnP(objp, image_points_1, camera_matrix_1, dist_coeffs_1)
_, rvec_2, tvec_2 = cv2.solvePnP(objp, image_points_2, camera_matrix_2, dist_coeffs_2)
_, rvec_3, tvec_3 = cv2.solvePnP(objp, image_points_3, camera_matrix_3, dist_coeffs_3)

# Convert rotation vectors to rotation matrices
R_1, _ = cv2.Rodrigues(rvec_1)
R_2, _ = cv2.Rodrigues(rvec_2)
R_3, _ = cv2.Rodrigues(rvec_3)

# Print the rotation matrices and translation vectors
print("Camera 1 pose:")
print("Rotation Matrix:\n", R_1)
print("Translation Vector:\n", tvec_1)

print("Camera 2 pose:")
print("Rotation Matrix:\n", R_2)
print("Translation Vector:\n", tvec_2)

print("Camera 3 pose:")
print("Rotation Matrix:\n", R_3)
print("Translation Vector:\n", tvec_3)


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


# Define the 3D points for the axis
axis = np.float32([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, -0.05]]).reshape(-1, 3)

img = cv2.imread("optical_positioning/drafts/raw_compute/img_0.png")
img, k_new = undistort_fisheye(img, K, D)
img_pts, _ = cv2.projectPoints(axis, rvec_1, tvec_1, k_new, D)
img = cv2.drawChessboardCorners(img, chessboard_size, image_points_1, True)
img = draw_axes(img, img_pts)

# Display the image
cv2.imshow("Pose Estimation", img)
cv2.waitKey(0)

img = cv2.imread("optical_positioning/drafts/raw_compute/img_1.png")
img, k_new = undistort_fisheye(img, K, D)
img_pts, _ = cv2.projectPoints(axis, rvec_2, tvec_2, k_new, D)
img = cv2.drawChessboardCorners(img, chessboard_size, image_points_2, True)
img = draw_axes(img, img_pts)

# Display the image
cv2.imshow("Pose Estimation", img)
cv2.waitKey(0)

img = cv2.imread("optical_positioning/drafts/raw_compute/img_2.png")
img, k_new = undistort_fisheye(img, K, D)
img_pts, _ = cv2.projectPoints(axis, rvec_3, tvec_3, k_new, D)
img = cv2.drawChessboardCorners(img, chessboard_size, image_points_3, True)
img = draw_axes(img, img_pts)

# Display the image
cv2.imshow("Pose Estimation", img)
cv2.waitKey(0)

cv2.destroyAllWindows()


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("u")
ax.set_ylabel("v")
ax.set_zlabel("w")

ax.set_box_aspect([1, 1, 1])
plt.axis("scaled")
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([0, 1])

ax.scatter(*[0, 0, 0], color="black")
ax.quiver(*[0, 0, 0], *axis[1], color="blue")
ax.quiver(*[0, 0, 0], *axis[2], color="green")
ax.quiver(*[0, 0, 0], *axis[3], color="red")

ax.scatter(tvec_1[0], tvec_1[2], -tvec_1[1], marker="^", color="red")
ax.scatter(tvec_2[0], tvec_2[2], -tvec_2[1], marker="^", color="green")
ax.scatter(tvec_3[0], tvec_3[2], -tvec_3[1], marker="^", color="blue")

plt.show()
