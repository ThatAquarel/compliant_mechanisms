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
K = np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 400.0], [0.0, 0.0, 1.0]])

camera_matrix_1 = np.array(K, dtype=np.float32)
dist_coeffs_1 = np.zeros((5, 1), dtype=np.float32)

camera_matrix_2 = np.array(K, dtype=np.float32)
dist_coeffs_2 = np.zeros((5, 1), dtype=np.float32)

camera_matrix_3 = np.array(K, dtype=np.float32)
dist_coeffs_3 = np.zeros((5, 1), dtype=np.float32)

# Detect chessboard corners for each camera image (example)


def find_points(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners
    # return cv2.fisheye.undistortPoints(corners, K, np.zeros((4, 1)))


image_points_1 = find_points(
    "optical_positioning/drafts/raw_compute/img_0.png"
)  # 2D points from camera 1 image
image_points_2 = find_points(
    "optical_positioning/drafts/raw_compute/img_1.png"
)  # 2D points from camera 2 image
image_points_3 = find_points(
    "optical_positioning/drafts/raw_compute/img_2.png"
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
        img, tuple(img_pts[0].ravel()), tuple(img_pts[1].ravel()), (255, 0, 0), 5
    )  # X-axis in blue
    img = cv2.line(
        img, tuple(img_pts[0].ravel()), tuple(img_pts[2].ravel()), (0, 255, 0), 5
    )  # Y-axis in green
    img = cv2.line(
        img, tuple(img_pts[0].ravel()), tuple(img_pts[3].ravel()), (0, 0, 255), 5
    )  # Z-axis in red
    return img


# Define the 3D points for the axis
axis = np.float32([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, -0.05]]).reshape(-1, 3)

img = cv2.imread("optical_positioning/drafts/raw_compute/img_0.png")
# Project the 3D points to the image plane
img_pts, _ = cv2.projectPoints(axis, rvec_1, tvec_1, K, np.zeros(0))

# Draw the chessboard corners and the axes
img = cv2.drawChessboardCorners(img, chessboard_size, image_points_1, True)
img = draw_axes(img, img_pts)

# Display the image
cv2.imshow("Pose Estimation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
