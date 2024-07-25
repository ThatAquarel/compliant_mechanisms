import glob

import cv2
import numpy as np

# Prepare object points
CHECKERBOARD = (6, 9)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []
imgpoints = []

# Load images
image_files = glob.glob("./optical_positioning/drafts/calibrate/img/img*")

gray = None
for file in image_files:
    img = cv2.imread(file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        chess = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        # cv2.imwrite(f"{file}_chess.png", chess)
        objpoints.append(objp)
        imgpoints.append(corners)

# Perform fisheye calibration
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]

ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    np.expand_dims(np.asarray(objpoints), -2),
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    # cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    # + cv2.fisheye.CALIB_CHECK_COND
    cv2.fisheye.CALIB_FIX_SKEW,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
)

# The intrinsic matrix K is now obtained
print("Intrinsic Matrix K:\n", K)
print("Distortion Coefficients D:\n", D)
