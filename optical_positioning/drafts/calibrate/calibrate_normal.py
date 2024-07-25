import cv2
import numpy as np
import glob

# Define the dimensions of the chessboard
chessboard_size = (9, 6)
# Termination criteria for corner sub-pix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the chessboard dimensions
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []
imgpoints = []

# Load all images from the directory
images = glob.glob("./optical_positioning/drafts/calibrate/img/img*")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Save the camera calibration result for later use
np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
