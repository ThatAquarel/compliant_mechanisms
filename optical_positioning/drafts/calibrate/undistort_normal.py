import cv2
import numpy as np

# Load the camera calibration data
calibration_data = np.load("calibration_data.npz")
mtx = calibration_data["mtx"]
dist = calibration_data["dist"]

# Open the camera stream (0 is the default camera, change if necessary)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the width and height of the frames
ret, frame = cap.read()
if not ret:
    print("Error: Could not read a frame.")
    cap.release()
    exit()

h, w = frame.shape[:2]

# Get the optimal new camera matrix
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Start the video stream
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Crop the image (optional, based on the ROI)
    # x, y, w, h = roi
    # undistorted_frame = undistorted_frame[y : y + h, x : x + w]

    # Display the resulting frame
    cv2.imshow("Undistorted Frame", undistorted_frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
