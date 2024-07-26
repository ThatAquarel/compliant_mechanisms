import cv2
import numpy as np


def set_parameters(cap):
    width = 1280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width * 2 / 3)


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
set_parameters(cap)


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


if not cap.isOpened():
    print("Cannot open camera")
    exit()


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


K = np.array(
    [
        [362.47678783, 0.0, 403.38813512],
        [0.0, 362.85757174, 427.99828281],
        [0.0, 0.0, 1.0],
    ]
)

D = np.array([-0.04804361, -0.00403489, -0.00232701, 0.00063726])


while True:
    ret, frame = cap.read()
    frame = frame[:, 240:1040]
    frame, K_new = undistort_fisheye(frame, K, D)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if type(ids) != type(None):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = np.array(corners)
        corners_shape = corners.shape
        corners = cv2.cornerSubPix(
            gray, corners.reshape((-1, 2)), (3, 3), (-1, -1), criteria
        )

        corners = corners.reshape(corners_shape)
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    if not ret:
        print("Can't receive frame (stream endq?). Exiting ...")
        continue
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
