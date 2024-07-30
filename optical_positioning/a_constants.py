import cv2
import numpy as np

N_CAMERAS = 3
CAMERAS = np.arange(N_CAMERAS)
FRAMERATE = 120


def camera_capture(id, width=1280):
    cap = cv2.VideoCapture(id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width * 2 / 3)

    if not cap.isOpened():
        print(f"cannot open camera id: {id}")
        exit()

    return cap


def camera_crop(frame):
    return frame[:, 240:1040]


DIM = (800, 800)


CALIBRATION_DIR = "optical_positioning/b_calibrate/"
MATRIX_DIR = "optical_positioning/c_calibrate_matrix/"

# shape: (aruco_id, 4, 3)
AXIS_IDS = np.arange(5)
AXIS_COORDINATES = np.array(
    [
        [
            [0.00, 0.06, 0.00],
            [0.06, 0.06, 0.00],
            [0.06, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ],
        [
            [0.07, 0.06, 0.00],
            [0.13, 0.06, 0.00],
            [0.13, 0.00, 0.00],
            [0.07, 0.00, 0.00],
        ],
        [
            [0.00, 0.13, 0.00],
            [0.06, 0.13, 0.00],
            [0.06, 0.07, 0.00],
            [0.00, 0.07, 0.00],
        ],
        [
            [-0.07, 0.06, 0.00],
            [-0.01, 0.06, 0.00],
            [-0.01, 0.00, 0.00],
            [-0.07, 0.00, 0.00],
        ],
        [
            [0.00, -0.01, 0.00],
            [0.06, -0.01, 0.00],
            [0.06, -0.07, 0.00],
            [0.00, -0.07, 0.00],
        ],
    ],
    dtype=np.float32,
)

ARUCO_ID_LIMIT = 32
