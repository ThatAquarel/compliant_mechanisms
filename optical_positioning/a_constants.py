import cv2
import time
import numpy as np

CAMERAS = np.arange(3)
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


CALIBRATION_DIR = "optical_positioning/b_calibrate/"
MATRIX_DIR = "optical_positioning/c_calibrate_matrix/"
