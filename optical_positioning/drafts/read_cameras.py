import time
import cv2 as cv
import numpy as np


def set_parameters(cap):
    # width = 1280  # fps 120
    # width = 960  # fps 144
    width = 640  # fps 144

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, width * 2 / 3)


cam_0 = cv.VideoCapture(0, cv.CAP_DSHOW)
cam_1 = cv.VideoCapture(1, cv.CAP_DSHOW)
set_parameters(cam_0)
set_parameters(cam_1)

if not cam_0.isOpened() or not cam_1.isOpened():
    print("Cannot open camera")
    exit()

prev = 0
i = 0
dts = np.empty(100, dtype=np.float32)

while True:
    if i == 100:
        print(1 / np.mean(dts))
        i = 0

    t = time.time()
    dt = time.time() - prev
    dts[i] = dt
    i += 1
    prev = t

    ret_0, frame_0 = cam_0.read()
    ret_1, frame_1 = cam_1.read()
    if not ret_0 or not ret_1:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow("frame", cv.hconcat([frame_0, frame_1]))
    if cv.waitKey(1) == ord("q"):
        break


cam_0.release()
cv.destroyAllWindows()
