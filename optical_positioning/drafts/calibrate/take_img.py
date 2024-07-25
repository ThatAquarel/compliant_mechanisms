import time
import cv2 as cv
import numpy as np


def set_parameters(cap):
    width = 1280
    # width = 960
    # width = 640

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, width * 2 / 3)


cap = cv.VideoCapture(0, cv.CAP_DSHOW)
set_parameters(cap)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev = 0
i = 0
dts = np.empty(100, dtype=np.float32)

img_n = 0

while True:
    if i == 100:
        print(1 / np.mean(dts))
        i = 0

    t = time.time()
    dt = time.time() - prev
    dts[i] = dt
    i += 1
    prev = t

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        continue

    cv.imshow("frame", frame)
    key = cv.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord("s"):
        cv.imwrite(f"./optical_positioning/drafts/calibrate/img/img_{img_n}.png", frame)
        img_n += 1


cap.release()
cv.destroyAllWindows()
