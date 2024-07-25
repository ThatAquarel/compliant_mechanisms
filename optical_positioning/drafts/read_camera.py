import time
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FPS, 120)

if not cap.isOpened():
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

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        continue

    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        break


cap.release()
cv.destroyAllWindows()
