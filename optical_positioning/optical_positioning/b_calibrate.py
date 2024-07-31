import time

import cv2
import numpy as np

import optical_positioning.a_constants as a_constants


def calibrate():
    for id in a_constants.CAMERAS:
        cap = a_constants.camera_capture(id)

        img_n = 0

        prev = 0
        i = 0
        dts = np.empty(a_constants.FRAMERATE, dtype=np.float32)

        saving = True
        while saving:
            if i == a_constants.FRAMERATE:
                print(1 / np.mean(dts))
                i = 0

            t = time.time()
            dt = time.time() - prev
            dts[i] = dt
            i += 1
            prev = t

            ret, frame = cap.read()
            frame = a_constants.camera_crop(frame)

            if not ret:
                print(f"cannot read camera id: {id}")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            cv2.imshow(f"camera id: {id}", frame)
            key = cv2.waitKey(1)

            if key != ord("q") and key != ord("c"):
                if key == ord("s"):
                    cv2.imwrite(
                        f"{a_constants.CALIBRATION_DIR}cam_{id}_{img_n}.png", frame
                    )
                    img_n += 1
                continue

            cap.release()
            cv2.destroyAllWindows()

            if key == ord("q"):
                exit()
            elif key == ord("c"):
                saving = False


if __name__ == "__main__":
    calibrate()
