import glob

import cv2
import numpy as np

import optical_positioning.a_constants as a_constants

CHECKERBOARD = (6, 9)


def calibrate_matrix():
    obj_pos = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3))
    obj_pos[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(
        -1, 2
    )

    for id in a_constants.CAMERAS:
        image_files = glob.glob(f"{a_constants.CALIBRATION_DIR}cam_{id}*")
        img_points = []
        gray = None

        for file in image_files:
            img = cv2.imread(file)
            gray = img[:, :, 0]

            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (8, 8),
                    (-1, -1),
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )
                chess = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
                cv2.imshow(file, chess)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                img_points.append(corners)

        obj_points = [obj_pos.reshape((-1, 1, 3)) for _ in range(len(img_points))]
        K = np.eye(3)
        D = np.zeros(4)

        rms, K, D, _, _ = cv2.fisheye.calibrate(
            obj_points,
            img_points,
            gray.shape[::-1],
            K,
            D,
            flags=cv2.fisheye.CALIB_FIX_SKEW
            + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_CHECK_COND,
        )

        print(f"camera {id}")
        print(f"rms error: {rms}")
        print(f"K: {K}")
        print(f"D: {D}")
        print()

        with open(f"{a_constants.MATRIX_DIR}cam_{id}_K.npy", "wb") as f:
            np.save(f, K)
        with open(f"{a_constants.MATRIX_DIR}cam_{id}_D.npy", "wb") as f:
            np.save(f, D)


if __name__ == "__main__":
    calibrate_matrix()
