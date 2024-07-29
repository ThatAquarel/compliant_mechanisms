import time
import itertools
from multiprocessing import Process, Event, Queue

import cv2
import numpy as np
import scipy.optimize

import a_constants


def undistort_map(K, D):
    K_new, _ = cv2.getOptimalNewCameraMatrix(
        K, D, a_constants.DIM, 1.0, a_constants.DIM
    )
    map_1, map_2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        K_new,
        a_constants.DIM,
        cv2.CV_16SC2,
    )

    return K_new, map_1, map_2


def aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)


def camera(camera_id, frame_event, stop_event, axis_events, axis_queues):
    with open(f"{a_constants.MATRIX_DIR}cam_{camera_id}_K.npy", "rb") as f:
        K = np.load(f)
    with open(f"{a_constants.MATRIX_DIR}cam_{camera_id}_D.npy", "rb") as f:
        D = np.load(f)
    K, map_1, map_2 = undistort_map(K, D)

    detector = aruco_detector()

    cap = a_constants.camera_capture(camera_id)
    while not stop_event.is_set():
        ret, frame = cap.read()
        frame = a_constants.camera_crop(frame)
        frame = cv2.remap(frame, map_1, map_2, interpolation=cv2.INTER_LINEAR)
        gray = frame[:, :, 0]

        corners, ids, _ = detector.detectMarkers(gray)

        if type(ids) != type(None):
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if not axis_events[camera_id].is_set():
                axis_events[camera_id].set()
                axis_queues[camera_id].put((ids, corners, K))

        if ret:
            frame_event.set()
            cv2.imshow(f"camera {camera_id}", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()

    cap.release()


def get_ids_intersection(ids_a, ids_b, bundle_a, bundle_b):
    ids_ab, index_a, index_b = np.intersect1d(ids_a, ids_b, return_indices=True)
    return ids_ab, bundle_a[index_a], bundle_b[index_b]


def get_points(proj_mat_a, proj_mat_b, corners_a, corners_b):
    points_ab_homogeneous = cv2.triangulatePoints(
        proj_mat_a, proj_mat_b, corners_a, corners_b
    )

    points_ab_homogeneous /= points_ab_homogeneous[-1]
    return points_ab_homogeneous[:-1]


def homogenize(points):
    w = np.ones((len(points), 1))
    return np.hstack((points, w))


def axis(_, axis_events, axis_queues, axis_ready, axis_data):
    print("axis solve")

    mono_proj_mat = []
    mono_corners = []
    mono_ids = []

    for i, (event, queue) in enumerate(zip(axis_events, axis_queues)):
        event.wait()
        ids, corners, K = queue.get()

        ids = ids.flatten()
        corners = np.array(corners)

        axis_index = np.argwhere(
            np.isin(ids, a_constants.AXIS_IDS, assume_unique=True)
        ).flatten()
        axis_ids = ids[axis_index].flatten()

        axis_3d = a_constants.AXIS_COORDINATES[axis_ids]
        axis_2d = corners[axis_index]

        ret, rvec, tvec = cv2.solvePnP(
            axis_3d.reshape((-1, 3)), axis_2d.reshape((-1, 2)), K, None
        )

        if ret:
            print(f"solve pnp success, index: {i}")

        r_1, _ = cv2.Rodrigues(rvec)
        proj_mat = K @ np.hstack((r_1, tvec.reshape((3, 1))))

        mono_proj_mat.append(proj_mat)
        mono_corners.append(axis_2d)
        mono_ids.append(axis_ids)
    print()

    stereo_ids = []
    stereo_points = []
    stereo_bundle_transform = []
    mono_combinations = itertools.combinations(range(a_constants.N_CAMERAS), 2)
    for i, (a, b) in enumerate(mono_combinations):
        corners_a, corners_b = mono_corners[a], mono_corners[b]

        ids_ab, corners_a, corners_b = get_ids_intersection(
            mono_ids[a], mono_ids[b], corners_a, corners_b
        )

        points_ab = get_points(
            mono_proj_mat[a],
            mono_proj_mat[b],
            corners_a.reshape((-1, 2)).T,
            corners_b.reshape((-1, 2)).T,
        )

        if i > 0:
            ids_base = stereo_ids[0]
            points_base = stereo_points[0]

            _, points_base, points_ab = get_ids_intersection(
                ids_base,
                ids_ab,
                points_base.T.reshape((-1, 4, 3)),
                points_ab.T.reshape((-1, 4, 3)),
            )
            points_base = homogenize(points_base.reshape((-1, 3)))
            points_ab = points_ab.reshape((-1, 3))

            def transform_mse(T, x, y):
                T = T.reshape((4, 3))
                xT = x @ T
                se = (xT - y) ** 2
                mse = np.mean(se, axis=1)
                return mse

            def eval(T):
                mse = np.mean(transform_mse(T, points_base, points_ab))
                de = 100 * np.sqrt(mse)
                print(f"stereo {a}_{b}, error with stereo 0_1: {de:.4f} cm")
                return de

            T_i = np.eye(4)[0:4, 0:3]
            ei = eval(T_i)

            print("bundle adj")
            result = scipy.optimize.least_squares(
                transform_mse, T_i.flatten(), args=(points_base, points_ab)
            )

            T_f = result.x.reshape((4, 3))
            ef = eval(T_f)

            if ei >= ef:
                print("use bundle transform")
                stereo_bundle_transform.append(T_f)
            else:
                print("use identity transform")
                stereo_bundle_transform.append(T_i)
            print()

        stereo_ids.append(ids_ab)
        stereo_points.append(points_ab)

    print("axis ready")
    axis_ready.set()


def position(stop_event, axis_ready, axis_data):
    axis_ready.wait()
    print("positioning using axis")


def manager(frame_event, stop_event):
    prev = 0
    i = 0
    dts = np.empty(720, dtype=np.float32)

    while not stop_event.is_set():
        frame_event.wait()
        frame_event.clear()

        if i == 720:
            print(1 / np.mean(dts) / a_constants.N_CAMERAS)
            i = 0
        t = time.time()
        dt = t - prev
        dts[i] = dt
        i += 1
        prev = t

    cv2.destroyAllWindows()


def main():
    stop_event = Event()
    frame_event = Event()
    manager_process = Process(target=manager, args=(frame_event, stop_event))

    axis_events = [Event() for _ in a_constants.CAMERAS]
    axis_queues = [Queue() for _ in a_constants.CAMERAS]
    camera_processes = [
        Process(
            target=camera,
            args=(cam_id, frame_event, stop_event, axis_events, axis_queues),
        )
        for cam_id in a_constants.CAMERAS
    ]

    axis_ready = Event()
    axis_data = Queue()
    axis_process = Process(
        target=axis, args=(stop_event, axis_events, axis_queues, axis_ready, axis_data)
    )
    position_process = Process(
        target=position, args=(stop_event, axis_ready, axis_data)
    )

    for cam_thread in camera_processes:
        cam_thread.start()
    manager_process.start()
    axis_process.start()
    position_process.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_event.set()

    for cam_thread in camera_processes:
        cam_thread.join()
    manager_process.join()
    axis_process.join()
    position_process.join()


if __name__ == "__main__":
    main()
