import time
import itertools
from multiprocessing import Process, Event, Queue
from queue import Empty

import cv2
import numpy as np
import scipy.optimize

import optical_positioning.a_constants as a_constants
import optical_positioning.e_render as e_render


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


def camera(id, frame_event, stop_event, axis_channels, position_channels, axis_ready):
    with open(f"{a_constants.MATRIX_DIR}cam_{id}_K.npy", "rb") as f:
        K = np.load(f)
    with open(f"{a_constants.MATRIX_DIR}cam_{id}_D.npy", "rb") as f:
        D = np.load(f)
    K, map_1, map_2 = undistort_map(K, D)

    detector = aruco_detector()

    cap = a_constants.camera_capture(id)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = a_constants.camera_crop(frame)
        frame = cv2.remap(frame, map_1, map_2, interpolation=cv2.INTER_LINEAR)
        gray = frame[:, :, 0]

        corners, ids, _ = detector.detectMarkers(gray)

        if type(ids) != type(None):
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            axis_event, axis_queue = axis_channels[id]
            if not axis_event.is_set():
                axis_event.set()
                axis_queue.put((ids, corners, K), block=False)

            if axis_ready.is_set():
                position_event, position_queue = position_channels[id]
                position_event.set()
                position_queue.put((ids, corners), block=False)

        frame_event.set()
        cv2.imshow(f"camera {id}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()

    cap.release()
    cv2.destroyWindow(f"camera {id}")


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


def axis(_, axis_channels, axis_ready, axis_data):
    print("axis solve")

    mono_proj_mat = []
    mono_corners = []
    mono_ids = []

    for i, (event, queue) in enumerate(axis_channels):
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
        print(f"n markers: {len(ids)}")

        r_1, _ = cv2.Rodrigues(rvec)
        proj_mat = K @ np.hstack((r_1, tvec.reshape((3, 1))))

        mono_proj_mat.append(proj_mat)
        mono_corners.append(axis_2d)
        mono_ids.append(axis_ids)
    print()

    stereo_ids = []
    stereo_points = []
    stereo_bundle_transform = []
    mono_combinations = list(itertools.combinations(range(a_constants.N_CAMERAS), 2))
    for i, (a, b) in enumerate(mono_combinations):
        corners_a, corners_b = mono_corners[a], mono_corners[b]

        ids_ab, corners_a, corners_b = get_ids_intersection(
            mono_ids[a], mono_ids[b], corners_a, corners_b
        )
        print(f"stereo {a}_{b} n intersect markers: {len(ids_ab)}")

        points_ab = get_points(
            mono_proj_mat[a],
            mono_proj_mat[b],
            corners_a.reshape((-1, 2)).T,
            corners_b.reshape((-1, 2)).T,
        )

        if i > 0:
            ids_base = stereo_ids[0]
            points_base = stereo_points[0]

            ids_base_ab, points_base, points_ab = get_ids_intersection(
                ids_base,
                ids_ab,
                points_base.T.reshape((-1, 4, 3)),
                points_ab.T.reshape((-1, 4, 3)),
            )
            print(
                f"stereo {a}_{b}, with stereo 0_1, n intersect markers: {len(ids_base_ab)}"
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

    axis_ready.set()
    axis_data.put(
        (mono_proj_mat, mono_combinations, stereo_bundle_transform), block=False
    )
    print("axis ready")
    print()


def position(stop_event, position_channels, axis_ready, axis_data, points_data):
    axis_ready.wait()
    mono_proj_mat, mono_combinations, stereo_bundle_transform = axis_data.get()
    markers_position = np.zeros((a_constants.ARUCO_ID_LIMIT, 4, 3), dtype=np.float32)
    markers_count = np.zeros(a_constants.ARUCO_ID_LIMIT, dtype=np.float32)
    markers_map = np.zeros(a_constants.ARUCO_ID_LIMIT, dtype=np.bool)

    print("axis data acquired, positioning using axis")

    while not stop_event.is_set():
        mono_data = []
        for event, queue in position_channels:
            event.wait()
            event.clear()
            mono_data.append(queue.get())
        if None in mono_data:
            break

        markers_map[:] = False
        markers_count[:] = 0
        markers_position[:] = 0
        for i, (a, b) in enumerate(mono_combinations):
            (ids_a, corners_a), (ids_b, corners_b) = mono_data[a], mono_data[b]

            ids_ab, corners_a, corners_b = get_ids_intersection(
                ids_a.flatten(),
                ids_b.flatten(),
                np.array(corners_a),
                np.array(corners_b),
            )

            if len(ids_ab) == 0:
                continue

            points_ab = get_points(
                mono_proj_mat[a],
                mono_proj_mat[b],
                corners_a.reshape((-1, 2)).T,
                corners_b.reshape((-1, 2)).T,
            )

            points_ab = points_ab.T
            if i > 0:
                points_ab = homogenize(points_ab) @ stereo_bundle_transform[i - 1]
            markers_map[ids_ab] = True
            markers_count[ids_ab] += 1
            markers_position[ids_ab] += points_ab.reshape((-1, 4, 3))
        markers_position[markers_map] /= markers_count[
            markers_map, np.newaxis, np.newaxis
        ]

        points_data.put((markers_position, markers_map), block=False)

        for _, queue in position_channels:
            while True:
                try:
                    queue.get_nowait()
                except Empty:
                    break

    stop_event.set()


def render(stop_event, points_data):
    window = e_render.init()
    scatter = None
    while not stop_event.is_set() and not e_render.window_should_close(window):
        while True:
            try:
                scatter = points_data.get_nowait()
            except Empty:
                break

        def draw():
            if scatter != None:
                markers_position, markers_map = scatter
                points = markers_position[markers_map].reshape((-1, 3))

                e_render.draw_points(points)

        e_render.update(window, draw)

    e_render.terminate()


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


def main():
    stop_event = Event()
    frame_event = Event()
    manager_process = Process(target=manager, args=(frame_event, stop_event))

    axis_channels = [(Event(), Queue()) for _ in a_constants.CAMERAS]
    position_channels = [(Event(), Queue()) for _ in a_constants.CAMERAS]

    axis_ready, axis_data = Event(), Queue()
    points_data = Queue()

    camera_processes = [
        Process(
            target=camera,
            args=(
                cam_id,
                frame_event,
                stop_event,
                axis_channels,
                position_channels,
                axis_ready,
            ),
            daemon=True,
        )
        for cam_id in a_constants.CAMERAS
    ]

    axis_process = Process(
        target=axis,
        args=(stop_event, axis_channels, axis_ready, axis_data),
        daemon=True,
    )
    position_process = Process(
        target=position,
        args=(stop_event, position_channels, axis_ready, axis_data, points_data),
        daemon=True,
    )

    render_process = Process(target=render, args=(stop_event, points_data), daemon=True)

    for cam_thread in camera_processes:
        cam_thread.start()
    manager_process.start()
    axis_process.start()
    position_process.start()
    render_process.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_event.set()
    frame_event.set()
    axis_ready.set()
    for event, queue in position_channels:
        event.set()
        queue.put(None, block=False)


if __name__ == "__main__":
    main()
