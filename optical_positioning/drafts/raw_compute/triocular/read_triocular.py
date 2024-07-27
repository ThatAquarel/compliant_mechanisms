import cv2
import time
import numpy as np

from multiprocessing import Process, Event


def set_parameters(cap):
    width = 1280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width * 2 / 3)


def camera(camera_id, frame_event, stop_event):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    set_parameters(cap)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_event.set()
            cv2.imshow(f"camera {camera_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()

    cap.release()


def manager(n_camera, frame_event, stop_event):
    prev = 0
    i = 0
    dts = np.empty(720, dtype=np.float32)

    while not stop_event.is_set():
        frame_event.wait()
        frame_event.clear()

        if i == 720:
            print(1 / np.mean(dts) / n_camera)
            i = 0
        t = time.time()
        dt = t - prev
        dts[i] = dt
        i += 1
        prev = t

    cv2.destroyAllWindows()


def main():
    n_camera = 3
    camera_ids = [i for i in range(n_camera)]

    stop_event = Event()
    frame_event = Event()

    camera_processes = [
        Process(target=camera, args=(cam_id, frame_event, stop_event))
        for cam_id in camera_ids
    ]
    manager_process = Process(target=manager, args=(n_camera, frame_event, stop_event))

    for cam_thread in camera_processes:
        cam_thread.start()
    manager_process.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_event.set()

    for cam_thread in camera_processes:
        cam_thread.join()
    manager_process.join()


if __name__ == "__main__":
    main()
