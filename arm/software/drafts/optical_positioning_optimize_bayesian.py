import struct
from queue import Empty
from multiprocessing import Process, Pipe, Queue, Event

import time
import serial
import imgui
import numpy as np
import scipy.optimize
from OpenGL.GL import *

from optical_positioning import d_compute_pose as optical_pose
from optical_positioning import e_render as optical_render


class commands:
    NUL = 0x00

    SYN = 0xAB
    ACK = 0x4B
    NAK = 0x5A
    ERR = 0x3C

    RESET = 0x01
    HOME_CARRIAGE = 0x02
    HOME_SERVO = 0x03
    MOVE_CARRIAGE = 0x04
    MOVE_SERVO = 0x05


class packet_t:
    cmd = 0
    buffer = 0

    def __init__(self, buffer=b"", cmd=commands.NUL):
        self.buffer = buffer
        self.cmd = cmd


def send_packet(ser_handle, packet: packet_t):
    packet_bytes = struct.pack(
        f"<xBBB{len(packet.buffer)}sBx",
        0x7E,  # U8 start of packet flag
        packet.cmd & 0xFF,  # U8 command
        len(packet.buffer) & 0xFF,  # U8 length of payload
        packet.buffer,  # U8 payload[len]
        0x7D,  # U8 end of packet flag
    )

    print(packet_bytes)
    ser_handle.write(packet_bytes)


def recv_packet(ser_handle):
    ser_handle.read_until(b"\x7E")

    command, length = struct.unpack("<cB", ser_handle.read(2))
    payload = b""
    if length:
        payload = struct.unpack(f"<{length}s", ser_handle.read(length))

    if ser_handle.read(1) != b"\x7D":
        print("serial frame end error")

    return packet_t(payload, ord(command))


center_find = np.array(
    [
        [-0.025, -0.06619087, +0.0105, 1.000],
        [-0.025, -0.06619087, +0.0605, 1.000],
        [+0.025, -0.06619087, +0.0605, 1.000],
        [+0.025, -0.06619087, +0.0105, 1.000],
        [0.000, 0.000, 0.000, 1.000],
    ],
    dtype=np.float32,
)


_r_circle = 0.06639085
_r_tip = 0.0285
_sample_n = 32
_sample_rad = np.linspace(0, 2 * np.pi, _sample_n)
_sample_xy = np.transpose(
    [np.cos(_sample_rad), np.sin(_sample_rad), np.ones(_sample_n)]
)
center_find_circle = _sample_xy * [_r_circle, _r_circle, +0.008]
center_find_tip = _sample_xy * [_r_tip, _r_tip, +0.00]

center_color = (0.5, 0.5, 0.5, 0.5)


def find_pose(reference_points, current_points):
    assert reference_points.shape == current_points.shape
    assert reference_points.shape[0] == 4

    centroid_ref = np.mean(reference_points, axis=0)
    centroid_cur = np.mean(current_points, axis=0)

    ref_centered = reference_points - centroid_ref
    cur_centered = current_points - centroid_cur

    H = ref_centered.T @ cur_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_cur - R @ centroid_ref

    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = R
    affine_matrix[:3, 3] = t

    return affine_matrix


def transform_points(points, transformation_matrix):
    n = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((n, 1))))

    transformed_points = transformation_matrix @ homogeneous_points.T

    transformed_points_cartesian = transformed_points[:3, :].T

    return transformed_points_cartesian


def draw_aruco_surface(markers_position):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    for marker in markers_position:
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(1.0, 1.0, 1.0, 0.5)
        for point in marker:
            glVertex3f(*point @ optical_render.T)

        glEnd()


def rendering(stop_event, points_data, processing_queue):
    window = optical_render.init()
    imgui_impl = optical_render.init_imgui(window)

    scatter = None
    while not stop_event.is_set() and not optical_render.window_should_close(window):
        while True:
            try:
                scatter = points_data.get_nowait()
            except Empty:
                break

        def draw():
            if scatter != None:
                markers_position, markers_map = scatter
                points = markers_position[markers_map].reshape((-1, 3))

                arm_centering_position, arm_centering_map = (
                    markers_position[8:32],
                    markers_map[8:32],
                )

                if np.sum(arm_centering_map) < 1:
                    return

                center_T = np.zeros((4, 4), dtype=np.float32)
                for n in np.argwhere(arm_centering_map).flatten():
                    center_find_marker = arm_centering_position[n]
                    pose_matrix = find_pose(center_find[0:4, 0:3], center_find_marker)
                    center_T += pose_matrix

                center_T /= np.sum(arm_centering_map)
                center_point = transform_points(center_find[[-1], 0:3], center_T)[0]
                center_circle = (
                    transform_points(center_find_circle, center_T) @ optical_render.T
                )
                center_tip = (
                    transform_points(center_find_tip, center_T) @ optical_render.T
                )

                processing_queue.put(center_point)

                glPointSize(6.0)
                glBegin(GL_POINTS)
                glColor4f(*center_color)
                glVertex3f(*center_point @ optical_render.T)
                glEnd()

                glBegin(GL_LINE_STRIP)
                glColor4f(*center_color)
                for point in center_circle:
                    glVertex3f(*point)
                glEnd()
                glBegin(GL_LINE_STRIP)
                glColor4f(*center_color)
                for point in center_tip:
                    glVertex3f(*point)
                glEnd()

                draw_aruco_surface(markers_position[markers_map])
                optical_render.draw_points(points)

            imgui.new_frame()
            imgui.begin("ImGui Window")
            imgui.text("Hello, world!")
            imgui.end()
            imgui.render()
            imgui_impl.process_inputs()
            imgui_impl.render(imgui.get_draw_data())

        optical_render.update(window, draw)

    imgui_impl.shutdown()
    optical_render.terminate()


def no_block_wait(event, secs):
    start = time.time()
    while not event.is_set() and (time.time() - start) < secs:
        pass


def empty_queue(queue):
    recv = None
    while True:
        try:
            recv = queue.get_nowait()
        except Empty:
            break
    return recv


def processing(
    stop_event, points_data, processing_queue, evaluate_event, evaluate_pipe
):
    ser = serial.Serial(port="COM6", baudrate=115200, dsrdtr=True)

    send_packet(ser, packet_t(cmd=commands.RESET))
    assert recv_packet(ser).cmd == commands.ACK

    send_packet(ser, packet_t(cmd=commands.HOME_CARRIAGE))
    assert recv_packet(ser).cmd == commands.ACK

    send_packet(ser, packet_t(cmd=commands.HOME_SERVO))
    assert recv_packet(ser).cmd == commands.ACK

    print("hardware reset")

    carriage_ready = False
    while not carriage_ready:
        no_block_wait(stop_event, 0.05)
        send_packet(
            ser, packet_t(buffer=struct.pack("<f", 0.0), cmd=commands.MOVE_CARRIAGE)
        )
        carriage_ready = recv_packet(ser).cmd != commands.ERR
    print("hardware ready")

    empty_queue(processing_queue)
    print("processing queue realtime")

    initial_point = None
    while not stop_event.is_set():
        try:
            center_point = processing_queue.get_nowait()
        except Empty:
            continue

        if type(initial_point) == type(None):
            initial_point = center_point
            target_point = initial_point + [0.1, 0.00, 0.00]
            print(f"processing sent target: {target_point}")
            evaluate_pipe.send(target_point)

        if evaluate_event.is_set():
            evaluate_event.clear()
            vector_recv = evaluate_pipe.recv()
            print(f"processing evaluate {vector_recv}")
            vector = [*vector_recv, 0, 0, 0, 0, 0, 0]
            vector = np.clip(vector, 0, 180).astype(np.uint8)

            send_packet(
                ser,
                packet_t(buffer=struct.pack("<9B", *vector), cmd=commands.MOVE_SERVO),
            )
            print("processing packet sent")
            no_block_wait(stop_event, 2.0)
            pos = empty_queue(processing_queue)
            print(f"processing get most recent pos: {pos}")
            evaluate_pipe.send(pos)

    ser.close()


def optimize(stop_event, evaluate_event, evaluate_pipe):
    target_point = evaluate_pipe.recv()

    x0 = np.array([0.0, 0.0, 0.0])
    print(f"initial guess: {x0}")

    def objective(vector):
        print(f"optimize evaluate {vector}")
        evaluate_event.set()
        evaluate_pipe.send(vector)
        center_point = evaluate_pipe.recv()
        print(f"optimize recv {center_point}")
        return np.mean((target_point - center_point) ** 2)

    initial_simplex = np.array(
        [
            x0,
            x0 + np.array([180.0, 0.0, 0.0]),
            x0 + np.array([0.0, 180.0, 0.0]),
            x0 + np.array([0.0, 0.0, 180.0]),
        ]
    )

    result = scipy.optimize.minimize(
        objective,
        x0,
        method="Nelder-Mead",
        bounds=[(0, 180), (0, 180), (0, 180)],
        options={
            "xatol": 0.0004,  # Convergence tolerance on x
            "fatol": 0.0004,  # Convergence tolerance on function value
            "initial_simplex": initial_simplex,  # Larger initial simplex for faster exploration
            "maxiter": 20,  # Maximum number of iterations
            "adaptive": True,  # Disable adaptive mode for more control
        },
    )

    print("The minimum is at:", result.x)
    print("Minimum value of the objective function:", result.fun)


def setup(stop_event, points_data):
    evaluate_event, (evaluate_pipe_0, evaluate_pipe_1) = Event(), Pipe()
    processing_queue = Queue()

    return (
        Process(
            target=rendering,
            args=(stop_event, points_data, processing_queue),
            daemon=True,
        ),
        Process(
            target=processing,
            args=(
                stop_event,
                points_data,
                processing_queue,
                evaluate_event,
                evaluate_pipe_0,
            ),
            daemon=True,
        ),
        Process(
            target=optimize,
            args=(stop_event, evaluate_event, evaluate_pipe_1),
            daemon=True,
        ),
    )


def main():
    optical_pose.main(setup)


if __name__ == "__main__":
    main()
