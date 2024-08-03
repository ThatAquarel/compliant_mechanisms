import time
import struct
from queue import Empty
from multiprocessing import Process, Queue

import imgui
import serial
import numpy as np
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


carriage_position = 0
servos_position = np.zeros(9, dtype=np.uint8)
frame_time = 0
frame_count = 0
frame_dt = 0
fps = 0


def rendering(stop_event, points_data, processing_queue):
    window = optical_render.init()
    imgui_impl = optical_render.init_imgui(window)
    imgui.style_colors_light()

    scatter = None
    while not stop_event.is_set() and not optical_render.window_should_close(window):
        while True:
            try:
                scatter = points_data.get_nowait()
            except Empty:
                break

        def draw():
            global carriage_position, servos_position, frame_time, frame_count, frame_dt, fps
            t = time.time()
            frame_dt += t - frame_time
            frame_time = t
            frame_count += 1

            if frame_count == 10:
                dt = frame_dt / 10
                fps = 1 / dt

                frame_dt = 0
                frame_count = 0

            imgui.new_frame()
            imgui.begin("controls")

            imgui.text("render")
            imgui.separator()
            imgui.text(f"fps: {fps:.4f}")

            if imgui.button("print positions"):
                print(f"servos positions: {repr(servos_position)}")
                print(f"carriage position: {repr(carriage_position)}")

            imgui.spacing()
            imgui.spacing()

            imgui.text("mechanical")
            imgui.separator()
            imgui.same_line()
            if imgui.button("home carriage"):
                processing_queue.put(packet_t(cmd=commands.HOME_CARRIAGE))
            imgui.same_line()
            if imgui.button("home servos"):
                processing_queue.put(packet_t(cmd=commands.HOME_SERVO))
            imgui.same_line()
            if imgui.button("reset"):
                processing_queue.put(packet_t(cmd=commands.RESET))

            imgui.spacing()
            imgui.spacing()

            imgui.text("servos (deg)")
            servo_changed = False
            for i, servo in enumerate(servos_position):
                if i % 3 == 0:
                    imgui.separator()
                stat, servos_position[i] = imgui.slider_int(f"{i}", servo, 0, 180)
                servo_changed |= stat
            if servo_changed:
                servo_buffer = struct.pack("<9B", *servos_position)
                packet = packet_t(buffer=servo_buffer, cmd=commands.MOVE_SERVO)
                processing_queue.put(packet)

            imgui.spacing()
            imgui.spacing()

            imgui.text("carriage (mm)")
            imgui.separator()
            carriage_changed, carriage_position = imgui.slider_float(
                "", carriage_position, 0.0, 300.0
            )
            if carriage_changed:
                steps_per_mm = 1000 / (60 * 2)
                carriage_buffer = struct.pack("<f", carriage_position * steps_per_mm)
                packet = packet_t(buffer=carriage_buffer, cmd=commands.MOVE_CARRIAGE)
                processing_queue.put(packet)

            imgui.end()

            imgui.render()
            imgui_impl.process_inputs()
            imgui_impl.render(imgui.get_draw_data())

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

        optical_render.update(window, draw)

    imgui_impl.shutdown()
    optical_render.terminate()


def processing(stop_event, points_data, processing_queue):
    try:
        ser = serial.Serial(port="COM6", baudrate=115200, dsrdtr=True)
    except serial.SerialException:
        print(f"serial connection failure")
        return

    while not stop_event.is_set():
        try:
            packet = processing_queue.get_nowait()
        except Empty:
            continue

        send_packet(ser, packet)
        ret = recv_packet(ser).cmd
        if ret != commands.ACK:
            print(f"packet sent error: {packet.cmd}")
            print(f"packet recv status: {ret}")


def setup(stop_event, points_data):
    processing_queue = Queue()

    return Process(
        target=rendering, args=(stop_event, points_data, processing_queue), daemon=True
    ), Process(
        target=processing, args=(stop_event, points_data, processing_queue), daemon=True
    )


def main():
    optical_pose.main(setup)


if __name__ == "__main__":
    main()
