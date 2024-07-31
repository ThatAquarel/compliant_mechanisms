from queue import Empty
from multiprocessing import Process

import imgui
import numpy as np
import scipy.optimize
from OpenGL.GL import *

from optical_positioning import d_compute_pose as optical_pose
from optical_positioning import e_render as optical_render


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


def processing(stop_event, points_data, processing_queue):
    print("processing started")


def setup(stop_event, points_data, processing_queue):
    return Process(
        target=rendering, args=(stop_event, points_data, processing_queue), daemon=True
    ), Process(
        target=processing, args=(stop_event, points_data, processing_queue), daemon=True
    )


def main():
    optical_pose.main(setup)


if __name__ == "__main__":
    main()
