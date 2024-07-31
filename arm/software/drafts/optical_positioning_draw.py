from queue import Empty
from multiprocessing import Process

from optical_positioning import d_compute_pose as optical_pose
from optical_positioning import e_render as optical_render


from OpenGL.GL import glBegin, glEnd, GL_TRIANGLE_FAN, glColor4f, glVertex3f


def draw_aruco_surface(markers_position):
    for marker in markers_position:
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.7, 0.7, 0.7, 0.2)
        for point in marker:
            glVertex3f(*point @ optical_render.T)

        glEnd()


def rendering(stop_event, points_data, processing_queue):
    print("rendering started")
    window = optical_render.init()
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

                draw_aruco_surface(markers_position[markers_map])
                optical_render.draw_points(points)

        optical_render.update(window, draw)

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
