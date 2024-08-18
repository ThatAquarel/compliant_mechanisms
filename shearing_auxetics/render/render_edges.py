import time
import glfw

from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from stl import mesh


def load_stl(filename):
    # Load the STL file
    return mesh.Mesh.from_file(filename)


def set_edge_color(edge_vector):
    # Normalize the edge vector
    norm = np.linalg.norm(edge_vector)
    if norm == 0:
        color = [1.0, 1.0, 1.0]  # Default to white if the vector is degenerate
    else:
        normalized_vector = edge_vector / norm
        color = np.abs(normalized_vector)  # Use absolute value to avoid negative colors
    glColor3f(color[0], color[1], color[2])  # Set the color based on orientation


def normalize(vectors):
    return 2 * (vectors - vectors.min()) / (vectors.max() - vectors.min()) - 1


def draw_edges(stl_mesh, len_limit):
    glBegin(GL_LINES)

    vectors = normalize(stl_mesh.vectors)
    for facet in vectors:
        for edge in [(0, 1), (1, 2), (2, 0)]:
            vertex_start = facet[edge[0]]
            vertex_end = facet[edge[1]]
            edge_vector = vertex_end - vertex_start

            if np.linalg.norm(edge_vector) <= len_limit:
                set_edge_color(edge_vector)
                glVertex3fv(vertex_start)
                glVertex3fv(vertex_end)
    glEnd()


def draw_faces(stl_mesh):
    vectors = normalize(stl_mesh.vectors)

    glBegin(GL_TRIANGLES)
    for facet in range(len(stl_mesh)):
        normal = stl_mesh.normals[facet]
        glNormal3fv(normal)  # Set the normal for the current face
        for vertex in vectors[facet]:
            glVertex3fv(vertex)
    glEnd()


def main():
    # Initialize GLFW
    if not glfw.init():
        return

    # Set the window size and create the window
    window = glfw.create_window(1000, 1000, "STL Viewer", None, None)
    if not window:
        glfw.terminate()
        return

    # Set the OpenGL context
    glfw.make_context_current(window)

    # Load the STL file
    # stl_mesh = load_stl("shearing_auxetics/openscad/flat_ratio_equal.stl")
    # stl_mesh = load_stl("shearing_auxetics/openscad/flat_asymmetric_ratio_0.5.stl")
    stl_mesh = load_stl("shearing_auxetics/openscad/flat_asymmetric_ratio_2.stl")

    # Enable depth test for proper occlusion
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)

    # Enable lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    # Define light properties
    light_position = [10, 10, 10, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Define semi-transparent material properties
    glMaterialfv(
        GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.5, 0.5, 0.5, 0.25]
    )  # Alpha = 0.5 for transparency
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 0.5])
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

    edge_steps = 20
    steps = np.linspace(0, 1, edge_steps)
    step_time = 0.025
    i = 0

    start_time = time.time()

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.86, 0.87, 0.87, 1.0)

        # Set the projection to orthogonal
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-2, 2, -2, 2, -2, 2)

        # Set the model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Rotate the view for better visualization (optional)
        glRotatef(35.264, 1, 0, 0)  # Rotate around x-axis by 35.264 degrees
        glRotatef(45, 0, 1, 0)  # Rotate around y-axis by 45 degrees

        glDisable(GL_LIGHTING)
        glColor3f(1, 1, 1)  # Set edge color to white
        n = i % 40
        if n < edge_steps:
            draw_edges(stl_mesh, steps[n])
        else:
            draw_edges(stl_mesh, steps[-1])
            glEnable(GL_LIGHTING)
            draw_faces(stl_mesh)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

        if (step_time * i + start_time) <= time.time():
            i += 1

    glfw.terminate()


if __name__ == "__main__":
    main()
