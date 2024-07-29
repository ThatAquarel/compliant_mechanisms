import glfw

from OpenGL.GL import *
from OpenGL.GLU import *

angle_x, angle_y = 0.0, 0.0
pan_x, pan_y = 0.0, 0.0
last_x, last_y = 0.0, 0.0
dragging = False
panning = False


def mouse_button_callback(window, button, action, mods):
    global dragging, panning
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            dragging = True
            panning = False
        elif action == glfw.RELEASE:
            dragging = False
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        if action == glfw.PRESS:
            dragging = True
            panning = True
        elif action == glfw.RELEASE:
            dragging = False
            panning = False


def mouse_callback(window, xpos, ypos):
    global last_x, last_y, angle_x, angle_y, pan_x, pan_y, dragging, panning
    if dragging:
        dx = xpos - last_x
        dy = ypos - last_y
        if panning:
            pan_x += dx * 0.001
            pan_y -= dy * 0.001
        else:
            angle_x += dy * 0.1
            angle_y += dx * 0.1
    last_x, last_y = xpos, ypos


def draw_axes():
    glBegin(GL_LINES)

    # X axis (red)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(-1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 0.0)

    # Y axis (green)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, -1.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)

    # Z axis (blue)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0.0, 0.0, -1.0)
    glVertex3f(0.0, 0.0, 1.0)

    glEnd()


def init():
    global last_x, last_y

    if not glfw.init():
        raise Exception("GLFW could not be initialized.")

    window = glfw.create_window(
        800, 800, "Orthographic Axes with Mouse Drag and Pan", None, None
    )
    if not window:
        glfw.terminate()
        raise Exception("GLFW window could not be created.")

    glfw.make_context_current(window)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)

    last_x, last_y = glfw.get_cursor_pos(window)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -0.5, 1.5, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    return window


def update(window, draw):
    if glfw.window_should_close(window):
        glfw.terminate()
        return False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.96, 0.97, 0.97, 1.0)
    glLoadIdentity()
    glTranslatef(pan_x, pan_y, 0.0)
    glRotatef(angle_x, 1.0, 0.0, 0.0)
    glRotatef(angle_y, 0.0, 1.0, 0.0)

    draw_axes()
    draw()

    glfw.swap_buffers(window)
    glfw.poll_events()

    return True
