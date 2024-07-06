import glfw
from OpenGL.GL import *
import imgui
from imgui.integrations.glfw import GlfwRenderer


def main():
    # Initialize GLFW
    if not glfw.init():
        return

    # Create a GLFW window
    window = glfw.create_window(1280, 720, "GLFW + PyOpenGL + ImGui", None, None)
    if not window:
        glfw.terminate()
        return

    # Set context
    glfw.make_context_current(window)

    # Initialize ImGui
    imgui.create_context()
    impl = GlfwRenderer(window)

    while not glfw.window_should_close(window):
        # Poll for and process events
        glfw.poll_events()
        impl.process_inputs()

        # Start a new ImGui frame
        imgui.new_frame()

        # Example ImGui window
        imgui.begin("Hello, world!")
        imgui.text("This is some useful text.")
        imgui.end()

        # Clear the screen
        glClearColor(0.16, 0.16, 0.16, 1.00)
        glClear(GL_COLOR_BUFFER_BIT)

        # Render ImGui
        imgui.render()
        impl.render(imgui.get_draw_data())

        # Swap front and back buffers
        glfw.swap_buffers(window)

    # Cleanup
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
