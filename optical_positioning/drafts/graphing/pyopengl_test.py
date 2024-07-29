import glfw
from OpenGL.GL import *
import imgui
from imgui.integrations.glfw import GlfwRenderer


def main():
    if not glfw.init():
        raise Exception("GLFW can't be initialized")

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "OpenGL with ImGui", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")

    glfw.make_context_current(window)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Initialize ImGui
    imgui.create_context()
    imgui_impl = GlfwRenderer(window)

    # Define a basic vertex shader and fragment shader
    vertex_shader_code = """
    #version 330
    in vec3 vertexPosition;
    void main() {
        gl_Position = vec4(vertexPosition, 1.0);
    }
    """

    fragment_shader_code = """
    #version 330
    out vec4 FragColor;
    void main() {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red color
    }
    """

    def compile_shader(source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise Exception(glGetShaderInfoLog(shader).decode())
        return shader

    def create_shader_program():
        vertex_shader = compile_shader(vertex_shader_code, GL_VERTEX_SHADER)
        fragment_shader = compile_shader(fragment_shader_code, GL_FRAGMENT_SHADER)
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise Exception(glGetProgramInfoLog(program).decode())
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        return program

    shader_program = create_shader_program()

    # Define vertices for a triangle
    vertices = [
        0.0,
        0.5,
        0.0,
        -0.5,
        -0.5,
        0.0,
        0.5,
        -0.5,
        0.0,
    ]

    # Create a vertex buffer object (VBO) and a vertex array object (VAO)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(
        GL_ARRAY_BUFFER,
        len(vertices) * 4,
        (GLfloat * len(vertices))(*vertices),
        GL_STATIC_DRAW,
    )

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)
        glBindVertexArray(vao)

        # Draw a triangle
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glBindVertexArray(0)

        # Start ImGui frame
        imgui.new_frame()

        # Create an ImGui window
        imgui.begin("ImGui Window")
        imgui.text("Hello, world!")
        imgui.end()

        # Render ImGui
        imgui.render()
        imgui_impl.process_inputs()
        imgui_impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)
        glfw.poll_events()

    imgui_impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
