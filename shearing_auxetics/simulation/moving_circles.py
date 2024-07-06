import glfw
from OpenGL.GL import *
import numpy as np
import ctypes

vertex_shader_code = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 offset;
void main() {
    gl_Position = vec4(position * 0.01 + offset, 0.0, 1.0);
}
"""

fragment_shader_code = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.1, 0.8, 0.3, 1.0);
}
"""


# Compile shader
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode("utf-8"))
    return shader


# Create shader program
def create_shader_program(vertex_code, fragment_code):
    vertex_shader = compile_shader(vertex_code, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_code, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode("utf-8"))
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program


# Function to generate circle vertices
def create_circle_vertices(num_segments):
    vertices = []
    for i in range(num_segments):
        theta = 2.0 * np.pi * i / num_segments
        x = np.cos(theta)
        y = np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices, dtype=np.float32)


def main():
    # Initialize GLFW
    if not glfw.init():
        return

    # Create a GLFW window
    window = glfw.create_window(1280, 720, "OpenGL Dynamic Circles", None, None)
    if not window:
        glfw.terminate()
        return

    # Set context
    glfw.make_context_current(window)

    # Create shader program
    shader_program = create_shader_program(vertex_shader_code, fragment_shader_code)
    glUseProgram(shader_program)

    # Generate circle vertex data
    num_segments = 50
    circle_vertices = create_circle_vertices(num_segments)

    # Initial positions for the circles
    num_circles = 10000
    circle_positions = (
        np.random.rand(num_circles, 2) * 2 - 1
    )  # Random positions between -1 and 1

    # Create VBO for circle vertices
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(
        GL_ARRAY_BUFFER, circle_vertices.nbytes, circle_vertices, GL_STATIC_DRAW
    )

    # Create VBO for instance positions
    instance_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
    glBufferData(
        GL_ARRAY_BUFFER, circle_positions.nbytes, circle_positions, GL_DYNAMIC_DRAW
    )

    # Create and bind VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # Bind circle vertices VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    # Bind instance positions VBO
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
    glVertexAttribDivisor(1, 1)  # Tell OpenGL this is an instanced vertex attribute

    while not glfw.window_should_close(window):
        # Poll for and process events
        glfw.poll_events()

        # Update positions (example: move circles in a random direction)
        circle_positions += (np.random.rand(num_circles, 2) - 0.5) * 0.01

        # Update instance VBO with new positions
        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, circle_positions.nbytes, circle_positions)

        # Clear the screen
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Draw instanced circles
        glBindVertexArray(vao)
        glDrawArraysInstanced(GL_LINE_LOOP, 0, num_segments, num_circles)

        # Swap front and back buffers
        glfw.swap_buffers(window)

    # Cleanup
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo, instance_vbo])
    glDeleteProgram(shader_program)
    glfw.terminate()


if __name__ == "__main__":
    main()
