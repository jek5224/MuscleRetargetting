import igl
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw


# GLOBAL STATE
window = None
V_np, original_faces, new_faces = None, None, None

def init_opengl():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)
    glLineWidth(1.5)

def draw_faces(faces, color):
    glColor4f(*color)
    glBegin(GL_TRIANGLES)
    for tri in faces:
        for vidx in tri:
            glVertex3fv(V_np[vidx])
    glEnd()

def draw_edges(faces, color):
    glColor3f(*color)
    glBegin(GL_LINES)
    for tri in faces:
        for i in range(3):
            v1 = V_np[tri[i]]
            v2 = V_np[tri[(i+1)%3]]
            glVertex3fv(v1)
            glVertex3fv(v2)
    glEnd()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -2.5)

    draw_faces(original_faces, (1.0, 0.5, 0.5, 0.3))
    draw_edges(original_faces, (1.0, 0.0, 0.0))

    draw_faces(new_faces, (0.5, 0.5, 1.0, 0.3))
    draw_edges(new_faces, (0.0, 0.0, 1.0))

    glfw.swap_buffers(window)

def main(obj_path):
    global window, V_np, original_faces, new_faces

    # Load mesh
    V, F = igl.read_triangle_mesh(obj_path)
    from igl import tetwild
    options = tetwild.TetwildOptions()
    options.simplify_mesh = True
    options.quality = 2.0

    V_tet = igl.eigen.MatrixXd()
    T = igl.eigen.MatrixXi()
    F_surface = igl.eigen.MatrixXi()
    F_tags = igl.eigen.MatrixXi()

    tetwild.tetrahedralize(V, F, options, V_tet, T, F_surface, F_tags)

    V_np = np.array(V_tet)
    F_np = np.array(F_surface)
    tags_np = np.array(F_tags).flatten()
    original_faces = F_np[tags_np == 1]
    new_faces = F_np[tags_np == 0]

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    window = glfw.create_window(800, 600, "Tetrahedral Visualization", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    init_opengl()

    while not glfw.window_should_close(window):
        display()
        glfw.poll_events()

    glfw.terminate()

# Example usage:
# main("your_model.obj")

path = "../Zygote_Meshes_Revised_Subdivided/Muscle/L_Adductor_Magnus.obj"
if __name__ == "__main__":
    main(path)