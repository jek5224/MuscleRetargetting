import os
import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

def load_obj(filename):
    vertices = []
    faces = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = tuple(map(float, parts[1:4]))
                vertices.append(vertex)
            elif line.startswith('f '):
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    
    print(f"Loaded {len(vertices)} vertices and {len(faces)} faces from {filename}")
    return vertices, faces

# Function to draw OBJ data
def draw_obj(vertices, faces):
    for face in faces:
        if len(face) == 3:  # Triangle
            glBegin(GL_TRIANGLES)
            for vertex_idx in face:
                glVertex3fv(vertices[vertex_idx])
            glEnd()
        elif len(face) == 4:  # Quad
            glBegin(GL_QUADS)
            for vertex_idx in face:
                glVertex3fv(vertices[vertex_idx])
            glEnd()

# Load multiple OBJ files
def load_objs_from_directory(directory):
    obj_files = [f for f in os.listdir(directory) if f.endswith('.obj')]
    models = []
    
    for obj_file in obj_files:
        vertices, faces = load_obj(os.path.join(directory, obj_file))
        models.append((vertices, faces))
    
    return models

# OpenGL display function
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
    
    glColor3f(1, 0, 0)  # Red color for debugging
    glBegin(GL_TRIANGLES)
    glVertex3f(-1, -1, 0)
    glVertex3f(1, -1, 0)
    glVertex3f(0, 1, 0)
    glEnd()
    
    glPushMatrix()
    glScalef(0.01, 0.01, 0.01)
    for i, (vertices, faces) in enumerate(models):
        glPushMatrix()
        glTranslatef((i - len(models)/2) * 2.5, 0, 0)  # Space out models
        draw_obj(vertices, faces)
        glPopMatrix()
    glPopMatrix()
    
    glutSwapBuffers()

# OpenGL reshape function
def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / float(height), 1, 100)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)  # Try GL_CW if nothing shows up
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_with_obj_files>")
        sys.exit(1)
    
    directory = sys.argv[1]
    models = load_objs_from_directory(directory)
    
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"OBJ Viewer")
    
    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMainLoop()
