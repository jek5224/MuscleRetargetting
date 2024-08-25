from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import quaternion
import math

def drawGround(height):
    glColor3f(0.8, 0.8, 0.8)
    glDisable(GL_LIGHTING)
    
    # glBegin(GL_QUADS)
    # glVertex3f(-50, height, -50)
    # glVertex3f(50, height, -50)
    # glVertex3f(50, height, 50)
    # glVertex3f(-50, height, 50)
    # glEnd()
    
    ## When Rendering Video
    w = 0.005
    count = 0
    for x in range(-20, 20):
        for z in range(-21, 20):
            if count % 2 == 0:
                glColor3f(0.9, 0.9, 0.9)
            else:
                glColor3f(0.8, 0.8, 0.8)
            glBegin(GL_QUADS)
    
            glVertex3f(x, height, z)
            glVertex3f(x + 1, height, z)
            glVertex3f(x + 1, height, z + 1)
            glVertex3f(x, height, z + 1)
            glEnd()
            count += 1

    glEnable(GL_LIGHTING)

def draw_axis(pos = np.array([0.0,0.0,0.0]), ori = np.quaternion(1.0, 0.0, 0.0, 0.0)):
    glPushMatrix()

    glTranslatef(pos[0], pos[1], pos[2])
    q = quaternion.as_rotation_vector(ori)
    glRotatef(np.rad2deg(np.linalg.norm(q)), q[0], q[1], q[2])

    glDisable(GL_LIGHTING)
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 0.0)
    glEnd()

    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)
    glEnd()

    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 1.0)
    glEnd()
    glEnable(GL_LIGHTING)
    glPopMatrix()

def draw_cube(size):
    glScaled(size[0], size[1], size[2])

    n = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0]
    ])
    vn = np.array([
            [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
            [-1.0 / 3.0, -1.0 / 3.0,  1.0 / 3.0],
            [-1.0 / 3.0,  1.0 / 3.0,  1.0 / 3.0],
            [-1.0 / 3.0,  1.0 / 3.0, -1.0 / 3.0],
            [ 1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
            [ 1.0 / 3.0, -1.0 / 3.0,  1.0 / 3.0],
            [ 1.0 / 3.0,  1.0 / 3.0,  1.0 / 3.0],
            [ 1.0 / 3.0,  1.0 / 3.0, -1.0 / 3.0]
        ])
    faces = np.array([
        [0,1,2,3],
        [3,2,6,7],
        [7,6,5,4],
        [4,5,1,0],
        [5,6,2,1],
        [7,4,0,3]
    ])

    v = np.zeros([8,3])


    v[0,0] = v[1,0] = v[2,0] = v[3,0] = -1.0/2
    v[4,0] = v[5,0] = v[6,0] = v[7,0] =  1.0/2
    v[0,1] = v[1,1] = v[4,1] = v[5,1] = -1.0/2
    v[2,1] = v[3,1] = v[6,1] = v[7,1] =  1.0/2
    v[0,2] = v[3,2] = v[4,2] = v[7,2] = -1.0/2
    v[1,2] = v[2,2] = v[5,2] = v[6,2] =  1.0/2
    
    for i in range(5,-1,-1):
        glBegin(GL_QUADS)
        glNormal3fv(n[i])
        for j in range(4):
            glVertex3fv(v[faces[i, j]])
        glEnd()

def draw_sphere(radius, slices, stacks):
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = math.sin(lat0)
        zr0 = math.cos(lat0)

        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = math.sin(lat1)
        zr1 = math.cos(lat1)

        # Start drawing quadrilateral strips
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j - 1) / slices
            x = math.cos(lng)
            y = math.sin(lng)

            # glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)

            # glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)

        glEnd()

