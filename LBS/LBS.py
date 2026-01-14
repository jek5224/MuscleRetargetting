from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import glm
import numpy as np
import copy
import time

width = 1500
height = 800

scale = 100

frames = 10
frametime = 1 / 30

nowframe = 0
framenum = 1

view = glm.normalize(glm.vec3(0.0, 0.0, 1.0))
lookat = glm.vec3(0.0, 0.0, 0.0)
viewangle = 60.0
d = 30
view_rot_angle = 0
rot_pause = True

LBS = 0
DBS = False
blendshape = False
rotate_show = True
text_show = True

DBS_was_on = False
blendshape_was_on = False

graph_x_s = -20
graph_x_e = 20
graph_x_m = (graph_x_s + graph_x_e) / 2
graph_y_s = -15
graph_y_e = -7.5
graph_y_m = (graph_y_s + graph_y_e) / 2


up_joint = glm.vec3(-20, 0, 0)
down_joint = glm.vec3(0, 0, 0)

# Defining in local coordinate
# for i in range(vertex_n + 1):
#     up_vertices.append(glm.vec3(20 - 20 * i / vertex_n, 5, 0))
# for i in range(vertex_n + 1):
#     up_vertices.append(glm.vec3(20 - 20 * i / vertex_n, -5, 0))

# Global coordinate of vertices
vertex_n = 30
slice_n = 4

def vertex_reset(v_n, s_n):
    global up_vertices, down_vertices

    up_vertices = []
    down_vertices = []

    slice_n = s_n
    vertex_n = v_n

    for k in range(slice_n):
        for i in range(vertex_n):
            up_vertices.append(up_joint + glm.vec3(20 * (i + 1) / vertex_n, \
                                                5 * np.cos(k / slice_n * 2 * np.pi + 1 / slice_n * np.pi),
                                                5 * np.sin(k / slice_n * 2 * np.pi + 1 / slice_n * np.pi)))

    for k in range(slice_n):
        for i in range(vertex_n):
            down_vertices.append(down_joint + glm.vec3(20 * i / vertex_n, \
                                                5 * np.cos(k / slice_n * 2 * np.pi + 1 / slice_n * np.pi),
                                                5 * np.sin(k / slice_n * 2 * np.pi + 1 / slice_n * np.pi)))
vertex_reset(vertex_n, slice_n)

up_curve_ratio = 0.5
down_curve_ratio = 0.5

def weight_reset(up_curve_r, down_curve_r):
    global up_curve_ratio, down_curve_ratio
    global up_curve_n, down_curve_n
    global w_up_no, w_down_no
    global w_up_LBS, w_down_LBS
    global w_up_LBS_quad, w_down_LBS_quad
    global w_up_LBS_cube, w_down_LBS_cube
    
    w_up_no = []
    w_down_no = []

    for i in range(len(up_vertices)):
        w_up_no.append([1, 0])
    for i in range(len(down_vertices)):
        w_down_no.append([0, 1])

    w_up_LBS = copy.deepcopy(w_up_no)
    w_down_LBS = copy.deepcopy(w_down_no)

    w_up_LBS_quad = copy.deepcopy(w_up_LBS)
    w_down_LBS_quad = copy.deepcopy(w_down_LBS)

    w_up_LBS_cube = copy.deepcopy(w_up_LBS)
    w_down_LBS_cube = copy.deepcopy(w_down_LBS)

    up_curve_ratio = up_curve_r
    down_curve_ratio = down_curve_r
    up_curve_n = int(vertex_n * up_curve_ratio)
    down_curve_n = int(vertex_n * down_curve_ratio)

    for k in range(slice_n):
        for i in range(up_curve_n):
            x = i / up_curve_n 
            w_up_LBS[(k + 1) * vertex_n - 1 - i][0] = 0.5 + 0.5 * x
            w_up_LBS[(k + 1) * vertex_n - 1 - i][1] = 1 - w_up_LBS[(k + 1) * vertex_n - 1 - i][0]

            w_up_LBS_quad[(k + 1) * vertex_n - 1 - i][0] = -0.5 * (x - 1) ** 2 + 1
            w_up_LBS_quad[(k + 1) * vertex_n - 1 - i][1] = 1 - w_up_LBS_quad[(k + 1) * vertex_n - 1 - i][0]

            w_up_LBS_cube[(k + 1) * vertex_n - 1 - i][0] = -(x ** 3) + 1.5 * x ** 2 + 0.5
            w_up_LBS_cube[(k + 1) * vertex_n - 1 - i][1] = 1 - w_up_LBS_cube[(k + 1) * vertex_n - 1 - i][0]

    for k in range(slice_n):
        for i in range(down_curve_n):
            x = i / down_curve_n
            w_down_LBS[k * vertex_n + i][1] = 0.5 + 0.5 * x
            w_down_LBS[k * vertex_n + i][0] = 1 - w_down_LBS[k * vertex_n + i][1]

            w_down_LBS_quad[k * vertex_n + i][1] = -0.5 * (x - 1) ** 2 + 1
            w_down_LBS_quad[k * vertex_n + i][0] = 1 - w_down_LBS_quad[k * vertex_n + i][1]

            w_down_LBS_cube[k * vertex_n + i][1] = -(x ** 3) + 1.5 * x ** 2 + 0.5
            w_down_LBS_cube[k * vertex_n + i][0] = 1 - w_down_LBS_cube[k * vertex_n + i][1]

weight_reset(up_curve_ratio, down_curve_ratio)

def drawSquare(midx, midy, h, color, angle, axis):
    glPushMatrix()
    glColor3f(0,0,0)
    glRotatef(angle, axis[0], axis[1], axis[2])

    glBegin(GL_LINE_LOOP)
    glVertex3f(midx + h / 2, midy + h / 2, 0)
    glVertex3f(midx - h / 2, midy + h / 2, 0)
    glVertex3f(midx - h / 2, midy - h / 2, 0)
    glVertex3f(midx + h / 2, midy - h / 2, 0)
    glEnd()

    glColor3f(color[0], color[1], color[2])
    glBegin(GL_POLYGON)
    glVertex3f(midx + h / 2, midy + h / 2, 0)
    glVertex3f(midx - h / 2, midy + h / 2, 0)
    glVertex3f(midx - h / 2, midy - h / 2, 0)
    glVertex3f(midx + h / 2, midy - h / 2, 0)
    glEnd()
    glPopMatrix()

def drawCube(midx, midy, midz, h, color, angle, axis):
    glPushMatrix()
    glTranslatef(midx, midy, midz)
    glRotatef(angle, axis[0], axis[1], axis[2])
    glColor3f(color[0], color[1], color[2])
    glScalef(h, h, h)
    glutSolidCube(1.0)
    glColor3f(0,0,0)
    glutWireCube(1.0001)
    glPopMatrix()

def drawCoordinate():
    glPushMatrix()
    # glTranslatef(x, y, z)
    # glRotatef(angle, axis[0], axis[1], axis[2])
    glBegin(GL_LINES)
    glColor3f(1,0,0)
    glVertex3f(0,0,0)
    glVertex3f(3,0,0)

    glColor3f(0,1,0)
    glVertex3f(0,0,0)
    glVertex3f(0,3,0)

    glColor3f(0,0,1)
    glVertex3f(0,0,0)
    glVertex3f(0,0,3)
    glEnd()

    glPushMatrix()
    glColor3f(0,0,1)
    glTranslatef(0,0,3)
    glutSolidCone(0.2,0.5,10,5)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0,3,0)
    glColor3f(0,1,0)
    glRotatef(90,-1,0,0)
    glutSolidCone(0.2,0.5,10,5)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(3,0,0)
    glColor3f(1,0,0)
    glRotatef(90,0,1,0)
    glutSolidCone(0.2,0.5,10,5)
    glPopMatrix()
    glPopMatrix()

def text(x, y, color, text):
        glColor3f(color[0], color[1], color[2])
        glWindowPos2f(x, y)

        glutBitmapString(GLUT_BITMAP_9_BY_15, text.encode('ascii'))

# Visualization part
def reshape(w, h):
    global width
    global height
    width = glutGet(GLUT_WINDOW_WIDTH)
    height = glutGet(GLUT_WINDOW_HEIGHT)

    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(viewangle, w / h, 1.0, 100000.0)
    gluLookAt(view.x * d, view.y * d, view.z * d, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)

    glutPostRedisplay()

def display(): 
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective (viewangle,  width/height, 1.0, 100000.0)
    gluLookAt(view[0] * d, view[1] * d, view[2] * d, lookat.x, lookat.y, lookat.z, 0.0, 1.0, 0.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # For visualizing
    # True LBS should be performed in global coordinate!
    # Let's avoid using Matrix overlapping
    # glPointSize(5.0)

    # glPushMatrix()
    # glColor3f(0,0,1)
    # glTranslatef(up_joint[0], up_joint[1], up_joint[2])
    # glRotatef(up_rotation, 0,0,1)
    # glBegin(GL_LINES)
    # for i in range(1, vertex_n + 1):
    #     glVertex3f(up_vertices[i][0], up_vertices[i][1], up_vertices[i][2])
    #     glVertex3f(up_vertices[i - 1][0], up_vertices[i - 1][1], up_vertices[i - 1][2])

    #     glVertex3f(up_vertices[i + 1 + vertex_n][0], up_vertices[i + 1 + vertex_n][1], up_vertices[i + 1 + vertex_n][2])
    #     glVertex3f(up_vertices[i + vertex_n][0], up_vertices[i + vertex_n][1], up_vertices[i + vertex_n][2])
    # glEnd()

    # drawSquare(0,0,0.5, [0,0,1])

    # glPushMatrix()
    # glColor3f(1,0,0)
    # glTranslatef(down_joint[0] - up_joint[0], down_joint[1] - up_joint[1], down_joint[2] - up_joint[2])
    # glRotatef(down_rotation, 0,0,1)
    # glBegin(GL_LINES)
    # for i in range(1, vertex_n + 1):
    #     glVertex3f(down_vertices[i][0], down_vertices[i][1], down_vertices[i][2])
    #     glVertex3f(down_vertices[i - 1][0], down_vertices[i - 1][1], down_vertices[i - 1][2])

    #     glVertex3f(down_vertices[i + 1 + vertex_n][0], down_vertices[i + 1 + vertex_n][1], down_vertices[i + 1 + vertex_n][2])
    #     glVertex3f(down_vertices[i + vertex_n][0], down_vertices[i + vertex_n][1], down_vertices[i + vertex_n][2])
    # glEnd()

    # drawSquare(0,0,0.5, [1,0,0])

    # glPopMatrix() 

    # glPopMatrix()

    global w_up_no, w_down_no

    w_up = w_up_no
    w_down = w_down_no
    if LBS >= 1:
        if LBS == 1:
            global w_up_LBS, w_down_LBS
            w_up = w_up_LBS
            w_down = w_down_LBS
        elif LBS == 2:
            global w_up_LBS_quad, w_down_LBS_quad
            w_up = w_up_LBS_quad
            w_down = w_down_LBS_quad
        else:
            global w_up_LBS_cube, w_down_LBS_cube
            w_up = w_up_LBS_cube
            w_down = w_down_LBS_cube
            

    nowtime = time.time()

    angles = []
    axises = []

    angle = 60 - 60 * np.cos(nowtime)
    axis = glm.vec3(0, 0, 1)

    angles.append(angle)
    axises.append(axis)

    # angle = 140 * np.cos(nowtime)
    # axis = glm.vec3(1, 0, 0)

    # angles.append(angle)
    # axises.append(axis)

    rot_mat = glm.mat4(1.0)

    for j in range(len(angles)):
        rot_mat = glm.rotate(rot_mat, glm.radians(angles[j]), axises[j])

    drawCube(up_joint[0], up_joint[1], up_joint[2], 0.5, glm.vec3(0,0,1), 0, axis)
    drawCube(down_joint[0], down_joint[1], down_joint[2], 0.5, glm.vec3(1,0,0), angle, axis)

    up_copy = copy.deepcopy(up_vertices)

    if blendshape and not DBS:
        for k in range(slice_n):
            for i in range(up_curve_n):
                index = (k + 1) * vertex_n - up_curve_n + i
                w_up_ki = w_up[index][0]
                w_down_ki = w_up[index][1]

                rot_index = glm.mat4(1.0)
                for j in range(len(angles)):
                    rot_index = glm.rotate(rot_index, glm.radians(angles[j] / 2 * (i + 1) / up_curve_n), axises[j])

                up_copy[index] += glm.inverse(w_up_ki * glm.mat4(1.0) + w_down_ki * rot_mat) *\
                                    (rot_index - w_up_ki * glm.mat4(1.0) - w_down_ki * rot_mat) * up_copy[index]

    if rotate_show:
        if not DBS:
            for i in range(len(up_copy)):
                up_copy[i] = w_up[i][0] * up_copy[i] + w_up[i][1] * rot_mat * up_copy[i]
        else:
            for i in range(len(up_copy)):
                # b = w_up[i][0] * glm.vec4(1, 0, 0, 0) + \
                #     w_up[i][1] * glm.vec4(np.cos(np.radians(angles[0])), 0, 0, np.sin(np.radians(angles[0])))
                b = w_up[i][0] * glm.vec4(1, 0, 0, 0) + \
                     w_up[i][1] * glm.vec4(np.cos(np.radians(angles[0])), np.sin(np.radians(angles[0])) * axises[0])
                b_norm = glm.length(b)
                b /= b_norm

                angle = np.arccos(b[0])
                axis = glm.vec3(1,0,0)
                if angle != 0:
                    denom = np.sqrt(1 - b[0] * b[0])
                    axis.x = b[1] / denom
                    axis.y = b[2] / denom
                    axis.z = b[3] / denom

                rot = glm.rotate(glm.mat4(1.0), angle, axis)
                up_copy[i] = rot * up_copy[i]

    glBegin(GL_LINES)
    for k in range(slice_n):
        for i in range(1, vertex_n):
            glColor3f(w_up[i][1], 0, w_up[i][0])
                
            glVertex3f(up_copy[i + k * vertex_n][0],\
                    up_copy[i + k * vertex_n][1],\
                    up_copy[i + k * vertex_n][2])
            glVertex3f(up_copy[i - 1 + k * vertex_n][0], \
                    up_copy[i - 1 + k * vertex_n][1], \
                    up_copy[i - 1 + k * vertex_n][2])
            
    for i in range(vertex_n):
        glColor3f(w_up[i][1], 0, w_up[i][0])
        for k in range(slice_n):
            if k != slice_n - 1:
                glVertex3f(up_copy[i + k * vertex_n][0],\
                           up_copy[i + k * vertex_n][1],\
                           up_copy[i + k * vertex_n][2])
                glVertex3f(up_copy[i + (k + 1) * vertex_n][0],\
                           up_copy[i + (k + 1) * vertex_n][1],\
                           up_copy[i + (k + 1) * vertex_n][2])
            else:
                glVertex3f(up_copy[i + k * vertex_n][0],\
                           up_copy[i + k * vertex_n][1],\
                           up_copy[i + k * vertex_n][2])
                glVertex3f(up_copy[i][0],\
                           up_copy[i][1],\
                           up_copy[i][2])

    down_copy = copy.deepcopy(down_vertices)

    if blendshape and not DBS:
        for k in range(slice_n):
            for i in range(down_curve_n):
                index = k * vertex_n + i
                w_up_ki = w_down[index][0]
                w_down_ki = w_down[index][1]
                rot_index = glm.mat4(1.0)
                for j in range(len(angles)):
                    rot_index = glm.rotate(rot_index, glm.radians(angles[j] / 2 + angles[j] / 2 * i / down_curve_n), axises[j])

                down_copy[index] += glm.inverse(w_up_ki * glm.mat4(1.0) + w_down_ki * rot_mat) *\
                                    (rot_index - w_up_ki * glm.mat4(1.0) - w_down_ki * rot_mat) * down_copy[index]
    
    if rotate_show:
        if not DBS:
            for i in range(len(down_copy)):
                down_copy[i] = w_down[i][0] * down_copy[i] + w_down[i][1] * rot_mat * down_copy[i]
        else:
            for i in range(len(down_copy)):
                # b = w_down[i][0] * glm.vec4(1, 0, 0, 0) + \
                #     w_down[i][1] * glm.vec4(np.cos(np.radians(angles[0])), 0, 0, np.sin(np.radians(angles[0])))
                b = w_down[i][0] * glm.vec4(1, 0, 0, 0) + \
                     w_down[i][1] * glm.vec4(np.cos(np.radians(angles[0])), np.sin(np.radians(angles[0])) * axises[0])
                b_norm = glm.length(b)
                b /= b_norm

                angle = np.arccos(b[0])
                axis = glm.vec3(1,0,0)
                if angle != 0:
                    denom = np.sqrt(1 - b[0] * b[0])
                    axis.x = b[1] / denom
                    axis.y = b[2] / denom
                    axis.z = b[3] / denom

                rot = glm.rotate(glm.mat4(1.0), angle, axis)
                down_copy[i] = rot * down_copy[i]

    glColor3f(1,0,1)
    for k in range(slice_n):
        glVertex3f(up_copy[(k + 1) * vertex_n - 1][0],\
                   up_copy[(k + 1) * vertex_n - 1][1],\
                   up_copy[(k + 1) * vertex_n - 1][2])
        glVertex3f(down_copy[k * (vertex_n)][0],\
                   down_copy[k * (vertex_n)][1],\
                   down_copy[k * (vertex_n)][2])

    for k in range(slice_n):
        for i in range(1, vertex_n):
            glColor3f(w_down[i][1], 0, w_down[i][0])

            glVertex3f(down_copy[i + k * vertex_n][0],\
                    down_copy[i + k * vertex_n][1],\
                    down_copy[i + k * vertex_n][2])
            glVertex3f(down_copy[i - 1 + k * vertex_n][0], \
                    down_copy[i - 1 + k * vertex_n][1], \
                    down_copy[i - 1 + k * vertex_n][2])
            
    for i in range(vertex_n):
        glColor3f(w_down[i][1], 0, w_down[i][0])               

        for k in range(slice_n):
            if k != slice_n - 1:
                glVertex3f(down_copy[i + k * vertex_n][0],\
                           down_copy[i + k * vertex_n][1],\
                           down_copy[i + k * vertex_n][2])
                glVertex3f(down_copy[i + (k + 1) * vertex_n][0],\
                           down_copy[i + (k + 1) * vertex_n][1],\
                           down_copy[i + (k + 1) * vertex_n][2])
            else:
                glVertex3f(down_copy[i + k * vertex_n][0],\
                           down_copy[i + k * vertex_n][1],\
                           down_copy[i + k * vertex_n][2])
                glVertex3f(down_copy[i][0],\
                           down_copy[i][1],\
                           down_copy[i][2])
                
    glEnd()

    # Drawing axis
    glPushMatrix()
    glTranslatef(up_joint[0], up_joint[1], up_joint[2])
    drawCoordinate()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(down_joint[0], down_joint[1], down_joint[2])
    for i in range(len(angles)):
        glRotatef(angles[i], axises[i][0], axises[i][1], axises[i][2])
    drawCoordinate()
    glPopMatrix()

    # Drawing graph
    glBegin(GL_LINES)
    glColor3f(0,0,0)
    # glVertex3f(graph_x_s, graph_y_m, 0)
    # glVertex3f(graph_x_e, graph_y_m, 0)

    glVertex3f(graph_x_m, graph_y_s, 0)
    glVertex3f(graph_x_m, graph_y_e, 0)

    for i in range(1, vertex_n):
        glColor3f(w_up[i][1], 0, w_up[i][0])
        glVertex3f(up_vertices[i - 1][0], graph_y_s + 5 * w_up[i - 1][0], 0)
        glVertex3f(up_vertices[i][0], graph_y_s + 5 * w_up[i][0], 0)

        glColor3f(w_up[i][0], 0, w_up[i][1])
        glVertex3f(up_vertices[i - 1][0], graph_y_s + 5 * w_up[i - 1][1], 0)
        glVertex3f(up_vertices[i][0], graph_y_s + 5 * w_up[i][1], 0)

        glColor3f(w_down[i][0], 0, w_down[i][1])
        glVertex3f(down_vertices[i - 1][0], graph_y_s + 5 * w_down[i - 1][0], 0)
        glVertex3f(down_vertices[i][0], graph_y_s + 5 * w_down[i][0], 0)

        glColor3f(w_down[i][1], 0, w_down[i][0])
        glVertex3f(down_vertices[i - 1][0], graph_y_s + 5 * w_down[i - 1][1], 0)
        glVertex3f(down_vertices[i][0], graph_y_s + 5 * w_down[i][1], 0)
    glEnd()
    text(width / 2 - 5, 0.05 * height, glm.vec3(0,0,0), '0')
    text(width / 2 + 2, 0.22 * height, glm.vec3(0,0,0), '1')
    glPushMatrix()
    glColor3f(0,0,0)
    glTranslatef(0, graph_y_e, 0)
    glRotatef(90,-1,0,0)
    glutSolidCone(0.2,0.5,10,5)
    glPopMatrix()

    # Text rendering
    
    if text_show:
        if blendshape:
            if rotate_show:
                text(50, height - 40, glm.vec3(0,0,0), 'Applying Blendshape')
            else:
                text(50, height - 40, glm.vec3(0,0,0), 'Applying Blendshape (joint rotation stop)')
        elif DBS:
            text(50, height - 40, glm.vec3(0,0,0), 'Dual Quaternion Blend Skinning')
        else:
            text(50, height - 40, glm.vec3(0,0,0), 'Linear Blend Skinning')
        
        text(50, height - 80, glm.vec3(0,0,0), 'l: Weight equation change, q: DQBS on/off, b: Blendshape on/off')

        text(50, height - 120, glm.vec3(0,0,0), 'c/C: Weight range change')
        text(50, height - 160, glm.vec3(0,0,0), 'x/X: Polygon change')
        text(50, height - 200, glm.vec3(0,0,0), 'v/V: Slice change')
        text(50, height - 240, glm.vec3(0,0,0), 'r: Joint rotation start/stop when blendshape is on')

        text(50, 120, glm.vec3(0,0,0), 'Space: Rotation start/stop')
        text(50, 80, glm.vec3(0,0,0), 'WASD: Viewpoint change')

        text(5, 5, glm.vec3(0,0,0),
            'From (' + str(np.round(view[0] * d, 3)) + ', ' + 
                    str(np.round(view[1] * d, 3)) + ', ' + 
                    str(np.round(view[2] * d, 3)) + ')')
    

    glutSwapBuffers()

def keyboard(key, x, y):
    global LBS, DBS
    global blendshape
    global DBS_was_on, blendshape_was_on, rotate_show
    global text_show
    if key == b'l':  
        LBS = (LBS + 1) % 4
        # if LBS == 0:
        #     print('LBS not working')
        # elif LBS == 1:
        #     print('LBS (Linear)')
        # elif LBS == 2:
        #     print('LBS (quadratic)')
        # elif LBS == 3:
        #     print('LBS (cubic)')
    elif key == b'q':
        DBS = not DBS
        if DBS and blendshape:
            blendshape = False
            blendshape_was_on = True
            rotate_show = True
        else:
            if blendshape_was_on:
                blendshape = True
                blendshape_was_on = False
        #print('Dual Quaternion:', DBS)
    elif key == b'b':
        blendshape = not blendshape
        if DBS and blendshape:
            DBS = False
            DBS_was_on = True
        else:
            if DBS_was_on:
                DBS = True
                DBS_was_on = False
        #print('BlendShape:', blendshape)
    elif key == b' ':
        global rot_pause
        rot_pause = not rot_pause
        #print('rot_pause:', rot_pause)
    elif key == b'r':
        if blendshape:
            rotate_show = not rotate_show
            #print('rotate_show:', rotate_show)
        else:
            rotate_show = True

    elif key in [b'w', b'a', b's', b'd']:
        if not rot_pause:
            rot_pause = True
        global view, view_rot_angle

        if key == b'w':
            view_rot_angle = 180
        elif key == b'a':
            view_rot_angle = -90
        elif key == b's':
            view_rot_angle = 0
        elif key == b'd':
            view_rot_angle = 90

    elif key in [b'v', b'V', b'x', b'X', b'c', b'C']:
        global vertex_n, slice_n
        global up_curve_ratio, down_curve_ratio

        if key in [b'v', b'V']:
            if key == b'v':
                vertex_n -= 1
                if vertex_n <= 3:
                    vertex_n = 3
            else:
                vertex_n += 1
        elif key in [b'x', b'X']:
            if key == b'x':
                slice_n -= 1
                if slice_n <= 3:
                    slice_n = 3
            else:
                slice_n += 1
        else:
            if key == b'C':
                up_curve_ratio += 0.1
                down_curve_ratio += 0.1

                if up_curve_ratio > 1:
                    up_curve_ratio = 1.0
                    down_curve_ratio = 1.0
            else:
                up_curve_ratio -= 0.1
                down_curve_ratio -= 0.1

                if up_curve_ratio < 0.11:
                    up_curve_ratio = 0.1
                    down_curve_ratio = 0.1

        vertex_reset(vertex_n, slice_n)
        weight_reset(up_curve_ratio, down_curve_ratio)

    elif key == b't':
        text_show = not text_show

def special(key, x, y):
    #print(key)
    pass

def timer(value):
    if not rot_pause:
        global view, view_rot_angle
        view_rot_angle += 1

    view = glm.normalize(glm.vec3(np.sin(np.radians(view_rot_angle)), 0.0, np.cos(np.radians(view_rot_angle))))
    
    glutPostRedisplay()
    glutTimerFunc(int(1 / 30 * 1000), timer, 0)

if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(150, 150)
    glutCreateWindow("LBS Tester")
    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special)
    glutTimerFunc(0, timer, 0)
    glutMainLoop()