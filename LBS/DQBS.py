from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import glm
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from PIL import Image

width = 1500
height = 800

view = glm.normalize(glm.vec3(0.0, 0.0, 1.0))
lookat = glm.vec3(0.0, 0.0, 0.0)
viewangle = 60.0
d = 30
view_rot_angle = 0
rot_pause = True

up_joint = np.array([-20, 0, 0])
down_joint = np.array([0, 0, 0])

is_mass = True
is_automated = False
is_DQBS = False
is_DQBS_automated = False

before_LBS = False

num_screenshots = 0
current_time = 0

class cube():
    def __init__(self, body, size, joint, axis, color):
        self.body = body
        self.size = size
        self.joint = joint
        self.axis = axis
        self.color = color
        self.angle = 0

    def setAngle(self, angle):
        self.angle = angle
    
    def draw(self, wireframe=True):
        drawCube(self.body, self.size, self.joint, self.angle, self.axis, self.color, wireframe=wireframe)
        glPushMatrix()
        com = self.getCOM()
        glTranslatef(com[0], com[1], com[2])
        glutSolidSphere(0.1, 10, 10)
        glPopMatrix()
    
    def getCOM(self):
        return self.joint + R.from_rotvec(np.radians(self.angle) * self.axis).apply(self.body)
    
upper_arm = cube(np.array([10, 0, 0]), np.array([20, 4, 5]), up_joint, np.array([0, 0, 1]), [1, 0, 0])
lower_arm = cube(np.array([10, 0, 0]), np.array([20, 4, 4]), down_joint, np.array([0, 0, 1]), [0, 1, 0])

weight_power = 3
class waypoint():
    def __init__(self, pos, order):
        self.pos = pos
        self.order = order

        self.origin_body = None
        self.insertion_body = None
        self.local_positions = None
        self.weights = None
        self.auto_weights = None
        
    def setProperties(self, origin_body, insertion_body):
        self.origin_body = origin_body
        self.insertion_body = insertion_body

        # origin_body_pos = origin_body.getCOM()
        # insertion_body_pos = insertion_body.getCOM()
        origin_body_pos = origin_body.joint
        insertion_body_pos = insertion_body.joint

        # print(origin_body_pos, insertion_body_pos, self.pos)

        # MASS version
        distance = []
        self.local_positions = [self.pos - origin_body_pos, self.pos - insertion_body_pos]

        distance.append(np.linalg.norm(self.pos - origin_body_pos))
        distance.append(np.linalg.norm(self.pos - insertion_body_pos))

        if distance[1] < 8:
            weight1 = 1 / np.power(distance[0], weight_power)
            weight2 = 1 / np.power(distance[1], weight_power)
            weight_sum = weight1 + weight2
            weight1 /= weight_sum
            weight2 /= weight_sum
        else:
            weight1 = 1
            weight2 = 0
        # weight1 = 1
        # weight2 = 0
        
        self.weights = [weight1, weight2]

    def setWeights(self, weights):
        self.auto_weights = weights
    
    def LBS_pos(self):
        if is_mass:
            return (self.origin_body.joint + self.local_positions[0]) * self.weights[0] + \
            (self.insertion_body.joint + R.from_rotvec(np.radians(self.insertion_body.angle) * self.insertion_body.axis).apply(self.local_positions[1])) * self.weights[1]
        elif is_DQBS:      
            b = self.weights[0] * np.array([1, 0, 0, 0]) + \
                self.weights[1] * np.concatenate([[np.cos(np.radians(self.insertion_body.angle))], np.sin(np.radians(self.insertion_body.angle)) * self.insertion_body.axis])
            b_norm = np.linalg.norm(b)
            b /= b_norm

            angle = np.arccos(b[0])
            axis = np.zeros(3)
            if angle != 0:
                denom = np.sqrt(1 - b[0] * b[0])
                axis[0] = b[1] / denom
                axis[1] = b[2] / denom
                axis[2] = b[3] / denom

            rot = R.from_rotvec(angle * axis)
            return rot.apply(self.pos)
        elif is_automated:
            return (self.origin_body.joint + self.local_positions[0]) * self.auto_weights[0] + \
            (self.insertion_body.joint + R.from_rotvec(np.radians(self.insertion_body.angle) * self.insertion_body.axis).apply(self.local_positions[1])) * self.auto_weights[1]
        elif is_DQBS_automated:
            b = self.auto_weights[0] * np.array([1, 0, 0, 0]) + \
                self.auto_weights[1] * np.concatenate([[np.cos(np.radians(self.insertion_body.angle))], np.sin(np.radians(self.insertion_body.angle)) * self.insertion_body.axis])
            b_norm = np.linalg.norm(b)
            b /= b_norm

            angle = np.arccos(b[0])
            axis = np.zeros(3)
            if angle != 0:
                denom = np.sqrt(1 - b[0] * b[0])
                axis[0] = b[1] / denom
                axis[1] = b[2] / denom
                axis[2] = b[3] / denom

            rot = R.from_rotvec(angle * axis)
            return rot.apply(self.pos)
    def non_LBS_pos(self):
        return self.origin_body.joint + self.local_positions[0], self.insertion_body.joint + R.from_rotvec(np.radians(self.insertion_body.angle) * self.insertion_body.axis).apply(self.local_positions[1])

class line_muscle():
    def __init__(self):
        self.origin = None
        self.origin_body = None
        self.insertion = None
        self.insertion_body = None
        self.waypoints = []
        self.num_waypoints = 0

        self.total_length = 0
        self.current_length = 0

    def setOrigin(self, pos, body):
        self.origin = pos
        self.origin_body = body

    def setInsertion(self, pos, body):
        self.insertion = pos
        self.insertion_body = body

    def addWaypoint(self, pos):
        self.num_waypoints += 1
        new_waypoint = waypoint(pos, self.num_waypoints)
        new_waypoint.setProperties(self.origin_body, self.insertion_body)
        self.waypoints.append(new_waypoint)

        self.total_length = 0
        poses = [self.origin] + [wp.pos for wp in self.waypoints] + [self.insertion]
        for i in range(len(poses) - 1):
            self.total_length += np.linalg.norm(poses[i] - poses[i + 1])

    def setWeights(self, a):
        # print(self.num_waypoints)
        for wp in self.waypoints:
            # print(wp.order)
            # Automated paper
            t = wp.order / (self.num_waypoints + 1)
            w = a * t * t - (a + 1) * t + 1

            print(w, 1-w)
            w = np.sqrt(1 - t * t)
            power = 1
            denom = 2.5
            a = 0.5
            w = np.power(1 - np.power(t, power), 1 / denom)

            # w = a * w + (1 - a) * (1 - t)

            # w = a * t * t * t - 3 / 2 * a * t * t + (a / 2 - 1) * t + 1
            wp.setWeights([w, 1 - w])

    def fixNearestWaypointWeights(self):
        v = 0.8
        self.waypoints[0].weights = [v, 1-v]
        self.waypoints[-1].weights = [1-v, v]
        self.waypoints[0].setWeights([v, 1-v])
        self.waypoints[-1].setWeights([1-v, v])

    def draw(self, before_LBS=False):
        if self.origin is None:
            return
        if self.insertion is None:
            return
        
        origin_pos = self.origin
        insertion_pos = R.from_rotvec(np.radians(self.insertion_body.angle) * self.insertion_body.axis).apply(self.insertion)
        poses = [origin_pos] + [wp.LBS_pos() for wp in self.waypoints] + [insertion_pos]
        glColor3f(1, 0, 0)
        glBegin(GL_LINE_STRIP)
        for pos in poses:
            glVertex3f(pos[0], pos[1], pos[2])
        glEnd()

        self.current_length = 0
        for i in range(len(poses) - 1):
            self.current_length += np.linalg.norm(poses[i] - poses[i + 1])

        for pos in poses:
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            glutSolidSphere(0.1, 5, 5)
            glPopMatrix()

        if before_LBS:
            non_LBS_pos = [wp.non_LBS_pos() for wp in self.waypoints]
            upper_pos = [pos[0] for pos in non_LBS_pos]
            lower_pos = [pos[1] for pos in non_LBS_pos]

            upper_poses = [origin_pos] + upper_pos + [insertion_pos]
            lower_poses = [origin_pos] + lower_pos + [insertion_pos]

            glColor4d(0, 0, 1, 0.2)
            glBegin(GL_LINE_STRIP)
            for pos in upper_poses:
                glVertex3f(pos[0], pos[1], pos[2])
            glEnd()

            glColor4d(0, 1, 0, 0.2)
            glBegin(GL_LINE_STRIP)
            for pos in lower_poses:
                glVertex3f(pos[0], pos[1], pos[2])
            glEnd()

muscles = []
muscle1 = line_muscle()
x_offset = -7

num_muscles = 5
num_waypoints = 10
muscle_angle = np.pi * 2
max_radius = 4
min_radius = 2
for i in range(num_muscles):
    new_muscle = line_muscle()
    if num_muscles > 1:
        rot_angle = (np.pi - muscle_angle) / 2 + muscle_angle * i / (num_muscles - 1)
        y = np.sin(rot_angle)
        z = np.cos(rot_angle)
    else:
        y = 1
        z = 0
    
    # find crossing point between line that connects origin and (y, z) and 10x10 square at origin

    origin = np.array([-8 + x_offset, min_radius * y, min_radius * z])
    insertion = np.array([8 + x_offset, min_radius * y, min_radius * z])

    new_muscle.setOrigin(origin, upper_arm)
    new_muscle.setInsertion(insertion, lower_arm)

    # new_muscle.addWaypoint(np.array([-5 + x_offset, 5 * y, 5 * z]))
    # new_muscle.addWaypoint(np.array([0 + x_offset, 6 * y, 6 * z]))
    # new_muscle.addWaypoint(np.array([5 + x_offset, 5 * y, 5 * z]))
    
    # for end in [1 / (num_waypoints * 2 + 2)]:
    #     waypoint_x = origin[0] + (insertion[0] - origin[0]) * end
    #     radius = -4 * (max_radius - min_radius) * np.power((end - 0.5), 2) + max_radius
    #     new_muscle.addWaypoint(np.array([waypoint_x, radius * y, radius * z]))

    for j in range(num_waypoints):
        waypoint_x = origin[0] + (insertion[0] - origin[0]) * (j + 1) / (num_waypoints + 1)
        radius = -4 * (max_radius - min_radius) * np.power(((j + 1) / (num_waypoints + 1) - 0.5), 2) + max_radius
        new_muscle.addWaypoint(np.array([waypoint_x, radius * y, radius * z]))

    # for end in [1 - 1 / (num_waypoints * 2 + 2)]:
    #     waypoint_x = origin[0] + (insertion[0] - origin[0]) * end
    #     radius = -4 * (max_radius - min_radius) * np.power((end - 0.5), 2) + max_radius
    #     new_muscle.addWaypoint(np.array([waypoint_x, radius * y, radius * z]))
    
    new_muscle.setWeights(0)
    # new_muscle.fixNearestWaypointWeights()
    muscles.append(new_muscle)


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

# def drawCube(midx, midy, midz, len_x, len_y, len_z, color, angle, axis, wireframe=False):
def drawCube(body, size, joint, angle, axis, color, wireframe=False):
    # glPushMatrix()
    # glTranslatef(midx, midy, midz)
    # glColor3f(0, 0, 0)
    # glutSolidSphere(0.1, 10, 10)
    # glRotatef(angle, axis[0], axis[1], axis[2])
    # glColor4d(color[0], color[1], color[2], 0.5)
    # glScalef(len_x, len_y, len_z)
    # if not wireframe:
    #     glutSolidCube(1.0)
    # glColor3f(0,0,0)
    # glutWireCube(1.0001)
    # glPopMatrix()

    glPushMatrix()
    glTranslatef(joint[0], joint[1], joint[2])
    glColor3f(0, 0, 0)
    glutSolidSphere(0.1, 10, 10)
    glRotatef(angle, axis[0], axis[1], axis[2])
    glTranslatef(body[0], body[1], body[2])
    glColor4d(color[0], color[1], color[2], 0.5)
    glScalef(size[0], size[1], size[2])
    if not wireframe:
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

    global current_time
    angle = 75 - 75 * np.cos(current_time * np.pi / 2)
    current_time += 1 / 30
    # I want angle to be 1/30 fps

    # angle = 150 * np.cos(time.time() / 2)
    axis = [0, 0, 1]

    # Drawing axis
    glPushMatrix()
    glTranslatef(up_joint[0], up_joint[1], up_joint[2])
    drawCoordinate()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(down_joint[0], down_joint[1], down_joint[2])
    glRotatef(angle, axis[0], axis[1], axis[2])
    drawCoordinate()
    glPopMatrix()
    
    # drawCube(-10, 0, 0, 20, 5, 5, [1, 0, 0], 0, [0, 0, 1], wireframe=True)
    # drawCube(10, 0, 0, 20, 5, 5, [0, 1, 0], angle, [0, 0, 1], wireframe=True)
    # drawCube([10, 0, 0], [20, 5, 5], up_joint, 0, [0, 0, 1], [1, 0, 0], wireframe=True)
    # drawCube([10, 0, 0], [20, 5, 5], down_joint, angle, axis, [0, 1, 0], wireframe=True)
    upper_arm.draw()
    lower_arm.setAngle(angle)
    lower_arm.draw()

    for muscle in muscles:
        muscle.draw()
        if before_LBS:
            muscle.draw(before_LBS= before_LBS)

    if is_mass:
        string = "MASS"
    elif is_automated:
        string = "Automated paper"
    elif is_DQBS:
        string = "DQBS"
    elif is_DQBS_automated:
        string = "DQBS Automated paper"

    text(50, height - 80, np.array([0, 0, 0]), string + " weights")

    for i, muscle in enumerate(muscles):
        text(50 + 50 * i, height - 120, np.array([0, 0, 0]), str(np.round(muscle.total_length, 2)))
        text(50 + 50 * i, height - 160, np.array([0, 0, 0]), str(np.round(muscle.current_length, 2)))
        text(50 + 50 * i, height - 200, np.array([0, 0, 0]), str(np.round(muscle.current_length / muscle.total_length, 2)))

    # Take screenshots for 5 seconds
    global num_screenshots
    if num_screenshots < 4 * 30 + 1:
        if num_screenshots == 0:
            num_screenshots += 1
        else:
            glReadBuffer(GL_FRONT)
            data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = Image.frombytes("RGB", (width, height), data)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save("screenshots/screenshot_%03d.png" % (num_screenshots))
            num_screenshots += 1

    glutSwapBuffers()

def keyboard(key, x, y):
    if key == b' ':
        global rot_pause
        rot_pause = not rot_pause
        #print('rot_pause:', rot_pause)
    if key in [b'\x1b', b'q']:
        # close the window
        glutLeaveMainLoop()
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
    elif key == b'b':
        global before_LBS
        before_LBS = not before_LBS
    elif key == b'm':
        global is_mass, is_automated, is_DQBS, is_DQBS_automated
        global num_screenshots, current_time
        if is_mass:
            is_mass = False
            is_automated = True
            is_DQBS = False
            is_DQBS_automated = False
        elif is_automated:
            is_mass = False
            is_automated = False
            is_DQBS = True
            is_DQBS_automated = False
        elif is_DQBS:
            is_mass = False
            is_automated = False
            is_DQBS = False
            is_DQBS_automated = True
        elif is_DQBS_automated:
            is_mass = True
            is_automated = False
            is_DQBS = False
            is_DQBS_automated = False
        num_screenshots = 0
        current_time = 0

def special(key, x, y):
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