#pip install imgui[glfw]
import imgui
import glfw
import numpy as np
import dartpy as dart
import viewer.gl_function as mygl
import quaternion
from PIL import Image

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from imgui.integrations.glfw import GlfwRenderer
from viewer.TrackBall import TrackBall
from learning.ray_model import loading_network
from numba import jit
from core.env import Env
from core.dartHelper import buildFromInfo

import time

## Light Option 
ambient = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
diffuse = np.array([0.6, 0.6, 0.6, 1.0], dtype=np.float32)

front_mat_shininess = np.array([60.0], dtype=np.float32)
front_mat_specular = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
front_mat_diffuse = np.array([0.5, 0.28, 0.38, 1.0], dtype=np.float32)

lmodel_ambient = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
lmodel_twoside = np.array([GL_FALSE])
light_pos = [    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 
                np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32)]

def initGL():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
    glShadeModel(GL_SMOOTH)
    glPolygonMode(GL_FRONT, GL_FILL)

    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos[0])
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient)
    glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside)

    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse)
    glLightfv(GL_LIGHT1, GL_POSITION, light_pos[1])

    glEnable(GL_LIGHT2)
    glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse)
    glLightfv(GL_LIGHT2, GL_POSITION, light_pos[2])

    glEnable(GL_LIGHTING)

    glEnable(GL_COLOR_MATERIAL)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular)
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_NORMALIZE)
    glEnable(GL_MULTISAMPLE)
    

## GLFW Initilization Function
def impl_glfw_init(window_name="Muscle Simulation", width=1920, height=1080):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

class GLFWApp():
    def __init__(self):
        super().__init__()

        ## Settin window option and initialization        
        self.name = "Muscle Simulation"
        self.width = 1920 
        self.height = 1080
        
        ## Camera Setting
        self.perspective = 45.0
        self.trackball = TrackBall()
        self.eye = np.array([0.0, 0.0, 1.0]) * np.power(1.05, 10)
        self.up = np.array([0.0, 1.0, 0.0])
        self.trans = np.array([0.0, 0.0, 0.0])
        self.zoom = 1.0

        self.trackball.set_trackball(np.array([self.width * 0.5, self.height * 0.5]), self.width * 0.5)
        self.trackball.set_quaternion(np.quaternion(1.0, 0.0, 0.0, 0.0))
        
        ## Camera transform flag
        self.mouse_down = False
        self.rotate = False
        self.translate = False

        self.mouse_x = 0
        self.mouse_y = 0
        self.motion_skel = None

        ## Flag         
        self.is_simulation = False
        self.draw_mesh = False
        self.draw_target_motion = False
        self.draw_pd_target = False
        self.draw_zero_pose = False

        self.draw_body = True
        self.draw_muscle = True
        self.draw_line_muscle = True
        self.line_width = 2
        self.draw_bone = False
        self.draw_joint = True

        self.draw_shadow = False

        # self.skel_change_realtime = True
        self.skel_change_symmetry = True

        self.reset_value = 0

        self.is_screenshot = False
        self.imagenum = 0

        self.max_checkpoint_files = []
        self.checkpoint_idx = 0
        self.result_path = './ray_results'
        self.checkpoint_update_str = ''
        self.get_max_checkpoints(self.result_path)

        imgui.create_context()
        self.window = impl_glfw_init(self.name, self.width, self.height)
        self.impl = GlfwRenderer(self.window)

        # Set Callback Function        
        ## Framebuffersize Callback Function
        def framebuffer_size_callback(window, width, height):
            self.width = width
            self.height = height
            glViewport(0, 0, width, height)
        glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)

        ## Mouse Callback Function 
        ### mouseButtonCallback
        def mouseButtonCallback(window, button, action, mods):
            # wantcapturemouse
            if not imgui.get_io().want_capture_mouse:
                self.mousePress(button, action, mods)
        glfw.set_mouse_button_callback(self.window, mouseButtonCallback)

        ### cursorPosCall back
        def cursorPosCallback(window, xpos, ypos):
            if not imgui.get_io().want_capture_mouse:
                self.mouseMove(xpos, ypos)
        glfw.set_cursor_pos_callback(self.window, cursorPosCallback)

        ### scrollCallback
        def scrollCallback(window, xoffset, yoffset):
            if not imgui.get_io().want_capture_mouse:
                self.mouseScroll(xoffset, yoffset)
        glfw.set_scroll_callback(self.window, scrollCallback)

        ## Keyboard Callback Function  
        def keyCallback(window, key, scancode, action, mods):
            if not imgui.get_io().want_capture_mouse:
                self.keyboardPress(key, scancode, action, mods)
        glfw.set_key_callback(self.window, keyCallback)

        self.env = None
        self.nn = None
        self.mus_nn = None

        ## For Graph Logging
        self.reward_buffer = []

    def get_max_checkpoints(self, path):
        self.max_checkpoint_files = []
        for item in os.listdir(path):
            main_path = os.path.join(path, item)
            if os.path.isdir(main_path):
                for subitem in os.listdir(main_path):
                    sub_path = os.path.join(main_path, subitem)
                    if os.path.isdir(sub_path):
                        if 'max_checkpoint' in os.listdir(sub_path):
                            self.max_checkpoint_files.append(os.path.join(sub_path, 'max_checkpoint'))
    
    def setEnv(self, env):  
        self.env = env
        self.motion_skel = self.env.skel.clone()
        self.reset(self.reset_value)
    
    def loadNetwork(self, path):
        self.nn, mus_nn, env_str = loading_network(path)
        # if env_str != None:
        #     self.setEnv(Env(env_str))   
        self.env.muscle_nn = mus_nn

    ## mousce button callback function
    def mousePress(self, button, action, mods):
        if action == glfw.PRESS:
            self.mouse_down = True
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.rotate = True
                self.trackball.start_ball(self.mouse_x, self.height - self.mouse_y)
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.translate = True
        elif action == glfw.RELEASE:
            self.mouse_down = False
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.rotate = False
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.translate = False

    ## mouse move callback function
    def mouseMove(self, xpos, ypos):
        dx = xpos - self.mouse_x
        dy = ypos - self.mouse_y

        self.mouse_x = xpos
        self.mouse_y = ypos

        if self.rotate:
            if dx != 0 or dy != 0:
                self.trackball.update_ball(xpos, self.height - ypos)

        if self.translate:
            rot = quaternion.as_rotation_matrix(self.trackball.curr_quat)
            self.trans += (1.0 / self.zoom) * rot.transpose() @ np.array([dx, -dy, 0.0])

    ## mouse scroll callback function
    def mouseScroll(self, xoffset, yoffset):
        if yoffset < 0:
            self.eye *= 1.05
        elif (yoffset > 0) and (np.linalg.norm(self.eye) > 0.5):
            self.eye *= 0.95
    
    def update(self):
        if self.nn is not None:
            obs = self.env.get_obs()
            action = self.nn.get_action(obs)
            _, _, done, _ = self.env.step(action)
        else:
            _, _, done, _ = self.env.step(np.zeros(self.env.num_action))
        # if done:
        #     self.is_simulation = False
        self.reward_buffer.append(self.env.get_reward())

    def drawShape(self, shape, color):
        if not shape:
            return
        
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glColor4d(color[0], color[1], color[2], color[3])
        if not self.draw_mesh:
            ## check the shape type
            if type(shape) == dart.dynamics.BoxShape:
                mygl.draw_cube(shape.getSize())
            
    def drawSkeleton(self, pos, color = np.array([0.5, 0.5, 0.5, 0.5])):
        self.motion_skel.setPositions(pos)

        for bn in self.motion_skel.getBodyNodes():
            glPushMatrix()
            glMultMatrixd(bn.getWorldTransform().matrix().transpose())
            for sn in bn.getShapeNodes():
                if not sn:
                    return
                va = sn.getVisualAspect()

                if not va or va.isHidden():
                    return
                
                glPushMatrix()
                glMultMatrixd(sn.getRelativeTransform().matrix().transpose())
                self.drawShape(sn.getShape(), color)
                
                glPopMatrix()
            glPopMatrix()
        pass
    
    def drawJoint(self, pos, color = np.array([0.0, 0.0, 0.0, 0.5])):
        self.motion_skel.setPositions(pos)

        glColor4d(color[0], color[1], color[2], color[3])
        for bn in self.motion_skel.getBodyNodes():
            glPushMatrix()
            bnWorldTransform = bn.getWorldTransform().matrix().transpose()
            glMultMatrixd(bnWorldTransform)
            
            j = bn.getParentJoint()
            
            jTransform = j.getTransformFromChildBodyNode().matrix().transpose()
            glMultMatrixd(jTransform)

            mygl.draw_sphere(0.005 * np.sqrt(2), 10, 10)
            glPopMatrix()

        
    def drawBone(self, pos, color = np.array([0.0, 0.0, 0.0, 0.5])):
        self.motion_skel.setPositions(pos)
        glColor4d(color[0], color[1], color[2], color[3])
        for bn in self.motion_skel.getBodyNodes():
            transform = bn.getWorldTransform().matrix() @ bn.getParentJoint().getTransformFromChildBodyNode().matrix()
            t_parent = transform[:3, 3]

            glPushMatrix()

            # if self.draw_joint:
            #     glPushMatrix()
            #     glTranslatef(t_parent[0], t_parent[1], t_parent[2])
            #     mygl.draw_sphere(0.01, 10, 10)
            #     glPopMatrix()
            numChild = bn.getNumChildBodyNodes()
            for i in range(numChild):
                bn_child = bn.getChildBodyNode(i)
                transform_child = bn_child.getWorldTransform().matrix() @ bn_child.getParentJoint().getTransformFromChildBodyNode().matrix()
                t_child = transform_child[:3, 3]

                glPushMatrix()
                m = (t_parent + t_child) / 2
                p2c = t_child - t_parent
                length = np.linalg.norm(p2c)
                p2c = p2c / length
                z = np.array([0, 0, 1])

                axis = np.cross(z, p2c)
                s = np.linalg.norm(axis)
                axis /= s
                c = np.dot(z, p2c)
                angle = np.rad2deg(np.arctan2(s, c))
                
                glTranslatef(m[0], m[1], m[2])
                glRotatef(angle, axis[0], axis[1], axis[2])
                mygl.draw_cube([0.01, 0.01, length])
                glPopMatrix()

            # if numChild == 0:
            #     t_child = bn.getWorldTransform().matrix()[:3, 3]
            #     glPushMatrix()
            #     m = (t_parent + t_child) / 2
            #     p2c = t_child - t_parent
            #     length = np.linalg.norm(p2c)
            #     p2c = p2c / length
            #     z = np.array([0, 0, 1])

            #     axis = np.cross(z, p2c)
            #     s = np.linalg.norm(axis)
            #     axis /= s
            #     c = np.dot(z, p2c)
            #     angle = np.rad2deg(np.arctan2(s, c))
                
            #     glTranslatef(m[0], m[1], m[2])
            #     glRotatef(angle, axis[0], axis[1], axis[2])
            #     mygl.draw_cube([0.01, 0.01, length])
            #     glPopMatrix()

            glPopMatrix()

    def drawMuscles(self, color=None):
        if color is not None:
            glColor4d(color[0], color[1], color[2], color[3])
        if self.draw_line_muscle:
            glLineWidth(self.line_width)
            idx = 0
            for m_wps in self.env.muscle_pos:
                a = self.env.muscle_activation_levels[idx]
                if color is None:
                    glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
                glBegin(GL_LINE_STRIP)
                for wp in m_wps:
                    glVertex3f(wp[0], wp[1], wp[2])
                glEnd() 
                idx += 1
        else:
            idx = 0
            for m_wps in self.env.muscle_pos:
                a = self.env.muscle_activation_levels[idx]
                if color is None:
                    glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
                for i_wp in range(len(m_wps) - 1):
                    t_parent = m_wps[i_wp]
                    t_child = m_wps[i_wp + 1]

                    glPushMatrix()
                    m = (t_parent + t_child) / 2
                    p2c = t_child - t_parent
                    length = np.linalg.norm(p2c)
                    p2c = p2c / length
                    z = np.array([0, 0, 1])

                    axis = np.cross(z, p2c)
                    s = np.linalg.norm(axis)
                    axis /= s
                    c = np.dot(z, p2c)
                    angle = np.rad2deg(np.arctan2(s, c))
                    
                    glTranslatef(m[0], m[1], m[2])
                    glRotatef(angle, axis[0], axis[1], axis[2])
                    mygl.draw_cube([0.01, 0.01, length])
                    glPopMatrix()
                idx += 1
            
    def drawSimFrame(self):
        initGL()
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glViewport(0, 0, self.width, self.height)
        gluPerspective(self.perspective, (self.width / self.height), 0.1, 100.0)
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], 0.0, 0.0, -1.0, self.up[0], self.up[1], self.up[2])
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.trackball.set_center(np.array([self.width * 0.5, self.height * 0.5]))
        self.trackball.set_radius(min(self.width, self.height) * 0.4)
        self.trackball.apply_gl_roatation()

        glScalef(self.zoom, self.zoom, self.zoom)
        glTranslatef(self.trans[0] * 0.001, self.trans[1] *0.001, self.trans[2] * 0.001)
        glEnable(GL_DEPTH_TEST)

        mygl.drawGround(-1E-3)

        if self.mouse_down:
            glLineWidth(1.5)
            mygl.draw_axis()
        
        if self.draw_target_motion:
            self.drawSkeleton(self.env.target_pos, np.array([1.0, 0.3, 0.3, 0.5]))
        if self.draw_body:
            if self.draw_bone:
                self.drawBone(self.env.skel.getPositions())
            if self.draw_joint:
                self.drawJoint(self.env.skel.getPositions())
            if self.draw_muscle:
                self.drawMuscles()
            self.drawSkeleton(self.env.skel.getPositions())
        if self.draw_pd_target:
            self.drawSkeleton(self.env.pd_target, np.array([0.3, 0.3, 1.0, 0.5]))
        if self.draw_zero_pose:
            glColor3f(0, 0, 0)
            for name, info in self.env.skel_info.items():
                if info['stretch'] != "None":
                    glPushMatrix()
                    p = info['stretch_origin']
                    glTranslatef(p[0], p[1], p[2])
                    # print(p)
                    # mygl.draw_sphere(0.005 * np.sqrt(2), 10, 10)

                    glBegin(GL_LINES)
                    a = info['stretch_axis'] * info['gap']
            
                    glVertex3f(0,0,0)
                    glVertex3f(a[0] , a[1], a[2])
                    glEnd()
                    glPopMatrix()

            if self.draw_joint:
                self.drawJoint(np.zeros(self.env.skel.getNumDofs()))
            self.drawSkeleton(np.zeros(self.env.skel.getNumDofs()), np.array([0.3, 1.0, 0.3, 0.5]))
            
            

        if self.draw_shadow:
            shadow_color = np.array([0.3, 0.3, 0.3, 1.0])
            glPushMatrix()
            glScalef(1,1E-3,1)
            glTranslatef(0, 0.000001, 0)
            if self.draw_bone:
                self.drawBone(self.env.skel.getPositions(), shadow_color)
            if self.draw_joint:
                self.drawJoint(self.env.skel.getPositions(), shadow_color)
            if self.draw_muscle:
                self.drawMuscles(shadow_color)
            
            if self.draw_target_motion:
                self.drawSkeleton(self.env.target_pos, np.array([1.0, 0.3, 0.3, 0.2]))
            if self.draw_body:
                self.drawSkeleton(self.env.skel.getPositions(), shadow_color)
            if self.draw_pd_target:
                self.drawSkeleton(self.env.pd_target, np.array([0.3, 0.3, 1.0, 0.2]))
            
            glPopMatrix()

    def retargetting(self, name, index):
        info = self.env.new_skel_info[name]

        stretches = info['stretches']

        stretch = stretches[index]
        size = info['size'][stretch]
        stretch_axis = info['stretch_axises'][index]
        gap = info['gaps'][index]

        '''
        # body_t based retargetting
        for muscle in info['muscles']:
            new_muscle_info = self.env.new_muscle_info[muscle]

            for waypoint in new_muscle_info['waypoints']:
                waypoint_ratio = waypoint['ratios'][index]
                waypoint_gap = waypoint['gaps'][index]
                if waypoint['body'] == name:
                    waypoint['p'] = info['body_t'] + stretch_axis * waypoint['ratio'] * size * 0.5 + waypoint_gap
        '''
        # starting point based retargetting
        for muscle in info['muscles']:
            new_muscle_info = self.env.new_muscle_info[muscle]

            for waypoint in new_muscle_info['waypoints']:
                waypoint_ratio = waypoint['ratios'][index]
                waypoint_gap = waypoint['gaps'][index]
                if waypoint['body'] == name:
                    waypoint['p'] = info['joint_t'] + gap + stretch_axis * waypoint_ratio * size + waypoint_gap

        if len(info['children']) > 0:
            for child in info['children']:
                parent_str = self.env.skel_info[child]['parent_str']
                
                new_parent_info = self.env.new_skel_info[parent_str]
                parent_info = self.env.skel_info[parent_str]

                parent_stretches = parent_info['stretches']
                if parent_str == name:
                    parent_stretch = parent_info['stretches'][index]
                    parent_stretch_axis = parent_info['stretch_axises'][index]
                    parent_size = new_parent_info['size'][parent_stretch]
                    parent_gap = parent_info['gaps'][index]
                else:
                    parent_stretch = parent_stretches[0]
                    parent_stretch_axis = parent_info['stretch_axises'][0]
                    parent_size = new_parent_info['size'][parent_stretch]
                    parent_gap = parent_info['gaps'][0]

                new_child_info = self.env.new_skel_info[child]
                child_info = self.env.skel_info[child]

                child_stretch = child_info['stretches'][0]
                child_stretch_axis = child_info['stretch_axises'][0]
                child_size = new_child_info['size'][child_stretch]
                child_gap = child_info['gaps'][0]
                child_gap_parent = child_info['gaps_parent'][0]

                new_child_info['joint_t'] = new_parent_info['joint_t'] + parent_gap + parent_stretch_axis * parent_size + child_gap_parent
                new_child_info['body_t'] = new_child_info['joint_t'] + child_gap + child_stretch_axis * child_size * 0.5

                if parent_str == name and len(parent_stretches) > 1:
                    for i in range(len(parent_stretches)):
                        if i != index:
                            parent_stretch_other = parent_info['stretches'][i]
                            parent_stretch_axis_other = parent_info['stretch_axises'][i]
                            parent_size_other = new_parent_info['size'][parent_stretch_other]
                            parent_gap_other = parent_info['gaps'][i]
                            new_child_info['gaps_parent'][i] = new_child_info['joint_t'] - (new_parent_info['joint_t'] + parent_gap_other + parent_stretch_axis_other * parent_size_other)

                '''
                for muscle in child_info['muscles']:
                    new_muscle_info = self.env.new_muscle_info[muscle]

                    for waypoint in new_muscle_info['waypoints']:
                        if waypoint['body'] == child:
                            waypoint['p'] = new_child_info['body_t'] + child_stretch_axis * waypoint['ratio'] * new_child_info['size'][child_stretch_index] * 0.5 + waypoint['gap']
                ''' 
                for muscle in child_info['muscles']:
                    new_muscle_info = self.env.new_muscle_info[muscle]

                    for waypoint in new_muscle_info['waypoints']:
                        waypoint_ratio = waypoint['ratios'][0]
                        waypoint_gap = waypoint['gaps'][0]
                        if waypoint['body'] == child:
                            waypoint['p'] = new_child_info['joint_t'] + child_gap + child_stretch_axis * waypoint_ratio * child_size + waypoint_gap

        foot_h = np.min([self.env.skel_info['TalusR']['body_t'][1], self.env.skel_info['TalusL']['body_t'][1]])
        new_foot_h = np.min([self.env.new_skel_info['TalusR']['body_t'][1], self.env.new_skel_info['TalusL']['body_t'][1]])

        if np.abs(new_foot_h - foot_h) > 1E-10:
            h_diff = foot_h - new_foot_h
            for name, info in self.env.new_skel_info.items():
                info['body_t'][1] += h_diff
                info['joint_t'][1] += h_diff

            for muscle_name, muscle_info in self.env.new_muscle_info.items():
                for waypoint in muscle_info['waypoints']:
                    waypoint['p'][1] += h_diff

    def newSkeleton(self):
        current_pos = self.env.skel.getPositions()
        self.env.world.removeSkeleton(self.env.skel)
        self.env.skel = buildFromInfo(self.env.new_skel_info, self.env.root_name)
        self.env.target_skel = self.env.skel.clone()
        self.env.world.addSkeleton(self.env.skel)
        # self.env.loading_muscle(self.env.muscle_path)
        self.env.loading_muscle_info(self.env.new_muscle_info)

        self.motion_skel = self.env.skel.clone()
        self.motion_skel.setPositions(current_pos)

        # self.reset(self.env.world.getTime())
        self.zero_reset()
        
    def drawUIFrame(self):
        imgui.new_frame()
        
        # imgui.show_test_window()

        imgui.set_next_window_size(400, 900, condition=imgui.ONCE)
        imgui.set_next_window_position(self.width - 410, 10, condition = imgui.ONCE)        

        # State Information 
        imgui.begin("Information")
        imgui.text("Elapsed\tTime\t:\t%.2f" % self.env.world.getTime())
        
        if imgui.tree_node("Camera Parameters"):
            imgui.text(f"Eye Position: {self.trans}")
            imgui.text(f"Eye Rotation: {quaternion.as_rotation_vector(self.trackball.curr_quat)})")
            imgui.text(f"Zoom: {self.eye}")
            imgui.tree_pop()

        if imgui.tree_node("Observation"):
            imgui.plot_histogram(
                label="##obs",
                values=self.env.get_obs().astype(np.float32),
                values_count=self.env.num_obs,
                scale_min=-10.0,
                scale_max =10.0,
                graph_size = (imgui.get_content_region_available_width(), 200)
            )
            imgui.tree_pop()
        
        if imgui.tree_node("Reward"):
            width = 60
            data_width = min(width, len(self.reward_buffer))                   
            value = np.zeros(width, dtype=np.float32)
            value[-data_width:] = np.array(self.reward_buffer[-data_width:], dtype=np.float32)
            imgui.plot_lines(
                label="##reward",
                values=value,
                values_count=width,
                scale_min=0.0,
                scale_max=1.0,
                graph_size=(imgui.get_content_region_available_width(), 200)
            )
            imgui.tree_pop()

        if imgui.tree_node("Rendering Mode"):
            # imgui.checkbox("Draw Mesh", self.draw_mesh)
            
            _, self.draw_target_motion = imgui.checkbox("Draw Target Motion", self.draw_target_motion)
            _, self.draw_pd_target = imgui.checkbox("Draw PD Target", self.draw_pd_target)
            _, self.draw_zero_pose = imgui.checkbox("Draw zero pose", self.draw_zero_pose)
            _, self.draw_body = imgui.checkbox("Draw Body", self.draw_body)
            _, self.draw_muscle = imgui.checkbox("Draw Muscle", self.draw_muscle)
            if self.draw_muscle:
                imgui.same_line()
                if imgui.radio_button("Line Muscle", self.draw_line_muscle):
                    self.draw_line_muscle = True 
                imgui.same_line()
                if imgui.radio_button("Cube Muscle", not self.draw_line_muscle):
                    self.draw_line_muscle = False

                changed, self.line_width = imgui.slider_float("Line Width", self.line_width, 0.1, 5.0)

            _, self.draw_bone = imgui.checkbox("Draw Bone", self.draw_bone)
            _, self.draw_joint = imgui.checkbox("Draw Joint", self.draw_joint)
            _, self.draw_shadow = imgui.checkbox("Draw Shadow", self.draw_shadow)

            imgui.tree_pop()
            
        changed, self.reset_value = imgui.slider_float("Reset Time", self.reset_value, 0.0, self.env.bvhs[self.env.bvh_idx].bvh_time)
        if changed:
            self.reset(self.reset_value)
        if imgui.button("Reset"):
            self.reset(self.reset_value)
        imgui.same_line()
        if imgui.button("Random Reset"):
            self.reset_value = np.random.random() * self.env.bvhs[self.env.bvh_idx].bvh_time
            self.reset(self.reset_value)
        imgui.same_line()
        if imgui.button("Zero Reset"):
            self.zero_reset()

        _, self.is_screenshot = imgui.checkbox("Take Screenshots", self.is_screenshot)
        
        if imgui.tree_node("Skel info"):
            # _, self.skel_change_realtime = imgui.checkbox("Change Skeleton in Real Time", self.skel_change_realtime)
            _, self.skel_change_symmetry = imgui.checkbox("Change Skeleton Symmetrically", self.skel_change_symmetry)
            for name, info in self.env.new_skel_info.items():
                orig_info = self.env.skel_info[name]

                stretches = info['stretches']
                for i in range(len(stretches)):
                    stretch = stretches[i]

                    if stretch == 0:
                        name_new = name + "_x"
                    elif stretch == 1:
                        name_new = name + "_y"
                    elif stretch == 2:
                        name_new = name + "_z"

                    size = info['size'][stretch]
                    orig_size = orig_info['size'][stretch]
                    
                    imgui.push_item_width(150)
                    changed, size = imgui.slider_float(name_new,
                                                                        size,
                                                                        min_value = orig_size * 0.25,
                                                                        max_value = orig_size * 4,
                                                                        format='%.3f')
                    imgui.pop_item_width()

                    if changed:
                        gap = info['gaps'][i]
                        stretch_axis = info['stretch_axises'][i]

                        info['body_t'] = info['joint_t'] + gap + stretch_axis * size * 0.5
                        
                        self.retargetting(name, i)

                        if self.skel_change_symmetry and name[-1] in ['R', 'L']:
                            if name[-1] == "R":
                                name_pair = name[:-1] + "L"
                            elif name[-1] == "L":
                                name_pair = name[:-1] + "R"

                            info_pair = self.env.new_skel_info[name_pair]
                            info_pair['size'][stretch] = info['size'][stretch].copy()

                            size_pair = info_pair['size'][stretch]
                            gap_pair = info_pair['gaps'][i]
                            stretch_axis_pair = info_pair['stretch_axises'][i]

                            info_pair['body_t'] = info_pair['joint_t'] + gap_pair + stretch_axis_pair * size_pair * 0.5

                            self.retargetting(name_pair, i)

                        self.newSkeleton()

                    imgui.same_line()
                    imgui.set_cursor_pos_x(300)
                    if imgui.button("Reset##" + name_new):
                        info['size'][stretch] = orig_info['size'][stretch].copy()

                        gap = info['gaps'][i]
                        stretch_axis = info['stretch_axises'][i]

                        info['body_t'] = info['joint_t'] + gap + stretch_axis * size * 0.5

                        self.retargetting(name, i)

                        self.newSkeleton()

            if imgui.button("Reset Skeleton"):
                for name, new_info in self.env.new_skel_info.items():
                    info = self.env.skel_info[name]
                    new_info['size'] = info['size'].copy()
                    new_info['body_t'] = info['body_t'].copy()
                    new_info['joint_t'] = info['joint_t'].copy()
                    if new_info['parent_str'] != "None":
                        new_info['gaps_parent'] = info['gaps_parent'].copy()

                for name, new_info in self.env.new_muscle_info.items():
                    info = self.env.muscle_info[name]
                    for new_waypoint, waypoint in zip(new_info['waypoints'], info['waypoints']):
                        if new_waypoint.get('p') is not None:
                            new_waypoint['p'] = waypoint['p'].copy()

                self.newSkeleton()

            imgui.tree_pop()

        if imgui.tree_node("Checkpoint"):
            if imgui.button("Update Networks"):
                self.get_max_checkpoints(self.result_path)
                self.checkpoint_update_str = f'Updated {time.strftime("%H:%M:%S", time.localtime())}'
            imgui.same_line()
            imgui.text(self.checkpoint_update_str)

            clicked, self.checkpoint_idx = imgui.listbox('', self.checkpoint_idx, self.max_checkpoint_files)

            if imgui.button("Load Network"):
                self.loadNetwork(self.max_checkpoint_files[self.checkpoint_idx])

            imgui.tree_pop()

        if imgui.tree_node("Activation Plot"):
            p0 = imgui.get_cursor_screen_pos()
            p1 = [imgui.get_content_region_available_width() + p0[0], p0[1] + 300]
            c = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) * 0.55]

            draw_list = imgui.get_window_draw_list()
            draw_list.path_clear()
            draw_list.path_rect(p0[0], p0[1], p1[0], p1[1])
            draw_list.path_fill_convex(imgui.get_color_u32_rgba(0.4, 0.4, 0.4, 1))
            draw_list.path_clear()
            draw_list.path_rect(p0[0], p0[1], p1[0], p1[1])
            draw_list.path_stroke(imgui.get_color_u32_rgba(1, 1, 1, 1), flags=imgui.DRAW_CLOSED)

            for bn in self.motion_skel.getBodyNodes():
                transform = bn.getWorldTransform().matrix() @ bn.getParentJoint().getTransformFromChildBodyNode().matrix()
                root_pos = np.array([transform[:3, 3][0], 0, 0])
                break

            idx = 0
            for m_wps in self.env.muscle_pos:
                a = self.env.muscle_activation_levels[idx]
                scale = 80
                for i_wp in range(len(m_wps) - 1):
                    s = (m_wps[i_wp] - root_pos) * scale
                    e = (m_wps[i_wp + 1] - root_pos) * scale
                    # s = m_wps[i_wp]
                    # e = m_wps[i_wp + 1]

                    s = [c[0] - s[0], c[1] - s[1]]
                    e = [c[0] - e[0], c[1] - e[1]]

                    # if s[0] < p0[0] or s[0] > p1[0] or s[1] < p0[1] or s[1] > p1[1]:
                    #     continue
                    # if e[0] < p0[0] or e[0] > p1[1] or e[1] < p0[1] or e[1] > p1[1]:
                    #     continue

                    draw_list.path_clear()
                    draw_list.path_line_to(s[0], s[1])
                    draw_list.path_line_to(e[0], e[1])
                    draw_list.path_stroke(imgui.get_color_u32_rgba(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a), flags=0, thickness=0.1)
                idx += 1
            imgui.tree_pop()

        imgui.end()
        imgui.render()

    def reset(self, reset_time=None):
        self.env.reset(reset_time)
        self.reward_buffer = [self.env.get_reward()]

    def zero_reset(self):
        self.env.zero_reset()
        self.reward_buffer = [self.env.get_reward()]

    def keyboardPress(self, key, scancode, action, mods):
        
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.window, True)
            elif key == glfw.KEY_SPACE:
                self.is_simulation = not self.is_simulation
            elif key == glfw.KEY_S:
                self.update()
            elif key == glfw.KEY_R:
                self.reset(self.reset_value)
            elif key == glfw.KEY_Z:
                self.zero_reset()
        pass

    def startLoop(self):        
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            
            self.impl.process_inputs()
            if self.is_simulation:
                self.update()
                if self.is_screenshot:
                    glReadBuffer(GL_FRONT)
                    data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
                    image = Image.frombytes("RGBA", (self.width, self.height), data)
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    image.save(f"{self.imagenum}.png") 
                    self.imagenum += 1
            
            ## Rendering Simulation
            self.drawSimFrame()
            
            self.drawUIFrame()
            
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
        self.impl.shutdown()
        glfw.terminate()
        return

