#pip install imgui[glfw]
import imgui
import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R
import dartpy as dart
import viewer.gl_function as mygl
import quaternion
from PIL import Image
from viewer.mesh_loader import MeshLoader
from sklearn.decomposition import PCA

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from imgui.integrations.glfw import GlfwRenderer
from viewer.TrackBall import TrackBall
from learning.ray_model import loading_network
from numba import jit
from core.env import Env
from core.dartHelper import buildFromInfo, exportSkeleton
from skeleton_section import SKEL_dart_info

import time

from models.smpl import SMPL
import torch

from skel.skel_model import SKEL

# from data.face_info import male_right_faces, female_right_faces
from data.skel_info import male_skin_right_faces as male_right_faces
from data.skel_info import male_skin_left_faces as male_left_faces
from data.skel_info import female_skin_right_faces as female_right_faces
from data.skel_info import female_skin_left_faces as female_left_faces
from data.skel_info import skel_joint_edges
from skel.kin_skel import skel_joints_name, pose_param_names, pose_limits

smpl_joint_names = [
    "Pelvis", 
    "L_Hip", 
    "R_Hip", 
    "Spine_01", 
    "L_Knee", 
    "R_Knee", 
    "Spine_02", 
    "L_Ankle", 
    "R_Ankle", 
    "Spine_03", 
    "L_Toe", 
    "R_Toe",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Palm",
    "R_Palm"
]
smpl_links = [
    (0, 1),
        (1, 4),
            (4, 7),
                (7, 10),    # Right Leg
    (0, 2),
        (2, 5),
            (5, 8),
                (8, 11),    # Left Leg
    (0, 3),
        (3, 6),
            (6, 9),
                (9, 12),
                    (12, 15),   # Neck
                (9, 13),
                    (13, 16),
                        (16, 18),
                            (18, 20),
                                (20, 22),   # Right Arm
                (9, 14),
                    (14, 17),
                        (17, 19),
                            (19, 21),
                                (21, 23)    # Left Arm
]

class Box:
    def __init__(self, 
                 name, 
                 pos, 
                 rot, 
                 size, 
                 color, 
                 joint, 
                 axes=None, 
                 corners=None, 
                 parent=None, 
                 isContact=None,
                 upper=None,
                 lower=None,
                 ori=None,
                 jointType=None,
                 axis=None):
        self.name = name
        self.pos = pos
        self.rot = rot
        self.rot_angle = np.rad2deg(np.linalg.norm(np.array(rot)))
        self.rot_axis = rot / self.rot_angle if self.rot_angle != 0 else np.array([0, 0, 0])
        self.size = size
        self.color = color
        self.joint = joint
        self.axes = axes
        self.corners = corners
        self.parent = parent
        self.isContact = isContact
        self.upper = upper
        self.lower = lower
        self.ori = ori
        self.jointType = jointType
        self.axis = axis

    def updateRot(self):
        self.rot_angle = np.rad2deg(np.linalg.norm(np.array(self.rot)))
        self.rot_axis = self.rot / self.rot_angle if self.rot_angle != 0 else np.array([0, 0, 0])

    def draw(self):
        glPushMatrix()
        glTranslatef(self.joint[0], self.joint[1], self.joint[2])
        glColor3f(0, 0, 0)
        mygl.draw_sphere(0.005, 10, 10)
        glPopMatrix()

        # if self.corners is not None:
        #     glColor3f(10, 0, 0)
        #     for corner in self.corners:
        #         glPushMatrix()
        #         glTranslatef(corner[0], corner[1], corner[2])
        #         mygl.draw_sphere(0.005, 10, 10)
        #         glPopMatrix()

        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])

        glPushMatrix()
        glScalef(0.1, 0.1, 0.1)
        if self.axes is not None:
            ax0 = self.axes[0]
            ax1 = self.axes[1]
            ax2 = self.axes[2]
            glBegin(GL_LINES)
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(ax0[0], ax0[1], ax0[2])
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(ax1[0], ax1[1], ax1[2])
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(ax2[0], ax2[1], ax2[2])
            glEnd()
        glPopMatrix()

        glRotatef(self.rot_angle, self.rot_axis[0], self.rot_axis[1], self.rot_axis[2])
        glColor4d(self.color[0], self.color[1], self.color[2], self.color[3])
        mygl.draw_cube(self.size)
        glPopMatrix()

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
    glBlendEquation(GL_FUNC_ADD)
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

        self.draw_obj = False
        self.obj_trans = 0.5
        self.obj_axis = {}

        self.draw_target_motion = False
        self.draw_pd_target = False
        self.draw_zero_pose = False
        self.draw_test_skeleton = False
        self.test_dofs = None

        self.draw_body = False
        self.body_trans = 0.5
        self.draw_muscle = False
        self.draw_line_muscle = True
        self.muscle_index = 0
        self.origin_v = None
        self.insertion_v = None
        self.line_width = 2
        self.draw_bone = False
        self.draw_joint = False

        self.draw_shadow = False

        # self.skel_change_realtime = True
        self.skel_change_symmetry = True
        self.skel_scale = 1
        
        self.reset_value = 0

        self.is_screenshot = False
        self.imagenum = 0

        # SMPL Model 
        self.draw_smpl = False
        self.draw_smpl_joint = False
        self.smpl_joint_index = 0
        self.draw_smpl_bone = False
        # self.draw_target_smpl = False

        self.smpl_offset = np.zeros(3)
        
        self.smpl_joints = None
        self.smpl_vertices = None
        self.smpl_colors = None

        self.smpl_zero_joints = None
        self.smpl_trans = 0.5

        self.smpl_scale = 1
        self.shape_parameters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.smpl_model = SMPL(
            "data/smpl", batch_size=2, create_transl=False, betas = torch.tensor(np.array([self.shape_parameters, self.shape_parameters], dtype=np.float32))
        )
        pos = torch.tensor(np.tile(np.zeros(3, dtype=np.float32), (2, 24, 1)))
        self.smpl_model_face_idxs = np.array(list(self.smpl_model.faces.flatten()), dtype=int)
        with torch.no_grad():
            res = self.smpl_model(betas = torch.tensor(np.array([self.shape_parameters, self.shape_parameters], dtype=np.float32)), body_pose = pos[:, 1:], global_orient=pos[:, 0].unsqueeze(1)) # , pose2rot = False)
            
            self.smpl_joints = res.smpl_joints
            self.smpl_vertices = res.vertices

            self.smpl_offset = np.array([0, -torch.min(self.smpl_vertices[0][:, 1]), 0])

            self.smpl_colors = np.ones((2, len(self.smpl_vertices[0]), 4)) * 0.8
            self.smpl_colors[:, :, 3] = self.smpl_trans

        self.smpl_vertex3 = [None, None]
        self.smpl_color4 = [None, None]
        self.smpl_normal = [None, None]

        self.max_checkpoint_files = []
        self.checkpoint_idx = 0
        self.result_path = './ray_results'
        self.checkpoint_update_str = ''
        self.get_max_checkpoints(self.result_path)

        self.skeleton_files = []
        self.skeleton_idx = 0
        self.data_path = './data'
        self.get_skeletons(self.data_path)

        self.skel_skel = None
        self.draw_dart_skel = True

        self.isDrawHalfSkin = True
        self.skin_direction = "right"
        self.draw_skel_skin = False
        self.draw_skel_skel = True
        self.draw_skel_joint = False
        self.draw_skel_joint_rot = False
        self.draw_skel_bone = False

        self.skel_vertex3 = [None, None]
        self.skel_color4 = [None, None]
        self.skel_normal = [None, None]

        self.skel_joints = None
        self.skel_joints_orig = None
        self.skel_joints_prev = None
        self.skel_joint_index = 0
        self.skel_oris = None
        self.skel_oris_prev = None
        self.skel_vertices = None
        self.skel_colors = None

        self.skel_zero_joints = None
        self.skel_opacity = 0.1

        self.skin_vertex3 = [None, None]
        self.skin_color4 = [None, None]
        self.skin_normal = [None, None]

        self.skin_vertices = None
        self.skin_colors = None

        self.skin_zero_joints = None
        self.skin_opacity = 1

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.skel_gender = "male"
        self.skel = SKEL(gender=self.skel_gender).to(self.device)

        # Set parameters to default values (T pose)
        self.skel_pose = torch.zeros(1, self.skel.num_q_params).to(self.device) # (1, 46)
        self.skel_pose_base = torch.zeros(1, self.skel.num_q_params).to(self.device) # (1, 46)
        # self.skel_pose[0][4] = -0.5
        # self.skel_pose[0][11] = -0.5
        for i in range(len(pose_param_names)):
            j = pose_param_names[i]
            if j in pose_limits.keys():
                if pose_limits[j][0] * pose_limits[j][1] < 0:
                    self.skel_pose_base[0][i] = 0
                    self.skel_pose[0][i] = 0
                else:
                    abs_vals = np.abs(np.array(pose_limits[j]))
                    base = pose_limits[j][0] if abs_vals[0] < abs_vals[1] else pose_limits[j][1]
                    self.skel_pose_base[0][i] = base
                    self.skel_pose[0][i] = base
            else:
                self.skel_pose_base[0][i] = 0
                self.skel_pose[0][i] = 0
        # # wrist_deviation_l: 45
        # self.skel_pose[0][29] = 1.247               # shoulder_r_x
        # self.skel_pose[0][30] = 0.185               # shoulder_r_y
        # self.skel_pose[0][31] = -0.416               # shoulder_r_z
        # self.skel_pose[0][33] = -3/4 * np.pi / 2    # pro_sup_r

        # self.skel_pose[0][39] = -1.385              # shoulder_l_x
        # self.skel_pose[0][40] = -0.231              # shoulder_l_y
        # self.skel_pose[0][41] = 0.139               # shoulder_l_z
        # self.skel_pose[0][43] = -np.pi / 2          # pro_sup_l
        self.skel_betas = torch.zeros(1, self.skel.num_betas).to(self.device) # (1, 10)
        self.skel_trans = torch.zeros(1, 3).to(self.device)

        # SKEL forward pass
        skel_output = self.skel(self.skel_pose, self.skel_betas, self.skel_trans)

        self.skel_joints = skel_output.joints[0]
        self.skel_joints_orig = skel_output.joints_orig[0]
        self.skel_oris = skel_output.joints_ori[0]

        from skel_f_updated import skel_f_updated_male, skel_f_updated_female
        self.skel_f_updated_male = np.array(skel_f_updated_male, dtype=int)
        self.skel_f_updated_female = np.array(skel_f_updated_female, dtype=int)
        if self.skel_gender == "male":
            self.skel_faces = self.skel_f_updated_male
        else:
            self.skel_faces = self.skel_f_updated_female

        skel_vertices = skel_output.skel_verts.detach().cpu().numpy()[0]

        self.skel_face_start_index = 94777
        self.skel_face_index =  94777

        # range(88733, 91333), (88733, "Atlas, C1")
        # list(range(89521, 91333)) + list(range(94777, 95763)), (89521, "Axis, C2")

        self.isSKELSection = False
        self.SKEL_section_unique = False
        
        self.skel_vertices = skel_vertices.copy()

        from skeleton_section import SKEL_face_index_male, SKEL_face_index_female
        self.SKEL_section_index_male = {}
        self.SKEL_section_index_female = {}
        self.skel_farthest_section_index = 0
        for i, (key_male, key_female) in enumerate(zip(SKEL_face_index_male.keys(), SKEL_face_index_female.keys())):
            self.SKEL_section_index_male[key_male] = i
            self.SKEL_section_index_female[key_female] = i

        self.SKEL_section_names_male = list(SKEL_face_index_male.keys())
        self.SKEL_section_names_female = list(SKEL_face_index_female.keys())

        self.SKEL_section_toggle_male = [True] * len(self.SKEL_section_names_male)
        self.SKEL_section_faces_male = []
        self.SKEL_unique_vertices_male = []

        for index in list(SKEL_face_index_male.values()):
            self.SKEL_section_faces_male.append(self.skel_f_updated_male[3 * index[0]: 3 * index[1]])
            self.SKEL_unique_vertices_male.append(np.unique(self.SKEL_section_faces_male[-1]))

        self.SKEL_section_toggle_female = [True] * len(self.SKEL_section_names_female)
        self.SKEL_section_faces_female = []        
        self.SKEL_unique_vertices_female = []
    
        for index in list(SKEL_face_index_female.values()):
            self.SKEL_section_faces_female.append(self.skel_f_updated_female[3 * index[0]: 3 * index[1]])
            self.SKEL_unique_vertices_female.append(np.unique(self.SKEL_section_faces_female[-1]))

        self.SKEL_section_colors = []
        self.SKEL_section_vertex3 = []
        self.SKEL_section_color4 = []
        self.SKEL_section_normal = []
        self.SKEL_section_midpoints = []

        self.gui_boxes = []
        self.cand_gui_box = None
        self.test_skel = None
        self.test_skel_dofs = None
        self.draw_skel_dofs = True
        self.draw_gui_boxes = False

        self.selected_face = None
        self.selected_point = None
        self.selected_mesh_index = None

        SKEL_section_vertices = self.skel_vertices
        SKEL_section_faces = self.SKEL_section_faces_male if self.skel_gender == "male" else self.SKEL_section_faces_female
        min_vertex = np.min(SKEL_section_vertices, axis=0)
        max_vertex = np.max(SKEL_section_vertices, axis=0)
        for faces in SKEL_section_faces:
            vertex_normals = np.zeros_like(SKEL_section_vertices)
            vertex_counts = np.zeros((SKEL_section_vertices.shape[0], 1))
            SKEL_section_vertex3 = SKEL_section_vertices[faces]

            # SKEL_section_midpoint = np.mean(SKEL_section_vertex3, axis=0)
            unique_vertex = SKEL_section_vertices[np.unique(faces)]
            SKEL_section_midpoint = np.mean(unique_vertex, axis=0)
            self.SKEL_section_midpoints.append(SKEL_section_midpoint)

            SKEL_section_colors = np.ones((len(SKEL_section_vertices), 4))
            SKEL_section_colors[:, :3] = (np.mean(SKEL_section_vertex3, axis=0) - min_vertex) / (max_vertex - min_vertex)
            # SKEL_section_colors[:, :3] = np.random.rand(3)
            SKEL_section_colors[:, 3] = self.skel_opacity
            self.SKEL_section_colors.append(SKEL_section_colors)

            SKEL_section_color4 = SKEL_section_colors[faces]
            self.SKEL_section_vertex3.append(SKEL_section_vertex3)
            self.SKEL_section_color4.append(SKEL_section_color4)

            SKEL_section_normal = np.cross(SKEL_section_vertex3[1::3] - SKEL_section_vertex3[0::3], SKEL_section_vertex3[2::3] - SKEL_section_vertex3[0::3])
            SKEL_section_normal = SKEL_section_normal / np.linalg.norm(SKEL_section_normal, axis=1)[:, np.newaxis]
            SKEL_section_normal = np.repeat(SKEL_section_normal, 3, axis=0)
            np.add.at(vertex_normals, faces, SKEL_section_normal)
            np.add.at(vertex_counts, faces, 1)
            vertex_counts[vertex_counts == 0] = 1
            vertex_normals /= vertex_counts
            self.SKEL_section_normal.append(vertex_normals[faces])

        
        # Thorax Realignment 
        L1 = self.SKEL_section_midpoints[68]
        C7 = self.SKEL_section_midpoints[133]
        a = (C7[0] - L1[0]) / (C7[1] - L1[1])
        b = L1[0] - a * L1[1]

        for i, thor in enumerate([80, 82, 83, 81, 79, 76, 75, 72, 74, 77, 73, 78]):
            fixed = a * self.SKEL_section_midpoints[thor][1] + b
            offset = fixed - self.SKEL_section_midpoints[thor][0]
            offset *= 0.5 / (5.5**2) * (i - 5.5) ** 2 + 0.5

            self.SKEL_section_vertex3[thor][:, 0] += offset
            self.SKEL_section_midpoints[thor][0] += offset
        
        # L2_C7 = [70, 68, 80, 82, 83, 81, 79, 76, 75, 72, 74, 77, 73, 78]
        L2_C7 = [70, 68, 78, 73, 77, 74, 72, 75, 76, 79, 81, 83, 82, 80, 133]
        for index, i in enumerate(range(len(L2_C7) - 3)):
            inf = self.SKEL_section_midpoints[L2_C7[i]]
            sup = self.SKEL_section_midpoints[L2_C7[i + 1]]
            a = (sup[2] - inf[2]) / (sup[1] - inf[1])
            b = inf[2] - a * inf[1]

            fix_vert = L2_C7[i + 2]
            fixed = a * self.SKEL_section_midpoints[fix_vert][1] + b
            offset = (fixed - self.SKEL_section_midpoints[fix_vert][2]) * (12 - i) / 12
            
            self.SKEL_section_vertex3[fix_vert][:, 2] += offset
            self.SKEL_section_midpoints[fix_vert][2] += offset

            up_vert = L2_C7[i + 3]
            y_offset = (sup[1] + self.SKEL_section_midpoints[up_vert][1]) / 2 - self.SKEL_section_midpoints[fix_vert][1]
            self.SKEL_section_vertex3[fix_vert][:, 1] += y_offset
            self.SKEL_section_midpoints[fix_vert][1] += y_offset

        self.skel_colors = np.ones((2, len(self.skel_vertices), 4))
        # self.skel_colors[:, :, :3] = (self.skel_vertices - min_vertex) / (max_vertex - min_vertex)
        self.skel_colors[:, :, 3] = self.skel_opacity

        vertex_normals = np.zeros_like(skel_vertices)
        vertex_counts = np.zeros((skel_vertices.shape[0], 1))
        self.skel_vertex3[0] = skel_vertices[self.skel_faces]
        self.skel_color4[0] = self.skel_colors[0][self.skel_faces]

        ## Compute normal vector according to vertex3
        self.skel_normal[0] = np.cross(self.skel_vertex3[0][1::3] - self.skel_vertex3[0][0::3], self.skel_vertex3[0][2::3] - self.skel_vertex3[0][0::3])
        self.skel_normal[0] = self.skel_normal[0] / np.linalg.norm(self.skel_normal[0], axis=1)[:, np.newaxis]
        self.skel_normal[0] = np.repeat(self.skel_normal[0], 3, axis=0)

        np.add.at(vertex_normals, self.skel_faces, self.skel_normal[0])
        np.add.at(vertex_counts, self.skel_faces, 1)
        vertex_counts[vertex_counts == 0] = 1
        vertex_normals /= vertex_counts
        self.skel_normal[0] = vertex_normals[self.skel_faces]
        
        # f = open(f'skel_f_{self.skel_gender}.py', 'w')
        # f.write(f'skel_f = {self.skel_faces.tolist()}\n')
        # f.close()

        # duplicate_vertices = np.where(vertex_counts == 1)[0]    
        # vertex_groups = []
        # while len(duplicate_vertices) > 0:
        #     group = [duplicate_vertices[0]]
        #     for i in range(1, len(duplicate_vertices)):
        #         if np.linalg.norm(skel_vertices[duplicate_vertices[0]] - skel_vertices[duplicate_vertices[i]]) < 0.0001:
        #             group.append(duplicate_vertices[i])
        #     vertex_groups.append(group)
        #     duplicate_vertices = duplicate_vertices[~np.isin(duplicate_vertices, group)]
        #     print(len(duplicate_vertices))

        # f = open(f'duplicate_vertices_{self.skel_gender}.py', 'w')
        # f.write('duplicate_vertices = [\n')
        # for group in vertex_groups:
        #     f.write(f"    {group},\n")
        # f.write(']\n')
        # f.close()

        self.skin_faces = np.array(list(self.skel.skin_f.cpu().flatten()), dtype=int)
        skin_vertices = skel_output.skin_verts.detach().cpu().numpy()[0]

        self.skin_vertices = skin_vertices.copy()

        if self.isDrawHalfSkin:
            if self.skel_gender == 'male':
                if self.skin_direction == "right":
                    self.skin_half_faces = male_right_faces
                else:
                    self.skin_half_faces = male_left_faces
            else:
                if self.skin_direction == "right":
                    self.skin_half_faces = female_right_faces
                else:
                    self.skin_half_faces = female_left_faces

            # # Used for checking half skin faces
            # self.skin_right_faces = []
            # self.cand_skin_right_faces = []
            # self.skin_left_faces = []
            # if self.skel_gender == 'male':
            #     threshold = 0.003
            # else:
            #     threshold = 0
            # for i in range(len(self.skin_faces) // 3):
            #     indices = self.skin_faces[i * 3: (i + 1) * 3]
            #     vertices = self.skin_vertices[indices]
            #     num_minus_x = np.sum(vertices[:, 0] < threshold)
            #     if num_minus_x == 3:
            #         self.skin_right_faces.extend(indices)
            #     elif num_minus_x == 1 or num_minus_x == 2:
            #         self.cand_skin_right_faces.extend(indices)
            #     else:
            #         self.skin_left_faces.extend(indices)
            # print(len(self.skin_right_faces))
            # print(len(self.skin_left_faces))
            # self.skin_faces = np.array(self.skin_right_faces, dtype=int)

            self.skin_faces = np.array(self.skin_half_faces, dtype=int)

        self.add_skin_index = 0

        self.skin_colors = np.ones((2, len(self.skin_vertices), 4)) * 0.8
        self.skin_colors[:, :, 3] = self.skin_opacity

        vertex_normals = np.zeros_like(skin_vertices)
        vertex_counts = np.zeros((skin_vertices.shape[0], 1))
        self.skin_vertex3[0] = skin_vertices[self.skin_faces]
        self.skin_color4[0] = self.skin_colors[0][self.skin_faces]

        ## Compute normal vector according to vertex3
        self.skin_normal[0] = np.cross(self.skin_vertex3[0][1::3] - self.skin_vertex3[0][0::3], self.skin_vertex3[0][2::3] - self.skin_vertex3[0][0::3])
        self.skin_normal[0] = self.skin_normal[0] / np.linalg.norm(self.skin_normal[0], axis=1)[:, np.newaxis]
        self.skin_normal[0] = np.repeat(self.skin_normal[0], 3, axis=0)
        np.add.at(vertex_normals, self.skin_faces, self.skin_normal[0])
        np.add.at(vertex_counts, self.skin_faces, 1)
        vertex_counts[vertex_counts == 0] = 1
        vertex_normals /= vertex_counts
        self.skin_normal[0] = vertex_normals[self.skin_faces]
    
        self.meshes = {}

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

    def get_skeletons(self, path):
        self.skeleton_files = []
        for item in os.listdir(path):
            if item.endswith('.xml') and 'skel' in item:
                self.skeleton_files.append(os.path.join(path, item))

    def update_skel(self, gender_changed=False):
        self.skel_joints_prev = self.skel_joints
        self.skel_oris_prev = self.skel_oris

        if gender_changed:
            self.skel = SKEL(gender=self.skel_gender).to(self.device)

            if self.skel_gender == "female":
                self.skel_faces = np.array(self.skel_f_updated_female, dtype=int)
            else:
                self.skel_faces = np.array(self.skel_f_updated_male, dtype=int)

        # SKEL forward pass
        skel_output = self.skel(self.skel_pose, self.skel_betas, self.skel_trans)

        self.skel_joints = skel_output.joints[0]
        self.skel_joints_orig = skel_output.joints_orig[0]
        self.skel_oris = skel_output.joints_ori[0]
        pelvis = self.env.skel.getBodyNode("Pelvis").getCOM() - self.skel_joints[0].cpu().numpy()
        # pelvis = np.array([0, 0, 0])

        skel_vertices = skel_output.skel_verts.detach().cpu().numpy()[0] + pelvis
        self.skel_vertices = skel_vertices.copy()

        # self.skel_vertices = skel_vertices.copy()
        SKEL_section_vertices = self.skel_vertices
        SKEL_section_faces = self.SKEL_section_faces_male if self.skel_gender == "male" else self.SKEL_section_faces_female
        # min_vertex = np.min(SKEL_section_vertices, axis=0)
        # max_vertex = np.max(SKEL_section_vertices, axis=0)

        for i, faces in enumerate(SKEL_section_faces):
            self.SKEL_section_colors[i][:, 3] = self.skel_opacity

            vertex_normals = np.zeros_like(SKEL_section_vertices)
            vertex_counts = np.zeros((SKEL_section_vertices.shape[0], 1))
            SKEL_section_vertex3 = SKEL_section_vertices[faces]
            SKEL_section_color4 = self.SKEL_section_colors[i][faces]
            self.SKEL_section_vertex3[i] = SKEL_section_vertex3
            self.SKEL_section_color4[i] = SKEL_section_color4

            unique_vertex = SKEL_section_vertices[np.unique(faces)]
            SKEL_section_midpoint = np.mean(unique_vertex, axis=0)
            self.SKEL_section_midpoints[i] = SKEL_section_midpoint

            SKEL_section_normal = np.cross(SKEL_section_vertex3[1::3] - SKEL_section_vertex3[0::3], SKEL_section_vertex3[2::3] - SKEL_section_vertex3[0::3])
            SKEL_section_normal = SKEL_section_normal / np.linalg.norm(SKEL_section_normal, axis=1)[:, np.newaxis]
            SKEL_section_normal = np.repeat(SKEL_section_normal, 3, axis=0)
            np.add.at(vertex_normals, faces, SKEL_section_normal)
            np.add.at(vertex_counts, faces, 1)
            vertex_counts[vertex_counts == 0] = 1
            vertex_normals /= vertex_counts
            self.SKEL_section_normal[i] = vertex_normals[faces]

        # Thorax Realignment 
        SKEL_section_index = self.SKEL_section_index_male if self.skel_gender == "male" else self.SKEL_section_index_female
        L1 = self.SKEL_section_midpoints[SKEL_section_index["L1"]]
        C7 = self.SKEL_section_midpoints[SKEL_section_index["C7"]]
        a = (C7[0] - L1[0]) / (C7[1] - L1[1])
        b = L1[0] - a * L1[1]

        # T1-12
        # for i, thor in enumerate([80, 82, 83, 81, 79, 76, 75, 72, 74, 77, 73, 78]):
        for i in range(12):
            thor = SKEL_section_index["T" + str(i + 1)]
            rib_l = SKEL_section_index["Left rib " + str(i + 1)]
            rib_r = SKEL_section_index["Right rib " + str(i + 1)]

            sects = [thor, rib_l, rib_r]
            
            if i < 7:
                rib_l_cart = SKEL_section_index["Left rib " + str(i + 1) + " cartilage"]
                rib_r_cart = SKEL_section_index["Right rib " + str(i + 1) + " cartilage"]
                sects.append(rib_l_cart)
                sects.append(rib_r_cart)
                if i == 0:
                    manu = SKEL_section_index["Manubrium"]
                    xiph = SKEL_section_index["Xiphoid Process"]
                    sects.append(manu)
                    sects.append(xiph)
                elif i == 1:
                    sternum = SKEL_section_index["Sternum Body"]
                    sects.append(sternum)
            elif i == 7:
                rib_l_cart = SKEL_section_index["Left 8910 costal cartilage"]
                rib_r_cart = SKEL_section_index["Right 8910 costal cartilage"]
                sects.append(rib_l_cart)
                sects.append(rib_r_cart)
            
            fixed = a * self.SKEL_section_midpoints[thor][1] + b
            offset = fixed - self.SKEL_section_midpoints[thor][0]
            offset *= 0.5 / (5.5**2) * (i - 5.5) ** 2 + 0.5
            
            for sect in sects:
                self.SKEL_section_vertex3[sect][:, 0] += offset
                self.SKEL_section_midpoints[sect][0] += offset

        # # Linearly decrease offset from T12 to T1; doesn't apply well to thorax near cervix
        # offset = a * self.SKEL_section_midpoints[78][1] + b - self.SKEL_section_midpoints[78][0]
        # for i, thor in enumerate([80, 82, 83, 81, 79, 76, 75, 72, 74, 77, 73, 78]):
        #     self.SKEL_section_vertex3[thor][:, 0] += offset * (i + 2) / 13
        #     self.SKEL_section_midpoints[thor][0] += offset * (i + 2) / 13

        ## This code makes thorax fit too much to cervix-lumbar line
        # for i in [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]:
        #     fixed = a * self.SKEL_section_midpoints[i][1] + b
        #     offset = fixed - self.SKEL_section_midpoints[i][0]
        #     self.SKEL_section_vertex3[i][:, 0] += offset
        #     self.SKEL_section_midpoints[i][0] += offset
        
        L2_C7 = []
        L2_C7.append(SKEL_section_index["L2"])
        L2_C7.append(SKEL_section_index["L1"])
        for i in range(12):
            L2_C7.append(SKEL_section_index["T" + str(12 - i)])
        L2_C7.append(SKEL_section_index["C7"])
        # L2_C7.append(SKEL_section_index["C6"])
        
        # L2_C7 = [70, 68, 78, 73, 77, 74, 72, 75, 76, 79, 81, 83, 82, 80, 133]
        for i in range(len(L2_C7) - 3):
            inf = self.SKEL_section_midpoints[L2_C7[i]]
            sup = self.SKEL_section_midpoints[L2_C7[i + 1]]
            a = (sup[2] - inf[2]) / (sup[1] - inf[1])
            b = inf[2] - a * inf[1]

            fix_vert = L2_C7[i + 2]
            rib_l = SKEL_section_index["Left rib " + str(12 - i)]
            rib_r = SKEL_section_index["Right rib " + str(12 - i)]

            sects = [fix_vert, rib_l, rib_r]
            
            if i > 4:
                rib_l_cart = SKEL_section_index["Left rib " + str(12 - i) + " cartilage"]
                rib_r_cart = SKEL_section_index["Right rib " + str(12 - i) + " cartilage"]
                sects.append(rib_l_cart)
                sects.append(rib_r_cart)
                if i == 11:
                    manu = SKEL_section_index["Manubrium"]
                    xiph = SKEL_section_index["Xiphoid Process"]
                    sects.append(manu)
                    sects.append(xiph)
                elif i == 10:
                    sternum = SKEL_section_index["Sternum Body"]
                    sects.append(sternum)
            elif i == 4:
                rib_l_cart = SKEL_section_index["Left 8910 costal cartilage"]
                rib_r_cart = SKEL_section_index["Right 8910 costal cartilage"]
                sects.append(rib_l_cart)
                sects.append(rib_r_cart)

            fixed = a * self.SKEL_section_midpoints[fix_vert][1] + b
            offset = (fixed - self.SKEL_section_midpoints[fix_vert][2]) * (12 - i) / 12
            
            up_vert = L2_C7[i + 3]
            y_offset = (sup[1] + self.SKEL_section_midpoints[up_vert][1]) / 2 - self.SKEL_section_midpoints[fix_vert][1]

            for sect in sects:
                self.SKEL_section_vertex3[sect][:, 2] += offset
                self.SKEL_section_midpoints[sect][2] += offset

                self.SKEL_section_vertex3[sect][:, 1] += y_offset
                self.SKEL_section_midpoints[sect][1] += y_offset
        
        def find_major_axes(vertices):
            # Step 1: Center the vertices by subtracting the mean
            centered_vertices = vertices - np.mean(vertices, axis=0)
    
            # Step 2: Perform PCA
            pca = PCA(n_components=3)
            pca.fit(centered_vertices)
            
            # Step 3: Get the principal components
            major_axes = pca.components_  # Each row is a major axis (PC1, PC2, PC3)
            explained_variance = pca.explained_variance_ratio_  # Proportion of variance along each axis
            
            return major_axes, explained_variance

        if len(self.gui_boxes) == 0:
            pass
            
            # # First Trial
            # # Simply find Bounding box aligned to xyz axes
            # for i, vertex3 in enumerate(self.SKEL_section_vertex3):
            #     min = np.min(vertex3, axis=0)
            #     max = np.max(vertex3, axis=0)
            #     mean = (min + max) / 2
            #     box_size = max - min
                
            #     if np.max(box_size) > 4 * np.min(box_size):
            #         print(self.SKEL_section_names_male[i], box_size, np.max(box_size) / np.min(box_size))

            #     self.gui_boxes.append(Box(i, mean, [0, 0, 0], box_size, [0.5, 0.5, 0.5, 0.3], mean))

            # # Second Trial
            # # Find bounding box for each major group
            # for group in [[0, 1],   # Skull
            #               [2],[3],[4],[5],[6],[7],[8],  # Cervix
            #               [9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],  # Thorax
            #               [21],[22],[23],[24],[25],     # Lumbar
            #               [26, 27, 71, 72, 73],     # Saccrococcygeal
            #               [28, 29, 30],  # Sternum
            #               range(31, 51),    # Left Ribs
            #               range(51, 71),    # Right Ribs,
            #               [74],[75, 76],    # Left Lower Limb
            #               [77],             # Talus
            #               range(78, 89),    # Calcaneus, Metacarpal
            #               range(89, 105),   # Phalanges
            #               [105],[106,107],  # Left Lower Limb
            #               [108],            # Talus
            #               range(109, 120),  # Calcaneus, Metacarpal
            #               range(120, 136),  # Phalanges
            #               [136],[137],[138],[139],  # Left Upper Limb
            #               range(140, 153),   # Left Carpals
            #               [153],[154],[155],[156],[157],[158],[159],[160],[161],[162],[163],[164],[165],[166],    # Left Fingers
            #               [167],[168],[169],[170],  # Right Upper Limb
            #               range(171, 184),   # Right Carpals
            #               [184],[185],[186],[187],[188],[189],[190],[191],[192],[193],[194],[195],[196],[197],    # Right Fingers
            #               ]:
            #     min_all = None
            #     max_all = None
            #     for elem in group:
            #         min = np.min(self.SKEL_section_vertex3[elem], axis=0)
            #         max = np.max(self.SKEL_section_vertex3[elem], axis=0)
            #         if min_all is None:
            #             min_all = min
            #             max_all = max
            #         else:
            #             min_all = np.minimum(min_all, min)
            #             max_all = np.maximum(max_all, max)
            #     mean = (min_all + max_all) / 2
            #     box_size = max_all - min_all
            #     self.gui_boxes.append(Box(i, mean, [0, 0, 0], box_size, [0.5, 0.5, 0.5, 0.3], mean))


            # # Third Trial
            # # Find bounding boxes aligned to PCA major axes of each section
            # for i, section_vertices in enumerate(self.SKEL_unique_vertices_male):
            #     vertices = self.skel_vertices[section_vertices]
            #     name = self.SKEL_section_names_male[i]
            #     major_axes, explained_variance = find_major_axes(vertices)

            #     if np.dot(np.cross(major_axes[0], major_axes[1]), major_axes[2]) < 0:
            #         major_axes = -major_axes

            #     if explained_variance[0] > 0.75 and not "rib" in name.lower() and not "cartilage" in name.lower():
            #         rot_mat = np.array(major_axes).T
            #         transformed_vertices = vertices @ rot_mat

            #         min = np.min(transformed_vertices, axis=0)
            #         max = np.max(transformed_vertices, axis=0)
            #         box_size = max - min
            #         mean = (min + max) / 2

            #         corners = np.array([
            #             [min[0], min[1], min[2]],
            #             [min[0], min[1], max[2]],
            #             [min[0], max[1], min[2]],
            #             [min[0], max[1], max[2]],
            #             [max[0], min[1], min[2]],
            #             [max[0], min[1], max[2]],
            #             [max[0], max[1], min[2]],
            #             [max[0], max[1], max[2]],
            #         ])
            #         corners = corners @ rot_mat.T
            #         mean = mean @ rot_mat.T

            #         rot_vec = R.from_matrix(rot_mat).as_rotvec()
            #         self.gui_boxes.append(Box(i, mean, rot_vec, box_size, [1, 0.5, 0.5, 0.3], mean, major_axes, corners))
            #     else:
            #         min = np.min(vertices, axis=0)
            #         max = np.max(vertices, axis=0)
            #         box_size = max - min
            #         mean = (min + max) / 2
            #         self.gui_boxes.append(Box(i, mean, [0, 0, 0], box_size, [0.5, 0.5, 1, 0.3], mean, major_axes))

            # Fourth Trial
            # Find PCA aligned bounding box for each group
            # Determine Parent, Joint, PCA align

            # ([Group elements], Name, Parent, Joint, PCA align, Contact)
            for info in SKEL_dart_info:
                vertices_all = []
                group = info[0]
                for elem in group:
                    # vertices = self.skel_vertices[self.SKEL_unique_vertices_male[elem]]
                    vertices = self.SKEL_section_vertex3[elem]
                    vertices_all.extend(list(vertices))

                name = info[1]
                parent = info[2]
                joint = info[3]
                isPCA = info[4]      
                isContact = info[5]
                upper = info[6]
                lower = info[7]
                jointType = info[8]
                
                # isPCA based partwise bounding box
                if isPCA:
                    major_axes, _ = find_major_axes(vertices_all)
                    if np.dot(np.cross(major_axes[0], major_axes[1]), major_axes[2]) < 0:
                        major_axes = -major_axes
                
                    rot_mat = np.array(major_axes).T
                    transformed_vertices = vertices_all @ rot_mat

                    min = np.min(transformed_vertices, axis=0)
                    max = np.max(transformed_vertices, axis=0)
                    box_size = max - min
                    mean = (min + max) / 2
                    mean = mean @ rot_mat.T

                    rot_vec = R.from_matrix(rot_mat).as_rotvec()
                else:
                    if len(vertices_all) == 0:
                        if name[-1] == "R":
                            SC = np.mean(self.skel_vertices[[101422, 100837]], axis=0)
                            AC = self.skel_vertices[180303]
                        else:
                            SC = np.mean(self.skel_vertices[[100621, 100807]], axis=0)
                            AC = self.skel_vertices[224083]
                        SC2AC = AC - SC
                        length = np.linalg.norm(SC2AC)
                        box_size = [0.02, 0.02, length]

                        SC2AC = SC2AC / length
                        z = np.array([0, 0, 1])

                        axis = np.cross(z, SC2AC)
                        s = np.linalg.norm(axis)
                        axis /= s
                        c = np.dot(z, SC2AC)
                        angle = np.arctan2(s, c)

                        mean = (SC + AC) / 2
                        rot_vec = angle * axis
                        major_axes = None
                    else:
                        min = np.min(vertices_all, axis=0)
                        max = np.max(vertices_all, axis=0)
                        box_size = max - min
                        mean = (min + max) / 2

                        rot_vec = [0, 0, 0]
                        major_axes = None
                color = [0.5, 0.5, 1, 0.3]

                # # Always no PCA
                # min = np.min(vertices_all, axis=0)
                # max = np.max(vertices_all, axis=0)
                # box_size = max - min
                # mean = (min + max) / 2
                # rot_vec = [0, 0, 0]
                # color = [0.5, 0.5, 0.5, 0.3]
                # major_axes = None

                # # Always PCA
                # major_axes, _ = find_major_axes(vertices_all)
                # if np.dot(np.cross(major_axes[0], major_axes[1]), major_axes[2]) < 0:
                #     major_axes = -major_axes
            
                # rot_mat = np.array(major_axes).T
                # transformed_vertices = vertices_all @ rot_mat

                # min = np.min(transformed_vertices, axis=0)
                # max = np.max(transformed_vertices, axis=0)
                # box_size = max - min
                # mean = (min + max) / 2
                # mean = mean @ rot_mat.T
                # rot_vec = R.from_matrix(rot_mat).as_rotvec()
                # color = [1, 0.5, 0.5, 0.3]

                if joint is not None:
                    if joint[0] == "Joint":
                        pelvis = self.env.skel.getBodyNode("Pelvis").getCOM() - self.skel_joints[0].cpu().numpy()
                        joint_index = joint[1]
                        joint_pos = self.skel_joints[joint_index].cpu() + pelvis

                        ori = self.skel_oris[joint[1]].cpu().numpy()
                    elif joint[0] == "Mid":
                        section_indices = joint[1]
                        vertices_joint_all = []
                        for elem in section_indices:
                            # vertices = self.skel_vertices[self.SKEL_unique_vertices_male[elem]]
                            vertices_joint = self.SKEL_section_vertex3[elem]
                            vertices_joint_all.extend(list(vertices_joint))
                        joint_pos = np.mean(vertices_joint_all, axis=0)
                        ori = np.eye(3)
                    elif joint[0] == "Face":
                        start, section_name = joint[2]

                        indices = np.array(joint[1]) -start
                        vertices_joint_all = []
                        for index in indices:
                            section_i = self.SKEL_section_index_male[section_name]
                            vertices_joint = self.SKEL_section_vertex3[section_i][index*3:index*3+3]
                            vertices_joint_all.extend(vertices_joint)
                        joint_pos = np.mean(vertices_joint_all, axis=0)
                        ori = np.eye(3)
                else:
                    joint_pos = mean
                    ori = None

                # 30 Knee R
                # 43 Knee L
                # 62 Elbow R
                # 63 Sup R
                # 91 Elbow L
                # 92 Sup L
                if jointType == "Revolute":
                    rotvec = R.from_matrix(ori).as_rotvec()
                    angle = np.linalg.norm(rotvec)
                    axis = ori / angle if angle != 0 else np.array([0, 0, 1])
                    if "Tibia" in name:
                        axis = axis[0]
                    elif "Radius" in name:  # 91 Elbow L, 92 Sup L
                        axis = axis[0]
                    else:
                        axis = axis[1]
                else:
                    axis = None

                self.gui_boxes.append(Box(name, 
                                          mean, 
                                          rot_vec, 
                                          box_size, 
                                          color, 
                                          joint_pos, 
                                          axes=major_axes, 
                                          parent=parent, 
                                          isContact=isContact,
                                          upper=upper,
                                          lower=lower,
                                          ori=ori,
                                          jointType=jointType,
                                          axis=axis
                                          ))

        self.skel_colors = np.ones((2, len(self.skel_vertices), 4)) * 0.8
        # self.skel_colors[:, :, :3] = (self.skel_vertices - min_vertex) / (max_vertex - min_vertex)
        self.skel_colors[:, :, 3] = self.skel_opacity

        vertex_normals = np.zeros_like(skel_vertices)
        vertex_counts = np.zeros((skel_vertices.shape[0], 1))
        self.skel_vertex3[0] = skel_vertices[self.skel_faces]
        self.skel_color4[0] = self.skel_colors[0][self.skel_faces]
        ## Compute normal vector according to vertex3
        self.skel_normal[0] = np.cross(self.skel_vertex3[0][1::3] - self.skel_vertex3[0][0::3], self.skel_vertex3[0][2::3] - self.skel_vertex3[0][0::3])
        self.skel_normal[0] = self.skel_normal[0] / np.linalg.norm(self.skel_normal[0], axis=1)[:, np.newaxis]
        self.skel_normal[0] = np.repeat(self.skel_normal[0], 3, axis=0)
        np.add.at(vertex_normals, self.skel_faces, self.skel_normal[0])
        np.add.at(vertex_counts, self.skel_faces, 1)
        vertex_counts[vertex_counts == 0] = 1
        vertex_normals /= vertex_counts
        self.skel_normal[0] = vertex_normals[self.skel_faces]

        self.skin_faces = np.array(list(self.skel.skin_f.cpu().flatten()), dtype=int)

        if self.isDrawHalfSkin:
            # self.skin_faces = np.array(self.skin_right_faces, dtype=int)

            if self.skel_gender == 'male':
                if self.skin_direction == "right":
                    self.skin_half_faces = male_right_faces
                else:
                    self.skin_half_faces = male_left_faces
            else:
                if self.skin_direction == "right":
                    self.skin_half_faces = female_right_faces
                else:
                    self.skin_half_faces = female_left_faces
            
            self.skin_faces = np.array(self.skin_half_faces, dtype=int)

        skin_vertices = skel_output.skin_verts.detach().cpu().numpy()[0] + pelvis

        # self.skin_vertices = skin_vertices.copy()

        self.skin_colors = np.ones((2, len(self.skin_vertices), 4)) * 0.8
        self.skin_colors[:, :, 3] = self.skin_opacity

        vertex_normals = np.zeros_like(skin_vertices)
        vertex_counts = np.zeros((skin_vertices.shape[0], 1))
        self.skin_vertex3[0] = skin_vertices[self.skin_faces]
        self.skin_color4[0] = self.skin_colors[0][self.skin_faces]
        ## Compute normal vector according to vertex3
        self.skin_normal[0] = np.cross(self.skin_vertex3[0][1::3] - self.skin_vertex3[0][0::3], self.skin_vertex3[0][2::3] - self.skin_vertex3[0][0::3])
        self.skin_normal[0] = self.skin_normal[0] / np.linalg.norm(self.skin_normal[0], axis=1)[:, np.newaxis]
        self.skin_normal[0] = np.repeat(self.skin_normal[0], 3, axis=0)
        np.add.at(vertex_normals, self.skin_faces, self.skin_normal[0])
        np.add.at(vertex_counts, self.skin_faces, 1)
        vertex_counts[vertex_counts == 0] = 1
        vertex_normals /= vertex_counts
        self.skin_normal[0] = vertex_normals[self.skin_faces]

        # if self.skel_joints_prev is not None:
        #     ori_gap = self.skel_oris - self.skel_oris_prev
        #     ori_gap_sum = torch.sum(ori_gap, axis=(1,2))

        #     non_zero_indices = torch.nonzero(ori_gap_sum, as_tuple=True)[0]

        #     if len(non_zero_indices) > 0:
        #         j = non_zero_indices[0]
        #         self.skel_joint_index = j

        #         rot = self.skel_oris_prev[j] * self.skel_oris[j].inverse()
        #         rot = rot.cpu().numpy()

        #         pos_gap = (self.skel_joints[j] - self.skel_joints_prev[j]).cpu().numpy()
        #         if np.linalg.norm(pos_gap) == 0:
        #             orig = self.skel_joints[j]
        #             print('same pos')
        #         else:
        #             orig = np.linalg.inv(rot - np.eye(3)) @ (rot @ self.skel_joints_prev[j].cpu().numpy() - self.skel_joints[j].cpu().numpy())
        #             print('diff pos')
        #         print(j, orig, R.from_matrix(rot).as_rotvec())
        #         print()

    def update_smpl(self):
        ## Update SMPL Vertex to character's vertex
        ## Pos 0 : Sim Character, Pos 1 : Ref Character
        pos = torch.tensor(np.tile(np.zeros(3, dtype=np.float32), (2, 24, 1)))

        # with torch.no_grad():
        #     res = self.smpl_model(body_pose = pos[:, 1:], global_orient=pos[:, 0].unsqueeze(1), betas = torch.tensor(np.array([self.shape_parameters,self.shape_parameters], dtype=np.float32)))
        #     self.smpl_zero_joints = res.smpl_joints[0] * self.smpl_scale

        #     self.smpl_colors = np.ones((2, len(self.smpl_vertices[0]), 4)) * 0.8
        #     self.smpl_colors[:, :, 3] = self.smpl_trans

        for jn_idx in range(self.env.skel.getNumJoints()):
            skel_joint = self.env.skel.getJoint(jn_idx)
            jn_name = skel_joint.getName()
            if jn_name in self.env.smpl_jn_idx:
                smpl_idx = self.env.smpl_jn_idx[jn_name]
                target_skel_joint = self.env.target_skel.getJoint(jn_idx)
                
                if skel_joint.getNumDofs() == 1:
                    skel_rot = skel_joint.getAxis() * skel_joint.getPosition(0)
                    target_skel_rot = target_skel_joint.getAxis() * target_skel_joint.getPosition(0)
                elif skel_joint.getNumDofs() == 3:
                    skel_rot = skel_joint.getPositions()
                    target_skel_rot = target_skel_joint.getPositions()
                elif skel_joint.getNumDofs() == 6:
                    skel_rot = skel_joint.getPositions()[:3]
                    target_skel_rot = target_skel_joint.getPositions()[:3]

                # fix_mat = np.eye(3)
                # if jn_name in ["FemurL", "FemurR", "ArmL", "ArmR", "TibiaL", "TibiaR", "ForeArmL", "ForeArmR"]:
                #     child_jn_name = self.env.skel_info[jn_name]["children"][0]
                #     child_smpl_idx = self.env.smpl_jn_idx[child_jn_name]

                #     skel_dir = self.env.new_skel_info[child_jn_name]['joint_t'] - self.env.new_skel_info[jn_name]['joint_t']
                #     skel_dir = skel_dir / np.linalg.norm(skel_dir)
                #     smpl_dir = (self.smpl_zero_joints[child_smpl_idx] - self.smpl_zero_joints[smpl_idx]).numpy()
                #     smpl_dir = smpl_dir / np.linalg.norm(smpl_dir)

                #     k = np.cross(smpl_dir, skel_dir)
                #     sin_theta = np.linalg.norm(k)
                #     cos_theta = np.dot(smpl_dir, skel_dir)

                #     skel_rot = k
                #     if sin_theta != 0:
                #         k = k / sin_theta

                #     K = np.array([[    0, -k[2],  k[1]],
                #                   [ k[2],     0, -k[0]],
                #                   [-k[1],  k[0],     0]
                #                 ])
                    
                #     fix_mat = fix_mat + K * sin_theta + np.dot(K, K) * (1 - cos_theta)

                #     skel_axis = skel_rot.copy()
                #     skel_angle = np.linalg.norm(skel_rot)
                #     if skel_angle != 0:
                #         skel_axis = skel_axis / skel_angle
                #     skel_cos_theta = np.cos(skel_angle)
                #     skel_sin_theta = np.sin(skel_angle)
                #     skel_one_minus_cos = 1 - skel_cos_theta

                #     R = np.array([
                #                     [skel_cos_theta + skel_axis[0]**2 * skel_one_minus_cos,
                #                     skel_axis[0] * skel_axis[1] * skel_one_minus_cos - skel_axis[2] * skel_sin_theta,
                #                     skel_axis[0] * skel_axis[2] * skel_one_minus_cos + skel_axis[1] * skel_sin_theta],

                #                     [skel_axis[1] * skel_axis[0] * skel_one_minus_cos + skel_axis[2] * skel_sin_theta,
                #                     skel_cos_theta + skel_axis[1]**2 * skel_one_minus_cos,
                #                     skel_axis[1] * skel_axis[2] * skel_one_minus_cos - skel_axis[0] * skel_sin_theta],

                #                     [skel_axis[2] * skel_axis[0] * skel_one_minus_cos - skel_axis[1] * skel_sin_theta,
                #                     skel_axis[2] * skel_axis[1] * skel_one_minus_cos + skel_axis[0] * skel_sin_theta,
                #                     skel_cos_theta + skel_axis[2]**2 * skel_one_minus_cos]
                #                 ])
                    
                #     skel_rot_matrix = R @ fix_mat

                #     angle = np.arccos((np.trace(skel_rot_matrix) - 1) / 2)
                #     if np.isclose(angle, 0):
                #         skel_rot = np.zeros(3)
                #     elif np.isclose(angle, np.pi):
                #         # Special case: angle is 180 degrees
                #         eigvals, eigvecs = np.linalg.eig(skel_rot_matrix)
                #         axis = eigvecs[:, np.isclose(eigvals, 1)].flatten().real
                #         skel_rot =  (axis / np.linalg.norm(axis)) * angle
                #     else:
                #         axis = np.array([
                #             skel_rot_matrix[2, 1] - skel_rot_matrix[1, 2],
                #             skel_rot_matrix[0, 2] - skel_rot_matrix[2, 0],
                #             skel_rot_matrix[1, 0] - skel_rot_matrix[0, 1]
                #         ]) / (2 * np.sin(angle))
                #         skel_rot = axis * angle
                    
                pos[0, smpl_idx] = torch.tensor(skel_rot)
                pos[1, smpl_idx] = torch.tensor(target_skel_rot)

        
        with torch.no_grad():
            res = self.smpl_model(body_pose = pos[:, 1:], global_orient=pos[:, 0].unsqueeze(1), betas = torch.tensor(np.array([self.shape_parameters,self.shape_parameters], dtype=np.float32)))
            # root_dif = self.env.skel.getBodyNode("Pelvis").getCOM() - res.smpl_joints[0][0].numpy()
            
            # root_dif = self.env.skel.getBodyNode("Pelvis").getCOM() + np.array([0, 0.25, 0])
            root_dif = self.env.skel.getBodyNode("Pelvis").getCOM() - res.smpl_joints[0][0].numpy() * self.smpl_scale
            self.smpl_vertices[0] = res.vertices[0] * self.smpl_scale + root_dif
            # self.smpl_vertices[1] = res.vertices[1] * smpl_scale + root_dif
            self.smpl_joints = res.smpl_joints[0] * self.smpl_scale + root_dif

            # self.smpl_vertices[0] = res.vertices[0] * smpl_scale + self.smpl_offset
            # self.smpl_joints = res.smpl_joints[0] * smpl_scale + self.smpl_offset
            
        # ## If use muscles, update colors 
        # if self.env.muscles:
        #     # # NotImplemented
        #     # vert_degree = (self.color_mapping @ torch.tensor(self.env.muscle_activation_levels, device="cuda", dtype=torch.float32)).cpu().numpy()            
        #     vert_degree = 0         
        #     # ## example
        #     self.smpl_colors[0, :, 1] = (1.0 - vert_degree)
        #     self.smpl_colors[0, :, 2] = (1.0 - vert_degree)

        ## Sim Character
        vertex_normals = np.zeros_like(self.smpl_vertices[0])
        vertex_counts = np.zeros((self.smpl_vertices[0].shape[0], 1))
        self.smpl_vertex3[0] = self.smpl_vertices[0][self.smpl_model_face_idxs].numpy()
        self.smpl_color4[0] = self.smpl_colors[0][self.smpl_model_face_idxs]
        ## Compute normal vector according to vertex3
        self.smpl_normal[0] = np.cross(self.smpl_vertex3[0][1::3] - self.smpl_vertex3[0][0::3], self.smpl_vertex3[0][2::3] - self.smpl_vertex3[0][0::3])
        self.smpl_normal[0] = self.smpl_normal[0] / np.linalg.norm(self.smpl_normal[0], axis=1)[:, np.newaxis]
        self.smpl_normal[0] = np.repeat(self.smpl_normal[0], 3, axis=0)
        np.add.at(vertex_normals, self.smpl_model_face_idxs, self.smpl_normal[0])
        np.add.at(vertex_counts, self.smpl_model_face_idxs, 1)
        vertex_counts[vertex_counts == 0] = 1
        vertex_normals /= vertex_counts
        self.smpl_normal[0] = vertex_normals[self.smpl_model_face_idxs]

    def fitSkel2SMPL(self):
        pass
        # First scale the skeleton using shoulder to pelvis length
        skel_shoulder_height = max(self.env.skel_info["ShoulderR"]["joint_t"][1], self.env.skel_info["ShoulderL"]["joint_t"][1])
        skel_pelvis_height  = self.env.skel_info["Pelvis"]["joint_t"][1]
        skel_length = skel_shoulder_height - skel_pelvis_height

        # SMPL_neck_height = self.smpl_joints[12][1].numpy()
        SMPL_shoulder_height = max(self.smpl_joints[16][1].numpy(), self.smpl_joints[17][1].numpy())
        SMPL_pelvis_height = self.smpl_joints[0][1].numpy()
        SMPL_length = SMPL_shoulder_height - SMPL_pelvis_height

        self.skel_scale = SMPL_length / skel_length

        # print(self.skel_scale)
        self.scaleSkeleton(self.skel_scale)
        self.newSkeleton()

        root_dif = self.env.skel.getBodyNode("Pelvis").getCOM() - self.smpl_joints[0].numpy() * self.smpl_scale
        self.smpl_joints += root_dif
        self.smpl_vertices[0] += root_dif

        # for match in [("Spine", "Pelvis", 3, 0),
        #               ("Torso", "Spine", 6, 3), 
        #               ("Neck", "Torso", 12, 6), 
        #               ("Head", "Neck", 15, 12)]:
        #     skel_c = match[0]
        #     skel_p = match[1]
        #     SMPL_c = match[2]
        #     SMPL_p = match[3]

        #     # move_vec = np.array([0, 1, 0]) * np.dot(np.array([0, 1, 0]), (self.smpl_joints[SMPL_c] - self.smpl_joints[SMPL_p]).numpy())
        #     # print(move_vec)
        #     move_vec = self.smpl_joints[SMPL_c].numpy() - self.env.new_skel_info[skel_c]["joint_t"]
        #     self.env.new_skel_info[skel_c]["joint_t"] = self.smpl_joints[SMPL_c].numpy().copy()
        #     # self.env.new_skel_info[skel_c]["body_t"] += move_vec

        # for match in [("ShoulderR", 14), ("ShoulderL", 13)]:
        #     skel_joint = match[0]
        #     SMPL_joint = match[1]

        #     move_vec = np.array([0, 0, self.smpl_joints[SMPL_joint][2].numpy() - self.env.new_skel_info[skel_joint]["joint_t"][2]])
        #     new_info = self.env.new_skel_info[skel_joint]
        #     info = self.env.skel_info[skel_joint]
        #     new_info["joint_t"] += move_vec
        #     new_info["body_t"] += move_vec

        #     stretch = new_info["stretches"][0]
        #     stretch_axis = new_info["stretch_axises"][0]

        #     for i in range(len(info['gaps'])):
        #         info['gaps'][i] = info['body_t'] - (info['joint_t'] + stretch_axis * info['size'][stretch] * 0.5)
            
        #     parent_info = self.env.new_skel_info[new_info['parent_str']]
        #     parent_stretches = parent_info['stretches']
        #     parent_joint_t = parent_info['joint_t']

        #     for i in range(len(info['gaps_parent'])):
        #         parent_stretch = parent_stretches[i]
        #         parent_size = parent_info['size'][parent_stretch]

        #         parent_stretch_axis = parent_info['stretch_axises'][i]
        #         parent_gap = parent_info['gaps'][i]

        #         gap_parent_cand1 = new_info['joint_t'] - (parent_joint_t + parent_gap + parent_stretch_axis * parent_size)
        #         gap_parent_cand2 = new_info['joint_t'] - (parent_joint_t + parent_gap)

        #         if np.linalg.norm(gap_parent_cand1) < np.linalg.norm(gap_parent_cand2):
        #             gap_parent = gap_parent_cand1
        #         else:
        #             gap_parent = gap_parent_cand2

        #         new_info['gaps_parent'][i][0] = gap_parent

        #     self.retargetting(skel_joint, 0)

        # for match in [("FemurR", 2), ("FemurL", 1), 
        #               ("ArmR", 17), ("ArmL", 16)]:
        #     skel_joint = match[0]
        #     SMPL_joint = match[1]

        #     new_info = self.env.new_skel_info[skel_joint]
        #     info = self.env.skel_info[skel_joint]
        #     move_vec = self.smpl_joints[SMPL_joint].numpy() - new_info["joint_t"]   
        #     new_info['joint_t'] = self.smpl_joints[SMPL_joint].numpy().copy()
        #     new_info["body_t"] += move_vec

        #     for muscle in new_info['muscles']:
        #         new_muscle_info = self.env.new_muscle_info[muscle]

        #         for waypoint in new_muscle_info['waypoints']:
        #             if waypoint['body'] == skel_joint:
        #                 waypoint['p'] += move_vec

        #     parent_info = self.env.new_skel_info[new_info['parent_str']]
        #     parent_stretches = parent_info['stretches']
        #     parent_joint_t = parent_info['joint_t']

        #     for i in range(len(info['gaps_parent'])):
        #         parent_stretch = parent_stretches[i]
        #         parent_size = parent_info['size'][parent_stretch]

        #         parent_stretch_axis = parent_info['stretch_axises'][i]
        #         parent_gap = parent_info['gaps'][i]

        #         gap_parent_cand1 = new_info['joint_t'] - (parent_joint_t + parent_gap + parent_stretch_axis * parent_size)
        #         gap_parent_cand2 = new_info['joint_t'] - (parent_joint_t + parent_gap)

        #         if np.linalg.norm(gap_parent_cand1) < np.linalg.norm(gap_parent_cand2):
        #             gap_parent = gap_parent_cand1
        #             same_direction_to_parent = True
        #         else:
        #             gap_parent = gap_parent_cand2
        #             same_direction_to_parent = False

        #         new_info['gaps_parent'][i][0] = gap_parent

            # if len(info['children']) > 0:
            #     for child in info['children']:
            #         new_child_info = self.env.new_skel_info[child]
            #         child_info = self.env.skel_info[child]

            #         new_child_info['joint_t'] = child_info['joint_t'] + move_vec
            #         new_child_info['body_t'] = child_info['body_t'] + move_vec

            #         for muscle in child_info['muscles']:
            #             new_muscle_info = self.env.new_muscle_info[muscle]
            #             muscle_info = self.env.muscle_info[muscle]
            #             for waypoint, waypoint_orig in zip(new_muscle_info['waypoints'], muscle_info["waypoints"]):
            #                 waypoint['p'] = waypoint_orig['p'] + move_vec

            # foot_h = np.min([self.env.skel_info['TalusR']['body_t'][1], self.env.skel_info['TalusL']['body_t'][1]])
            # new_foot_h = np.min([self.env.new_skel_info['TalusR']['body_t'][1], self.env.new_skel_info['TalusL']['body_t'][1]])

            # if np.abs(new_foot_h - foot_h) > 1E-10:
            #     h_diff = foot_h - new_foot_h
            #     for name, info in self.env.new_skel_info.items():
            #         info['body_t'][1] += h_diff
            #         info['joint_t'][1] += h_diff

            #     for muscle_name, muscle_info in self.env.new_muscle_info.items():
            #         for waypoint in muscle_info['waypoints']:
            #             waypoint['p'][1] += h_diff
        
        for match in [("Pelvis", "FemurR", "TibiaR", 0, 5), ("Pelvis", "FemurL", "TibiaL", 0, 4)]:
            pass
            skel_p = match[0]
            skel_f = match[1]
            skel_t = match[2]

            SMPL_p = match[3]
            SMPL_t = match[4]

            # || joint_f + gap + stretch_axis * new_size + gap_parent - joint_p|| = desired_length
            desired_length = np.linalg.norm((self.smpl_joints[SMPL_t] - self.smpl_joints[SMPL_p]).numpy())

            info = self.env.new_skel_info[skel_f]
            stretch = info['stretches'][0]
            v = info["stretch_axises"][0]
            a = info["joint_t"] + info["gaps"][0] + self.env.skel_info[skel_t]["gaps_parent"][0][0] - self.env.new_skel_info[skel_p]["joint_t"]
            va = np.dot(v, a)
            sq = np.sqrt(va ** 2 + desired_length ** 2 - np.linalg.norm(a) ** 2)
            size_cand1 = -va + sq
            size_cand2 = -va - sq

            print(skel_f, size_cand1, size_cand2)

            info["size"][stretch] = size_cand1 if size_cand1 > 0 else size_cand2

            gap = info['gaps'][0]
            stretch_axis = info['stretch_axises'][0]

            info['body_t'] = info['joint_t'] + gap + stretch_axis * info['size'][stretch] * 0.5

            self.retargetting(skel_p, 0)

        for match in [
                    # ("FemurR", "TibiaR", 2, 5), ("FemurL", "TibiaL", 1, 4), 
                      ("TibiaR", "TalusR", 5, 8), ("TibiaL", "TalusL", 4, 7),
                      ("ArmR", "ForeArmR", 17, 19), ("ArmL", "ForeArmL", 16, 18),
                      ("ForeArmR", "HandR", 19, 21), ("ForeArmL", "HandL", 18, 20)
                      ]:
            skel_p = match[0]
            skel_c = match[1]
            SMPL_p = match[2]
            SMPL_c = match[3]

            # || gap + stretch_axis * new_size + gap_parent || = desired_length
            desired_length = np.linalg.norm((self.smpl_joints[SMPL_p] - self.smpl_joints[SMPL_c]).numpy())

            info = self.env.new_skel_info[skel_p]
            stretch = info['stretches'][0]
            v = info["stretch_axises"][0]
            a = info["gaps"][0] + self.env.skel_info[skel_c]["gaps_parent"][0][0]
            va = np.dot(v, a)
            sq = np.sqrt(va ** 2 + desired_length ** 2 - np.linalg.norm(a) ** 2)
            size_cand1 = -va + sq
            size_cand2 = -va - sq
        
            info["size"][stretch] = size_cand1 if size_cand1 > 0 else size_cand2
            gap = info['gaps'][0]
            stretch_axis = info['stretch_axises'][0]

            info['body_t'] = info['joint_t'] + gap + stretch_axis * info['size'][stretch] * 0.5

            self.retargetting(skel_p, 0)

        for match in [("HandR", 21, 23), ("HandL", 20, 22)]:
            skel_p = match[0]
            SMPL_p = match[1]
            SMPL_c = match[2]
            desired_length = np.linalg.norm((self.smpl_joints[SMPL_p] - self.smpl_joints[SMPL_c]).numpy())

            info = self.env.new_skel_info[skel_p]
            stretch = info['stretches'][0]
            stretch_axis = info['stretch_axises'][0]
            info['size'][stretch] = desired_length * 1.8
            info['body_t'] = info['joint_t'] + info['gaps'][0] + stretch_axis * info['size'][stretch] * 0.5

            self.retargetting(skel_p, 0)

    def setObjScale(self):
        for bn in self.env.skel.getBodyNodes():
            self.meshes[bn.getName()].new_vertices_4 = self.meshes[bn.getName()].vertices_4.copy()

            new_info = self.env.new_skel_info[bn.getName()]
            info = self.env.skel_info[bn.getName()]
            stretch = info['stretches'][0]
            stretch_scale = new_info['size'][stretch] / info['size'][stretch] 
            self.meshes[bn.getName()].new_vertices_4[:, stretch] *= stretch_scale

            transform = bn.getWorldTransform().matrix() @ bn.getParentJoint().getTransformFromChildBodyNode().matrix()
            t_parent = transform[:3, 3]
            
            numChild = bn.getNumChildBodyNodes()
            if numChild > 1:
                info = self.env.skel_info[bn.getName()]
                stretch_axis = info['stretch_axises'][0]

                max_dot = 0
                for i in range(numChild):
                    bn_child = bn.getChildBodyNode(i)
                    transform_child = bn_child.getWorldTransform().matrix() @ bn_child.getParentJoint().getTransformFromChildBodyNode().matrix()
                    t_child = transform_child[:3, 3]

                    cand_p2c = t_child - t_parent
                    length = np.linalg.norm(cand_p2c)
                    cand_p2c = cand_p2c / length

                    cand_max_dot = np.abs(np.dot(stretch_axis, cand_p2c))
                    if cand_max_dot > max_dot:
                        max_dot = cand_max_dot
                        p2c = cand_p2c

                if max_dot < 0.99:
                    p2c = self.env.skel_info[bn.getName()]['stretch_axises'][0]

            elif numChild == 1:
                bn_child = bn.getChildBodyNode(0)
                transform_child = bn_child.getWorldTransform().matrix() @ bn_child.getParentJoint().getTransformFromChildBodyNode().matrix()
                t_child = transform_child[:3, 3]

                p2c = t_child - t_parent
                length = np.linalg.norm(p2c)
                p2c = p2c / length

            else:
                p2c = self.env.skel_info[bn.getName()]['stretch_axises'][0]

            # self.meshes[bn.getName()].axis = p2c
            axis_joint = p2c

            info = self.env.skel_info[bn.getName()]
            stretch = info['stretches'][0]
            if stretch == 0:
                axis_stretch = np.array([1, 0, 0])
            elif stretch == 1:
                axis_stretch = np.array([0, 1, 0])
            else:
                axis_stretch = np.array([0, 0, 1])

            if np.dot(axis_stretch, info['stretch_axises'][0]) < 0:
                axis_stretch = -axis_stretch

            axis = np.cross(axis_stretch, axis_joint)

            rot = R.from_rotvec(axis)
            rotmat = R.as_matrix(rot)
            self.meshes[bn.getName()].new_vertices_4 = np.dot(self.meshes[bn.getName()].new_vertices_4, rotmat.T)

    def setEnv(self, env):  
        self.env = env
        self.motion_skel = self.env.skel.clone()
        self.test_dofs = np.zeros(self.motion_skel.getNumDofs())

        for k, info in self.env.mesh_info.items():
            self.meshes[k] = MeshLoader()
            self.meshes[k].load(info)

        # self.reset(self.reset_value)
        self.zero_reset()

        for bn in self.env.skel.getBodyNodes():
            transform = bn.getWorldTransform().matrix() @ bn.getParentJoint().getTransformFromChildBodyNode().matrix()
            t_parent = transform[:3, 3]
            
            self.meshes[bn.getName()].vertices_4 -= t_parent

            numChild = bn.getNumChildBodyNodes()
            if numChild > 1:
                info = self.env.skel_info[bn.getName()]
                stretch_axis = info['stretch_axises'][0]

                max_dot = 0
                for i in range(numChild):
                    bn_child = bn.getChildBodyNode(i)
                    transform_child = bn_child.getWorldTransform().matrix() @ bn_child.getParentJoint().getTransformFromChildBodyNode().matrix()
                    t_child = transform_child[:3, 3]

                    cand_p2c = t_child - t_parent
                    length = np.linalg.norm(cand_p2c)
                    cand_p2c = cand_p2c / length

                    cand_max_dot = np.abs(np.dot(stretch_axis, cand_p2c))
                    if cand_max_dot > max_dot:
                        max_dot = cand_max_dot
                        p2c = cand_p2c

                if max_dot < 0.99:
                    p2c = self.env.skel_info[bn.getName()]['stretch_axises'][0]

            elif numChild == 1:
                bn_child = bn.getChildBodyNode(0)
                transform_child = bn_child.getWorldTransform().matrix() @ bn_child.getParentJoint().getTransformFromChildBodyNode().matrix()
                t_child = transform_child[:3, 3]

                p2c = t_child - t_parent
                length = np.linalg.norm(p2c)
                p2c = p2c / length

            else:
                p2c = self.env.skel_info[bn.getName()]['stretch_axises'][0]

            # self.meshes[bn.getName()].axis = p2c
            axis_joint = p2c

            info = self.env.skel_info[bn.getName()]
            stretch = info['stretches'][0]
            if stretch == 0:
                axis_stretch = np.array([1, 0, 0])
            elif stretch == 1:
                axis_stretch = np.array([0, 1, 0])
            else:
                axis_stretch = np.array([0, 0, 1])

            if np.dot(axis_stretch, info['stretch_axises'][0]) < 0:
                axis_stretch = -axis_stretch

            axis = np.cross(axis_joint, axis_stretch)

            rot = R.from_rotvec(axis)
            rotmat = R.as_matrix(rot)
            self.meshes[bn.getName()].vertices_4 = np.dot(self.meshes[bn.getName()].vertices_4, rotmat.T)

        # self.update_smpl()

        # self.fitSkel2SMPL()
        self.setObjScale()
        self.newSkeleton()
    
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

            if button == glfw.MOUSE_BUTTON_MIDDLE:
                xpos, ypos = glfw.get_cursor_pos(self.window)
                selected_face, selected_point, selected_mesh_index = self.pick_face(xpos, ypos, self.width, self.height)
                if selected_face is not None:
                    self.selected_face = selected_face
                    self.selected_point = selected_point
                    self.selected_mesh_index = selected_mesh_index
                print(self.selected_face, self.selected_point, self.SKEL_section_names_male[self.selected_mesh_index])

        elif action == glfw.RELEASE:
            self.mouse_down = False
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.rotate = False
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.translate = False

    def get_ray_from_cursor(self, cursor_x, cursor_y, screen_width, screen_height):
        projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        x = cursor_x
        y = screen_height - cursor_y  # Convert GLFW coordinates to OpenGL
        near = gluUnProject(x, y, 0.0, modelview_matrix, projection_matrix, viewport)
        far = gluUnProject(x, y, 1.0, modelview_matrix, projection_matrix, viewport)

        ray_origin = np.array(near)
        ray_direction = np.array(far) - ray_origin
        ray_direction /= np.linalg.norm(ray_direction)
        return ray_origin, ray_direction

    def ray_intersects_triangles(self, ray_origin, ray_direction, v0, v1, v2):
        # Expand ray_direction to match the shape of v0, v1, and v2
        ray_direction = np.expand_dims(ray_direction, axis=0)  # Shape (1, 3)
        
        # Prepare edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Calculate determinants
        h = np.cross(ray_direction, edge2)  # Cross product with ray direction, shape (num_faces, 3)
        a = np.einsum('ij,ij->i', edge1, h)  # Dot product with shape (num_faces,)

        # Filter out triangles that are parallel to the ray (|a| close to zero)
        epsilon = 1e-7
        mask = np.abs(a) > epsilon
        if not np.any(mask):
            return None, None  # No intersection

        # Calculate parameters u and v for triangles that are not parallel
        f = 1.0 / a[mask]
        s = ray_origin - v0[mask]
        u = f * np.einsum('ij,ij->i', s, h[mask])
        valid_u = (u >= 0) & (u <= 1)

        # Continue with valid triangles only
        q = np.cross(s[valid_u], edge1[mask][valid_u])
        v = f[valid_u] * np.einsum('ij,ij->i', ray_direction.repeat(len(q), axis=0), q)
        valid_v = (v >= 0) & (u[valid_u] + v <= 1)

        # Calculate intersection points for the remaining triangles
        t = f[valid_u][valid_v] * np.einsum('ij,ij->i', edge2[mask][valid_u][valid_v], q[valid_v])
        valid_intersections = t > epsilon  # Ignore intersections behind the ray origin

        if np.any(valid_intersections):
            # Compute intersection points and return the closest one
            intersection_points = ray_origin + np.outer(t[valid_intersections], ray_direction.squeeze())
            distances = np.linalg.norm(intersection_points - ray_origin, axis=1)
            closest_index = np.argmin(distances)
            closest_point = intersection_points[closest_index]
            closest_face_index = np.where(mask)[0][valid_u][valid_v][closest_index]
            
            return closest_face_index, closest_point

        return None, None  # No intersection
    
    def pick_face(self, cursor_x, cursor_y, screen_width, screen_height):
        ray_origin, ray_direction = self.get_ray_from_cursor(cursor_x, cursor_y, screen_width, screen_height)

        closest_face = None
        closest_point = None
        closest_mesh_index = None
        closest_distance = float('inf')
        
        SKEL_section_toggle = self.SKEL_section_toggle_male if self.skel_gender == "male" else self.SKEL_section_toggle_female
        for i, isOn in enumerate(SKEL_section_toggle):
            if isOn:
                vertex3 = self.SKEL_section_vertex3[i]

                v0 = vertex3[0::3]
                v1 = vertex3[1::3]
                v2 = vertex3[2::3]

                face, point = self.ray_intersects_triangles(ray_origin, ray_direction, v0, v1, v2)
                if point is not None:
                    distance = np.linalg.norm(point - ray_origin)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_face = face
                        closest_point = point
                        closest_mesh_index = i

        return closest_face, closest_point, closest_mesh_index

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
            try:
                _, _, done, _ = self.env.step(action)
            except:
                _, _, done, _ = self.env.step(np.zeros(self.env.num_action))
        else:
            _, _, done, _ = self.env.step(np.zeros(self.env.num_action))
        # if done:
        #     self.is_simulation = False
        self.reward_buffer.append(self.env.get_reward())

        self.update_smpl()

    def drawShape(self, shape, color):
        if not shape:
            return
        
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glColor4d(color[0], color[1], color[2], color[3])
        # if not self.draw_mesh:
        #     ## check the shape type
        #     if type(shape) == dart.dynamics.BoxShape:
        #         mygl.draw_cube(shape.getSize())

        mygl.draw_cube(shape.getSize())
    
    def drawObj(self, pos, color = np.array([0.5, 0.5, 0.5, 0.5])):
        self.motion_skel.setPositions(pos)
        
        glPushMatrix()

        self.motion_skel.setPositions(pos)
        for bn in self.motion_skel.getBodyNodes():
            transform = bn.getWorldTransform().matrix() @ bn.getParentJoint().getTransformFromChildBodyNode().matrix()
            t_parent = transform[:3, 3]
            r_parent = transform[:3, :3]

            glPushMatrix()

            glTranslatef(t_parent[0], t_parent[1], t_parent[2])

            rot = R.from_matrix(r_parent)
            rotvec = rot.as_rotvec()
            angle = np.linalg.norm(rotvec)
            axis = rotvec / angle if angle != 0 else np.array([0, 0, 1])
            glRotatef(np.rad2deg(angle), axis[0], axis[1], axis[2])
            
            glScalef(self.skel_scale, self.skel_scale, self.skel_scale)

            name = list(self.env.muscle_info.keys())[self.muscle_index]
            wps = self.env.muscle_info[name]['waypoints']
            origin_body = wps[0]['body']
            insertion_body = wps[-1]['body']

            if bn.getName() in [origin_body, insertion_body]:
                self.meshes[bn.getName()].draw(np.array([1, 1, 0, self.obj_trans]))

                glColor4d(1,0,0,1)
                for i in range(len(self.meshes[bn.getName()].new_vertices_4) // 4):
                    v0, v1, v2, v3 = self.meshes[bn.getName()].new_vertices_4[i * 4 : (i + 1) * 4]
                    glBegin(GL_LINE_LOOP)
                    glVertex3f(v0[0], v0[1], v0[2])
                    glVertex3f(v1[0], v1[1], v1[2])
                    glVertex3f(v2[0], v2[1], v2[2])
                    glVertex3f(v3[0], v3[1], v3[2])
                    glEnd()

            else:
                self.meshes[bn.getName()].draw(np.array([color[0], color[1], color[2], self.obj_trans]))

            # self.meshes[bn.getName()].draw(np.array([color[0], color[1], color[2], self.obj_trans]))

            glPopMatrix()

        glPopMatrix()


    def drawSkeleton(self, pos, color = np.array([0.5, 0.5, 0.5, 0.5])):
        self.motion_skel.setPositions(pos)

        for bn in self.motion_skel.getBodyNodes():
            glPushMatrix()
            # self.meshes[bn.getName()].draw()
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

    def drawSKELskel(self, pos, color = np.array([0.5, 0.5, 1, 0.5])):
        self.skel_skel.setPositions(pos)

        for bn in self.skel_skel.getBodyNodes():
            glPushMatrix()
            bnWorldTransform = bn.getWorldTransform().matrix().transpose()
            glMultMatrixd(bnWorldTransform)
            j = bn.getParentJoint()
            jTransform = j.getTransformFromChildBodyNode().matrix().transpose()
            glMultMatrixd(jTransform)

            glColor4d(0, 0, 0, 1)
            mygl.draw_sphere(0.005 * np.sqrt(2), 10, 10)
            glPopMatrix()

        # for bn in self.skel_skel.getBodyNodes():
        #     transform = bn.getWorldTransform().matrix() @ bn.getParentJoint().getTransformFromChildBodyNode().matrix()
        #     t_parent = transform[:3, 3]

        #     glPushMatrix()
        #     numChild = bn.getNumChildBodyNodes()
        #     for i in range(numChild):
        #         bn_child = bn.getChildBodyNode(i)
        #         transform_child = bn_child.getWorldTransform().matrix() @ bn_child.getParentJoint().getTransformFromChildBodyNode().matrix()
        #         t_child = transform_child[:3, 3]

        #         glPushMatrix()
        #         m = (t_parent + t_child) / 2
        #         p2c = t_child - t_parent
        #         length = np.linalg.norm(p2c)
        #         p2c = p2c / length if length != 0 else np.array([0, 0, 1])
        #         z = np.array([0, 0, 1])

        #         axis = np.cross(z, p2c)
        #         s = np.linalg.norm(axis)
        #         axis = axis / s if s != 0 else np.array([0, 0, 1])
        #         c = np.dot(z, p2c)
        #         angle = np.rad2deg(np.arctan2(s, c))
                
        #         glTranslatef(m[0], m[1], m[2])
        #         glRotatef(angle, axis[0], axis[1], axis[2])
        #         mygl.draw_cube([0.01, 0.01, length])
        #         glPopMatrix()
        #     glPopMatrix()

        for bn in self.skel_skel.getBodyNodes():
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

        for bn in self.motion_skel.getBodyNodes():
            glPushMatrix()
            bnWorldTransform = bn.getWorldTransform().matrix().transpose()
            glMultMatrixd(bnWorldTransform)
            
            j = bn.getParentJoint()
            
            jTransform = j.getTransformFromChildBodyNode().matrix().transpose()
            glMultMatrixd(jTransform)

            glColor4d(color[0], color[1], color[2], color[3])
            mygl.draw_sphere(0.005 * np.sqrt(2), 10, 10)

            # glColor4d(1, 0, 0, 0.1)
            # mygl.draw_sphere(0.05, 10, 10)
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

    def drawTestMuscles(self, color=None):
        if color is not None:
            glColor4d(color[0], color[1], color[2], color[3])
        glLineWidth(self.line_width)
        idx = 0
        for m_wps in self.env.test_muscle_pos:
            glBegin(GL_LINE_STRIP)
            for wp in m_wps:
                glVertex3f(wp[0], wp[1], wp[2])
            glEnd() 
            idx += 1
       
    def drawMuscles(self, color=None):
        if color is not None:
            glColor4d(color[0], color[1], color[2], color[3])
        if self.draw_line_muscle:
            glLineWidth(self.line_width)
            for idx, m_wps in enumerate(self.env.muscle_pos):
                a = self.env.muscle_activation_levels[idx]
                if color is None:
                    if idx == self.muscle_index:
                        glColor4d(10, 0, 0, 1)
                    else:
                        glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
                    # glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
                glBegin(GL_LINE_STRIP)
                for wp in m_wps:
                    glVertex3f(wp[0], wp[1], wp[2])
                glEnd() 

                # # Draw Origin an dInsertion Points
                # glColor4d(1, 0, 0, 1)
                # for i in [0, -1]:
                #     glPushMatrix()
                #     glTranslatef(m_wps[i][0], m_wps[i][1], m_wps[i][2])
                #     mygl.draw_sphere(0.003, 10, 10)
                #     glPopMatrix()
        else:
            for idx, m_wps in enumerate(self.env.muscle_pos):
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
                    
        if self.origin_v is not None:
            glColor4d(10, 0, 0, 1)
            for v in [self.origin_v, self.insertion_v]:
                glPushMatrix()
                glTranslatef(v[0], v[1], v[2])
                mygl.draw_sphere(0.002, 5, 5)
                glPopMatrix()    

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

        self.drawSKELCharacter()
        if self.draw_gui_boxes:
            self.drawGuiBoxes()

        if self.mouse_down:
            glLineWidth(1.5)
            mygl.draw_axis()
        
        if self.draw_smpl_bone:
            self.drawSmplBone()
        if self.draw_smpl_joint:
            self.drawSmplJoint()
        glDepthMask(GL_FALSE)
        if self.draw_smpl:
            self.drawSmplCharacter(np.array([0.5, 0.5, 0.5, self.smpl_trans]))
        glDepthMask(GL_TRUE)

        if self.draw_target_motion:
            self.drawSkeleton(self.env.target_pos, np.array([1.0, 0.3, 0.3, 0.5]))
        
        if self.draw_bone:
            self.drawBone(self.env.skel.getPositions())
        if self.draw_joint:
            self.drawJoint(self.env.skel.getPositions())
        if self.draw_obj:
            self.drawObj(self.env.skel.getPositions())
        if self.draw_muscle:
            self.drawMuscles()
        if self.draw_body:
            self.drawSkeleton(self.env.skel.getPositions(), np.array([0.5, 0.5, 0.5, self.body_trans]))

        if self.draw_pd_target:
            self.drawSkeleton(self.env.pd_target, np.array([0.3, 0.3, 1.0, 0.5]))

        if self.draw_zero_pose:
            # glColor3f(0, 0, 0)
            for name, info in self.env.new_skel_info.items():
                # if info['stretch'] != "None":
                glPushMatrix()
                # if name == "Pelvis":
                #     print(info['joint_t'])
                glTranslatef(info['joint_t'][0], info['joint_t'][1], info['joint_t'][2])
                for i in range(len(info['stretches'])):
                    gap = info['gaps'][i]

                    # print(p)
                    # mygl.draw_sphere(0.005 * np.sqrt(2), 10, 10)

                    glBegin(GL_LINES)
                    a = gap
                    glColor3f(0, 0, 0)
                    glVertex3f(0,0,0)
                    glVertex3f(a[0] , a[1], a[2])
                    glEnd()

                if info.get('gaps_parent') is not None:
                    glColor3f(1, 0, 0)
                    glBegin(GL_LINES)
                    for gap_parent, stretch_axis in info['gaps_parent']:
                        glVertex3f(0,0,0)
                        glVertex3f(-gap_parent[0], -gap_parent[1], -gap_parent[2])
                    glEnd()

                glPopMatrix()

            if self.draw_joint:
                self.drawJoint(np.zeros(self.env.skel.getNumDofs()))
            self.drawSkeleton(np.zeros(self.env.skel.getNumDofs()), np.array([0.3, 1.0, 0.3, 0.5]))

        if self.draw_test_skeleton:
            if self.draw_bone:
                self.drawBone(self.test_dofs)
            if self.draw_joint:
                self.drawJoint(self.test_dofs)
            if self.draw_muscle:
                self.drawTestMuscles(np.array([0.1, 0.1, 0.1, 0.1]))
            self.drawSkeleton(self.test_dofs, np.array([0.3, 1.0, 1.0, 0.5]))

        if self.draw_dart_skel and self.env.skel_skel is not None:
                self.drawSKELskel(self.env.skel_skel.getPositions())
        if self.test_skel is not None and self.draw_skel_dofs:
                self.drawSKELskel(self.test_skel_dofs)

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

            if self.draw_smpl:
                self.drawSmplCharacter()
            
            glPopMatrix()


    def drawSmplBone(self):
        color = np.array([0, 0, 1, 0.5])
        glColor4d(color[0], color[1], color[2], color[3])
        for link in smpl_links:
            glPushMatrix()
            t_parent = self.smpl_joints[link[0]]
            t_child = self.smpl_joints[link[1]]
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

    def drawSmplJoint(self):
        highlight_color = np.array([0.0, 0.0, 1.0, 0.5])
        normal_color = np.array([0.0, 0.0, 0.5, 0.5])
        
        for i, j in enumerate(self.smpl_joints):
            if i == self.smpl_joint_index:
                color = highlight_color
            else:
                color = normal_color
            glColor4f(color[0], color[1], color[2], color[3])
            glPushMatrix()
            glTranslatef(j[0], j[1], j[2])
            mygl.draw_sphere(0.005 * np.sqrt(2), 10, 10)
            glPopMatrix()

    def drawGuiBoxes(self):
        if self.gui_boxes is None:
            return
        
        # pelvis = self.env.skel.getBodyNode("Pelvis").getCOM() - self.skel_joints[0].cpu().numpy()
        for box in self.gui_boxes:       
            glPushMatrix()
            # glTranslatef(pelvis[0], pelvis[1], pelvis[2])
            box.draw()
            glPopMatrix()

        if self.cand_gui_box is not None:
            self.cand_gui_box.draw()
            

    def reorder_vertices_by_depth(self, vertices, colors, normals):
        # Convert camera position to a NumPy array for easy calculations
        camera_position = self.eye

        # Calculate the centroid for each face
        num_faces = len(vertices) // 3  # Each face consists of 3 vertices
        centroids = np.zeros((num_faces, 3))

        for i in range(num_faces):
            # Get the vertices for the current face
            face_vertices = vertices[i * 3:(i + 1) * 3]
            # Calculate the centroid of the face
            centroids[i] = np.mean(face_vertices, axis=0)

        # Calculate depth for each face centroid
        depths = np.linalg.norm(centroids - camera_position, axis=1)

        # Create an index array for sorting
        sorted_indices = np.argsort(depths)[::-1]

        # Reorder vertices, colors, and normals based on sorted face indices
        sorted_vertices = np.vstack([vertices[i * 3:(i + 1) * 3] for i in sorted_indices])
        sorted_colors = np.vstack([colors[i * 3:(i + 1) * 3] for i in sorted_indices])
        sorted_normals = np.vstack([normals[i * 3:(i + 1) * 3] for i in sorted_indices])

        return sorted_vertices, sorted_colors, sorted_normals

    def drawSKELCharacter(self, color=np.array([0.5, 0.5, 0.5, 0.5])):
        # sorted_vertices, sorted_colors, sorted_normals = self.reorder_vertices_by_depth(self.skel_vertex3[0], self.skel_color4[0], self.skel_normal[0])

        root_dif = self.env.skel.getBodyNode("Pelvis").getCOM() - self.skel_joints[0].cpu().numpy()

        if self.draw_skel_joint_rot:
            for i in range(len(self.skel_joints)):
                j_ = self.skel_joints[i].cpu().numpy() + root_dif
                ori = self.skel_oris[i].cpu().numpy()
                ori = R.from_matrix(ori).as_rotvec()
                angle = np.linalg.norm(ori)
                axis = ori / angle if angle != 0 else np.array([0, 0, 1])
                glPushMatrix()

                glTranslatef(j_[0], j_[1], j_[2])
                glRotatef(np.rad2deg(angle), axis[0], axis[1], axis[2])
                glScalef(0.1, 0.1, 0.1)
                glBegin(GL_LINES)
                glColor3f(1, 0, 0)
                glVertex3f(0, 0 ,0)
                glVertex3f(1, 0, 0)
                glColor3f(0, 1, 0)
                glVertex3f(0, 0 ,0)
                glVertex3f(0, 1, 0)
                glColor3f(0, 0, 1)
                glVertex3f(0, 0 ,0)
                glVertex3f(0, 0, 1)
                glEnd()

                glPopMatrix()
        
        # Draw SKEL joint
        if self.draw_skel_joint:
            for i in range(len(self.skel_joints)):
                joint_p = self.skel_joints[i].cpu() + root_dif
                joint_orig_p = self.skel_joints_orig[i].cpu() + root_dif
                if i == self.skel_joint_index:
                    glColor3f(1, 0.7, 0.7)
                else:
                    glColor3f(0, 0, 0)
                glPushMatrix()
                glTranslatef(joint_p[0], joint_p[1], joint_p[2])
                mygl.draw_sphere(0.005 * np.sqrt(2), 10, 10)
                glPopMatrix()

                # glPushMatrix()
                # glColor3f(1, 0, 0)
                # glTranslatef(joint_orig_p[0], joint_orig_p[1], joint_orig_p[2])
                # mygl.draw_sphere(0.005 * np.sqrt(2), 10, 10)
                # glPopMatrix()

        # Draw SKEL joint edges
        if self.draw_skel_bone:
            glColor4d(color[0], color[1], color[2], color[3])
            for edge in skel_joint_edges:
                t_parent = self.skel_joints[edge[0]].cpu().numpy() + root_dif
                t_child = self.skel_joints[edge[1]].cpu().numpy() + root_dif

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
        
        # glColor3f(1, 0, 0)
        # for i in range(0, len(self.cand_skin_right_faces) // 3):
        #     v0, v1, v2 = self.skin_vertices[self.cand_skin_right_faces[i*3:i*3+3]] + root_dif
        #     if i == self.add_skin_index:
        #         glColor3f(0, 0, 1)
        #     else:
        #         glColor3f(1, 0, 0)
        #     glBegin(GL_TRIANGLES)
        #     glVertex3f(v0[0], v0[1], v0[2])
        #     glVertex3f(v1[0], v1[1], v1[2])
        #     glVertex3f(v2[0], v2[1], v2[2])
        #     glEnd() 

        # Used for skeleton sectioning
        glPushMatrix()
        for i in range(self.skel_face_start_index, self.skel_face_index):
            glColor3f(1, 1, 1)
            v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]]# + root_dif
            glBegin(GL_TRIANGLES)
            glVertex3f(v0[0], v0[1], v0[2])
            glVertex3f(v1[0], v1[1], v1[2])
            glVertex3f(v2[0], v2[1], v2[2])
            glEnd()

            if i == self.skel_face_index - 1:
                glColor3f(1, 0, 0)
            else:
                glColor3f(0, 0, 1)
            glBegin(GL_LINE_LOOP)
            glVertex3f(v0[0], v0[1], v0[2])
            glVertex3f(v1[0], v1[1], v1[2])
            glVertex3f(v2[0], v2[1], v2[2])
            glEnd()
        glPopMatrix()

        # glPushMatrix()
        # for i in [106843, 106844]:
        #     glColor3f(1,1,1)
        #     v0, v1, v2 = self.skel_vertices[self.skel_faces[i*3:i*3+3]]# + root_dif
        #     # print(self.skel_faces[i*3:i*3+3])
        #     glBegin(GL_TRIANGLES)
        #     glVertex3f(v0[0], v0[1], v0[2])
        #     glVertex3f(v1[0], v1[1], v1[2])
        #     glVertex3f(v2[0], v2[1], v2[2])
        #     glEnd()

        #     if i == self.skel_face_index - 1:
        #         glColor3f(1, 0, 0)
        #     else:
        #         glColor3f(0, 0, 1)
        #     glBegin(GL_LINE_LOOP)
        #     glVertex3f(v0[0], v0[1], v0[2])
        #     glVertex3f(v1[0], v1[1], v1[2])
        #     glVertex3f(v2[0], v2[1], v2[2])
        #     glEnd()
        # glPopMatrix()

        # glPushMatrix()
        # for i in [[200629, 200652]]:
        #     glColor3f(1, 1, 1)
        #     v0, v1 = self.skel_vertices[i]
        #     mean = (v0 + v1) / 2
        #     glPushMatrix()
        #     glTranslatef(mean[0], mean[1], mean[2])
        #     mygl.draw_sphere(0.005, 10, 10)
        #     glPopMatrix()
        # glPopMatrix()

        
        if self.draw_skel_skel:
            # for i, midpoint in enumerate(self.SKEL_section_midpoints):
            #     if i in [131, 132, 133, 134, 135]: # Cervix: 
            #         glColor3f(200, 0, 0)
            #     elif i in [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]: # Thorax
            #         glColor3f(0, 200, 0)
            #     elif i in [67, 68, 69, 70, 71]: # Lumbar
            #         glColor3f(0, 0, 200)
            #     else:
            #         continue
            #     glPushMatrix()
            #     glTranslatef(midpoint[0], midpoint[1], midpoint[2])
            #     mygl.draw_sphere(0.005, 10, 10)
            #     glPopMatrix()

            # L1 = self.SKEL_section_midpoints[68]
            # C7 = self.SKEL_section_midpoints[133]
            # glBegin(GL_LINES)
            # glVertex3f(L1[0], L1[1], L1[2])
            # glVertex3f(C7[0], C7[1], C7[2])
            # glEnd()

            # T11 = self.SKEL_section_midpoints[73]
            # T12 = self.SKEL_section_midpoints[78]
            # L2 = self.SKEL_section_midpoints[70]
            # glColor3f(0, 0, 200)
            # glBegin(GL_LINE_STRIP)
            # glVertex3f(T11[0], T11[1], T11[2])
            # glVertex3f(T12[0], T12[1], T12[2])
            # glVertex3f(L1[0], L1[1], L1[2])
            # glVertex3f(L2[0], L2[1], L2[2])
            # glEnd()
            
            if self.isSKELSection:
                SKEL_section_toggle = self.SKEL_section_toggle_male if self.skel_gender == "male" else self.SKEL_section_toggle_female
                for i, isOn in enumerate(SKEL_section_toggle):
                    if isOn:
                        # min = np.min(self.SKEL_section_vertex3[i], axis=0)
                        # max = np.max(self.SKEL_section_vertex3[i], axis=0)
                        # mean = (min + max) / 2
                        # print((mean - self.gui_boxes[i].pos)[1])
                        # for pos in [min, max, mean]:
                        #     glPushMatrix()
                        #     glTranslatef(pos[0], pos[1], pos[2])
                        #     glColor4d(0, 1, 1, 1)
                        #     mygl.draw_sphere(0.005, 10, 10)
                        #     glPopMatrix()

                        glPushMatrix()
                        glEnableClientState(GL_COLOR_ARRAY)
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                        glColorPointer(4, GL_FLOAT, 0, self.SKEL_section_color4[i])
                        glEnableClientState(GL_VERTEX_ARRAY)
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                        glVertexPointer(3, GL_FLOAT, 0, self.SKEL_section_vertex3[i])
                        glEnableClientState(GL_NORMAL_ARRAY)
                        glNormalPointer(GL_FLOAT, 0, self.SKEL_section_normal[i])
                        glDrawArrays(GL_TRIANGLES, 0, len(self.SKEL_section_vertex3[i]))
                        glDisableClientState(GL_NORMAL_ARRAY)
                        glDisableClientState(GL_VERTEX_ARRAY)
                        glDisableClientState(GL_COLOR_ARRAY)
                        glPopMatrix()

                        if i == self.selected_mesh_index and self.selected_face is not None:
                            glColor4d(10,10,10,1)
                            for j in range(len(self.SKEL_section_vertex3[i]) // 3):
                                v0, v1, v2 = self.SKEL_section_vertex3[i][j * 3 : (j + 1) * 3]
                                glBegin(GL_LINE_LOOP)
                                glVertex3f(v0[0], v0[1], v0[2])
                                glVertex3f(v1[0], v1[1], v1[2])
                                glVertex3f(v2[0], v2[1], v2[2])
                                glEnd()

                            glPushMatrix()
                            glColor3f(10, 10, 10)
                            v0, v1, v2 = self.SKEL_section_vertex3[i][self.selected_face*3:self.selected_face*3+3]
                            glBegin(GL_TRIANGLES)
                            glVertex3f(v0[0], v0[1], v0[2])
                            glVertex3f(v1[0], v1[1], v1[2])
                            glVertex3f(v2[0], v2[1], v2[2])
                            glEnd()

                            glColor3f(0, 0, 0)
                            glBegin(GL_LINE_LOOP)
                            glVertex3f(v0[0], v0[1], v0[2])
                            glVertex3f(v1[0], v1[1], v1[2])
                            glVertex3f(v2[0], v2[1], v2[2])
                            glEnd()

                            for j, v in enumerate([v0, v1, v2]):
                                glPushMatrix()
                                if j == 0:
                                    glColor3f(10, 0, 0)
                                elif j == 1:
                                    glColor3f(0, 10, 0)
                                else:
                                    glColor3f(0, 0, 10)
                                glTranslatef(v[0], v[1], v[2])
                                mygl.draw_sphere(0.0005, 5, 5)
                                glPopMatrix()

                            glColor3f(0, 0, 0)
                            a, b, c = self.selected_point
                            glTranslatef(a, b, c)
                            mygl.draw_sphere(0.0005, 5, 5)

                            glPopMatrix()
  
            else:
                glPushMatrix()
                glEnableClientState(GL_COLOR_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glColorPointer(4, GL_FLOAT, 0, self.skel_color4[0])
                glEnableClientState(GL_VERTEX_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glVertexPointer(3, GL_FLOAT, 0, self.skel_vertex3[0])
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, self.skel_normal[0])
                glDrawArrays(GL_TRIANGLES, 0, len(self.skel_vertex3[0]))
                glDisableClientState(GL_NORMAL_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_COLOR_ARRAY)
                glPopMatrix()

        if self.draw_skel_skin:
            glPushMatrix()
            glEnableClientState(GL_COLOR_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glColorPointer(4, GL_FLOAT, 0, self.skin_color4[0])
            glEnableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glVertexPointer(3, GL_FLOAT, 0, self.skin_vertex3[0])
            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, 0, self.skin_normal[0])
            glDrawArrays(GL_TRIANGLES, 0, len(self.skin_vertex3[0]))
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glPopMatrix()

        

    def drawSmplCharacter(self, color=np.array([0.5, 0.5, 0.5, 0.5])):
        glPushMatrix()
        glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glColorPointer(4, GL_FLOAT, 0, self.smpl_color4[0])
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glVertexPointer(3, GL_FLOAT, 0, self.smpl_vertex3[0])
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, self.smpl_normal[0])
        glDrawArrays(GL_TRIANGLES, 0, len(self.smpl_vertex3[0]))
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glPopMatrix()

    def retargetting(self, name, index=0):
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
                if waypoint['body'] == name and not name in ["Pelvis", "Spine", "Torso"]:
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
                
                # parent_stretch = parent_stretches[0]
                # parent_stretch_axis = parent_info['stretch_axises'][0]
                # parent_size = new_parent_info['size'][parent_stretch]
                # parent_gap = parent_info['gaps'][0]

                new_child_info = self.env.new_skel_info[child]
                child_info = self.env.skel_info[child]

                child_stretch = child_info['stretches'][0]
                child_stretch_axis = child_info['stretch_axises'][0]
                child_size = new_child_info['size'][child_stretch]
                child_gap = child_info['gaps'][0]
                child_gap_parent_tuple = child_info['gaps_parent'][0]
                child_gap_parent = child_gap_parent_tuple[0]
                same_direction_to_parent = child_gap_parent_tuple[1]

                if same_direction_to_parent:
                    new_child_info['joint_t'] = new_parent_info['joint_t'] + parent_gap + parent_stretch_axis * parent_size + child_gap_parent
                else:
                    new_child_info['joint_t'] = new_parent_info['joint_t'] + parent_gap + child_gap_parent
                new_child_info['body_t'] = new_child_info['joint_t'] + child_gap + child_stretch_axis * child_size * 0.5

                # if parent_str == name and len(parent_stretches) > 1:
                #     for i in range(len(parent_stretches)):
                #         if i != index:
                #             parent_stretch_other = parent_info['stretches'][i]
                #             parent_stretch_axis_other = parent_info['stretch_axises'][i]
                #             parent_size_other = new_parent_info['size'][parent_stretch_other]
                #             parent_gap_other = parent_info['gaps'][i]
                #             new_child_info['gaps_parent'][i] = new_child_info['joint_t'] - (new_parent_info['joint_t'] + parent_gap_other + parent_stretch_axis_other * parent_size_other)

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
                        if waypoint['body'] == child and not child in ["Pelvis", "Spine", "Torso"]:
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

    def scaleSkeleton(self, scale):
        for name, info in self.env.new_skel_info.items():
            orig_info = self.env.skel_info[name]
            info['joint_t'] = orig_info['joint_t'] * scale
            info['body_t'] = orig_info['body_t'] *  scale
            info['size'] = orig_info['size'] * scale
            for i in range(len(info['gaps'])):
                info['gaps'][i] = orig_info['gaps'][i] * scale
            if info['parent_str'] != "None":
                for i in range(len(info['gaps_parent'])):
                    info['gaps_parent'][i][0] = orig_info['gaps_parent'][i][0] * scale
        
        for name, info in self.env.new_muscle_info.items():
            orig_muscle_info = self.env.muscle_info[name]
            for waypoint_i, waypoint in enumerate(info['waypoints']):
                orig_waypoint = orig_muscle_info['waypoints'][waypoint_i]
                waypoint['p'] = orig_waypoint['p'] * scale
                for i in range(len(waypoint['gaps'])):
                    waypoint['gaps'][i] = orig_waypoint['gaps'][i] * scale
        
    def newSkeleton(self):
        current_pos = self.env.skel.getPositions()
        self.env.world.removeSkeleton(self.env.skel)
        self.env.skel = buildFromInfo(self.env.new_skel_info, self.env.root_name)
        self.env.target_skel = self.env.skel.clone()
        self.env.world.addSkeleton(self.env.skel)

        self.env.loading_muscle_info(self.env.new_muscle_info)
        self.env.loading_test_muscle_info(self.env.new_muscle_info)

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
            _, self.draw_test_skeleton = imgui.checkbox("Draw test Skeleton", self.draw_test_skeleton)
            _, self.draw_body = imgui.checkbox("Draw Body", self.draw_body)
            if self.draw_body:
                imgui.same_line()
                imgui.push_item_width(100)
                _, self.body_trans = imgui.slider_float("Body Transparency",
                                                                self.body_trans,
                                                                min_value = 0.0,
                                                                max_value = 1.0,
                                                                format='%.3f')
                imgui.pop_item_width()
            _, self.draw_muscle = imgui.checkbox("Draw Muscle", self.draw_muscle)
            if self.draw_muscle:
                imgui.same_line()
                if imgui.radio_button("Line Muscle", self.draw_line_muscle):
                    self.draw_line_muscle = True 
                imgui.same_line()
                if imgui.radio_button("Cube Muscle", not self.draw_line_muscle):
                    self.draw_line_muscle = False

                changed, self.line_width = imgui.slider_float("Line Width", self.line_width, 0.1, 5.0)

                # Show selected SKEL joint
                if imgui.button("<##muscle"):
                    self.muscle_index -= 1
                    if self.muscle_index < 0:
                        self.muscle_index = len(self.env.muscle_pos) - 1
                imgui.same_line()
                imgui.text(f"{self.muscle_index}")
                imgui.same_line()
                if imgui.button(">##muscle"):
                    self.muscle_index += 1
                    if self.muscle_index >= len(self.env.muscle_pos):
                        self.muscle_index = 0
                imgui.same_line()
                if imgui.button("Find End Vertices"):
                    name = list(self.env.muscle_info.keys())[self.muscle_index]
                    wps_info = self.env.muscle_info[name]['waypoints']
                    origin_body = wps_info[0]['body']
                    insertion_body = wps_info[-1]['body']

                    wps = self.env.muscle_pos[self.muscle_index]
                    origin_p = wps[0]
                    insertion_p = wps[-1]

                    for bn in self.motion_skel.getBodyNodes():
                        if bn.getName() == origin_body:
                            origin_bn = bn
                        if bn.getName() == insertion_body:
                            insertion_bn = bn

                    # glPushMatrix()

                    # glTranslatef(t_parent[0], t_parent[1], t_parent[2])

                    # rot = R.from_matrix(r_parent)
                    # rotvec = rot.as_rotvec()
                    # angle = np.linalg.norm(rotvec)
                    # axis = rotvec / angle if angle != 0 else np.array([0, 0, 1])
                    # glRotatef(np.rad2deg(angle), axis[0], axis[1], axis[2])
                    
                    # glScalef(self.skel_scale, self.skel_scale, self.skel_scale)

                    min_origin = np.inf
                    origin_v = None
                    transform = origin_bn.getWorldTransform().matrix() @ origin_bn.getParentJoint().getTransformFromChildBodyNode().matrix()
                    t_origin = transform[:3, 3]
                    r_origin = transform[:3, :3]

                    origin_vertices = self.meshes[origin_body].new_vertices_4
                    origin_vertices = origin_vertices @ r_origin.T + t_origin
                    for v in origin_vertices:
                        dist = np.linalg.norm(v - origin_p)
                        if dist < min_origin:
                            min_origin = dist
                            origin_v = v

                    min_insertion = np.inf
                    insertion_v = None
                    transform = insertion_bn.getWorldTransform().matrix() @ insertion_bn.getParentJoint().getTransformFromChildBodyNode().matrix()
                    t_insertion = transform[:3, 3]
                    r_insertion = transform[:3, :3]

                    insertion_vertices = self.meshes[insertion_body].new_vertices_4
                    insertion_vertices = insertion_vertices @ r_insertion.T + t_insertion
                    for v in insertion_vertices:
                        dist = np.linalg.norm(v - insertion_p)
                        if dist < min_insertion:
                            min_insertion = dist
                            insertion_v = v

                    self.origin_v = origin_v
                    self.insertion_v = insertion_v
                    print(origin_v, insertion_v)
                    
            _, self.draw_obj = imgui.checkbox("Draw Object", self.draw_obj)
            imgui.same_line()
            imgui.push_item_width(100)
            changed, self.obj_trans = imgui.slider_float("OBJ Transparency",
                                                            self.obj_trans,
                                                            min_value = 0.0,
                                                            max_value = 1.0,
                                                            format='%.3f')
            imgui.pop_item_width()
            _, self.draw_bone = imgui.checkbox("Draw Bone", self.draw_bone)
            _, self.draw_joint = imgui.checkbox("Draw Joint", self.draw_joint)
            _, self.draw_shadow = imgui.checkbox("Draw Shadow", self.draw_shadow)


            _, self.draw_smpl = imgui.checkbox("Draw SMPL", self.draw_smpl)
            if self.draw_smpl:
                imgui.same_line()
                imgui.push_item_width(100)
                changed, self.smpl_trans = imgui.slider_float("SMPL Transparency",
                                                                self.smpl_trans,
                                                                min_value = 0.0,
                                                                max_value = 1.0,
                                                                format='%.3f')
                imgui.pop_item_width()
                if changed:
                    self.update_smpl()
            _, self.draw_smpl_bone = imgui.checkbox("Draw SMPL Bone", self.draw_smpl_bone)
            _, self.draw_smpl_joint = imgui.checkbox("Draw SMPL Joint", self.draw_smpl_joint)
            imgui.same_line()
            if imgui.button("<##smpl_joint"):
                self.smpl_joint_index -= 1
                if self.smpl_joint_index < 0:
                    self.smpl_joint_index = len(self.smpl_joints) - 1
            imgui.same_line()

            imgui.push_item_width(150)
            # imgui.text("%02d: %s" % (self.smpl_joint_index, smpl_joint_names[self.smpl_joint_index]))
            imgui.text(f"{self.smpl_joint_index:02d}: {smpl_joint_names[self.smpl_joint_index]:<{10}}")
            imgui.pop_item_width()

            imgui.same_line()
            if imgui.button(">##smpl_joint"):
                self.smpl_joint_index += 1
                if self.smpl_joint_index >= len(self.smpl_joints):
                    self.smpl_joint_index = 0
            imgui.same_line()
            if imgui.button("Print All##SMPL"):
                print()
                print("skel_joint_ts = np.array([")
                for name, info in self.env.skel_info.items():
                    joint_t = info['joint_t']
                    # print(np.round(info['joint_t'], 4), f', # {name}')
                    print(f"[{joint_t[0]:.4f}, {joint_t[1]:.4f}, {joint_t[2]:.4f}], # {name}")
                print("])")

                print("skel_body_ts = np.array([")
                for name, info in self.env.new_skel_info.items():
                    body_t = info['body_t']
                    #print(np.round(info['body_t'], 4), f', # {name}')
                    print(f"[{body_t[0]:.4f}, {body_t[1]:.4f}, {body_t[2]:.4f}], # {name}")
                print("])")

                print()
                print("SMPL_joint_ts = np.array([")
                for i, j in enumerate(self.smpl_joints):
                    joint_t = j.clone().numpy()
                    # print(np.round(j.clone().numpy(), 4), f', # {smpl_joint_names[i]}')
                    print(f"[{joint_t[0]:.4f}, {joint_t[1]:.4f}, {joint_t[2]:.4f}], # {smpl_joint_names[i]}")
                print("])")

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

            imgui.push_item_width(150)
            changed, self.skel_scale = imgui.slider_float("Skel scale",
                                                            self.skel_scale,
                                                            min_value = 0.8,
                                                            max_value = 1.2,
                                                            format='%.3f')
            imgui.pop_item_width()

            imgui.same_line()
            imgui.set_cursor_pos_x(300)
            if changed:
                self.scaleSkeleton(self.skel_scale)
                self.newSkeleton()

            if imgui.button("Reset##skel_scale"):
                self.skel_scale = 1
                self.scaleSkeleton(self.skel_scale)
                self.newSkeleton()
                
            for name, info in self.env.new_skel_info.items():
                orig_info = self.env.skel_info[name]

                stretches = info['stretches']

                for i in range(len(stretches)):
                    stretch = stretches[i]

                    if stretch == 0:
                        name_new = name + " x"
                    elif stretch == 1:
                        name_new = name + " y"
                    elif stretch == 2:
                        name_new = name + " z"

                    orig_size = orig_info['size'][stretch]
                    
                    imgui.push_item_width(150)
                    changed, info['size'][stretch] = imgui.slider_float(name_new,
                                                                        info['size'][stretch],
                                                                        min_value = orig_size * 0.25,
                                                                        max_value = orig_size * 4,
                                                                        format='%.3f')
                    imgui.pop_item_width()

                    if changed:
                        gap = info['gaps'][i]
                        stretch_axis = info['stretch_axises'][i]

                        info['body_t'] = info['joint_t'] + gap + stretch_axis * info['size'][stretch] * 0.5
                        
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

                        self.setObjScale()
                        self.newSkeleton()

                    imgui.same_line()
                    imgui.set_cursor_pos_x(300)
                    if imgui.button("Reset##" + name_new):
                        info['size'][stretch] = orig_info['size'][stretch].copy()
                        gap = info['gaps'][i]
                        stretch_axis = info['stretch_axises'][i]

                        info['body_t'] = info['joint_t'] + gap + stretch_axis * info['size'][stretch] * 0.5

                        self.retargetting(name, i)

                        if self.skel_change_symmetry and name[-1] in ['R', 'L']:
                            if name[-1] == "R":
                                name_pair = name[:-1] + "L"
                            elif name[-1] == "L":
                                name_pair = name[:-1] + "R"

                            info_pair = self.env.new_skel_info[name_pair]
                            info_pair['size'][stretch] = self.env.skel_info[name_pair]['size'][stretch].copy()

                            size_pair = info_pair['size'][stretch]
                            gap_pair = info_pair['gaps'][i]
                            stretch_axis_pair = info_pair['stretch_axises'][i]

                            info_pair['body_t'] = info_pair['joint_t'] + gap_pair + stretch_axis_pair * size_pair * 0.5

                            self.retargetting(name_pair, i)
                            
                        self.setObjScale()
                        self.newSkeleton()

            if imgui.button("Reset Skeleton"):
                self.skel_scale = 1
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

                self.setObjScale()
                self.newSkeleton()

            if imgui.button("Export Skeleton & Muscle"):
                exportSkeleton(self.env.new_skel_info, self.env.root_name, "new_skel.xml")
                self.env.exportMuscle(self.env.new_muscle_info, "new_muscle.xml")

            imgui.tree_pop()

        if imgui.tree_node("Test rotvecs"):
            for i in range(self.motion_skel.getNumDofs()):
                imgui.push_item_width(150)
                changed, self.test_dofs[i] = imgui.slider_float(f"DOF {i}", self.test_dofs[i], -3.0, 3.0)
                if changed:
                    self.env.test_skel.setPositions(self.test_dofs)
                    self.env.test_muscles.update()
                    self.env.test_muscle_pos = self.env.test_muscles.getMusclePositions()

                imgui.pop_item_width()
                imgui.same_line()
                if imgui.button(f"Reset##test_dof{i}"):
                    self.test_dofs[i] = 0.0
                    self.env.test_skel.setPositions(self.test_dofs)
                    self.env.test_muscles.update()
                    self.env.test_muscle_pos = self.env.test_muscles.getMusclePositions()
            imgui.tree_pop()
        
        if imgui.tree_node("SKEL Parameters"):
            if imgui.button("Export Skeleton"):
                pass
                filename = "skel_skel.xml"
                def tw(file, string, tabnum):
                    for _ in range(tabnum):
                        file.write("\t")
                    file.write(string + "\n")

                f = open(f"data/{filename}", 'w')
                tw(f, "<Skeleton name=\"Skeleton\">", 0)

                for box in self.gui_boxes:
                    tw(f, "<Node name=\"%s\" parent=\"%s\">" % (box.name, box.parent), 1)

                    tw(f, "<Body type=\"%s\" mass=\"%f\" size=\"%s\" contact=\"%s\" color=\"%s\" obj=\"%s\" stretch=\"%s\">" % 
                        ("Box", 
                        7.0, 
                        " ".join(np.array(box.size).astype(str)), 
                        "On" if box.isContact else "Off", 
                        " ".join(np.array(box.color).astype(str)), 
                        "",
                        "1",
                        ), 
                    2)
                    
                    tw(f, "<Transformation linear=\"%s\" translation=\"%s\"/>" % 
                    (" ".join(R.from_rotvec(box.rot).as_matrix().astype(str).flatten()), 
                        " ".join(box.pos.astype(str))
                        ), 
                        3)

                    tw(f, "</Body>", 2)

                    upper = np.array(box.upper) if box.upper is not None else np.array([1.6, 1.6, 1.6])
                    lower = np.array(box.lower) if box.lower is not None else np.array([-1.6, -1.6, -1.6])

                    # upper = np.array([0.01, 0.01, 0.01])
                    # lower = np.array([-0.01, -0.01, -0.01])

                    if box.jointType == "Free":
                        tw(f, "<Joint type=\"Free\">", 2)
                    elif box.jointType == "Ball":
                        tw(f, "<Joint type=\"Ball\" lower=\"%s\" upper=\"%s\">" %
                                    (" ".join(lower.astype(str)),
                                    " ".join(upper.astype(str)),
                                    ),
                                    2)
                    elif box.jointType == "Revolute":
                        tw(f, "<Joint type=\"Revolute\" axis=\"%s\" lower=\"%s\" upper=\"%s\">" %
                                    (" ".join(box.axis.astype(str)),
                                    lower.astype(str),
                                    upper.astype(str),
                                    ),
                                    2)
                        
                    tw(f, "<Transformation linear=\"%s\" translation=\"%s\"/>" %
                            (" ".join(box.ori.astype(str).flatten()),
                            " ".join(np.array(box.joint).astype(str))),
                            3)
                    
                    tw(f, "</Joint>", 2)

                    tw(f, "</Node>", 1)
                tw(f, "</Skeleton>", 0)
                f.close()

                from core.dartHelper import saveSkeletonInfo, buildFromInfo
                skel_info, root_name, _, _, _, _ = saveSkeletonInfo("data/skel_skel.xml")
                self.env.skel_skel = buildFromInfo(skel_info, root_name)
                self.skel_skel = self.env.skel_skel.clone()
                self.env.world.addSkeleton(self.env.skel_skel)
                self.env.kp_skel = 300.0 * np.ones(self.skel_skel.getNumDofs()) 
                self.env.kv_skel = 20.0 * np.ones(self.skel_skel.getNumDofs())
                self.env.kp_skel[:6] = 0.0
                self.env.kv_skel[:6] = 0.0

                self.gui_boxes = []

                self.test_skel = self.skel_skel.clone()
                if self.test_skel_dofs is None:
                    self.test_skel_dofs = np.zeros(self.test_skel.getNumDofs())

            if self.test_skel is not None:
                if imgui.tree_node("Test rotvecs"):
                    for i in range(self.skel_skel.getNumDofs()):
                        imgui.push_item_width(150)
                        changed, self.test_skel_dofs[i] = imgui.slider_float(f"DOF {i}", self.test_skel_dofs[i], -3.0, 3.0)
                        imgui.pop_item_width()
                        imgui.same_line()
                        if imgui.button(f"Reset##test_dof{i}"):
                            self.test_skel_dofs[i] = 0.0
                    imgui.tree_pop()
                
            _, self.draw_skel_skin = imgui.checkbox("Draw SKEL skin", self.draw_skel_skin)
            imgui.same_line()
            imgui.set_cursor_pos_x(200)
            imgui.push_item_width(100)
            changed, self.skin_opacity = imgui.slider_float(f"Opacity##skin", self.skin_opacity, 0, 1)
            if changed:
                self.update_skel()
            imgui.pop_item_width()
            _, self.draw_skel_skel = imgui.checkbox("Draw SKEL skel", self.draw_skel_skel)
            imgui.same_line()
            imgui.set_cursor_pos_x(200)
            imgui.push_item_width(100)
            changed, self.skel_opacity = imgui.slider_float(f"Opacity##skel", self.skel_opacity, 0, 1)
            if changed:
                self.update_skel()
            imgui.pop_item_width()
            _, self.draw_skel_joint = imgui.checkbox("Draw SKEL joint", self.draw_skel_joint)
            _, self.draw_skel_joint_rot = imgui.checkbox("Draw SKEL joint rot", self.draw_skel_joint_rot)
            _, self.draw_skel_bone = imgui.checkbox("Draw SKEL bone", self.draw_skel_bone)
            _, self.draw_dart_skel = imgui.checkbox("Draw SKEL dart", self.draw_dart_skel)
            _, self.draw_skel_dofs = imgui.checkbox("Draw Test Skeleton", self.draw_skel_dofs)
            _, self.draw_gui_boxes = imgui.checkbox("Draw GUI boxes", self.draw_gui_boxes)
            if self.draw_skel_skin:
                changed, self.isDrawHalfSkin = imgui.checkbox("Draw Half Skin", self.isDrawHalfSkin)
                if changed:
                    self.update_skel()
                if self.isDrawHalfSkin:
                    imgui.set_cursor_pos_x(200)
                    imgui.same_line()
                    if imgui.radio_button("Left", self.skin_direction == "left"):
                        self.skin_direction = "left"
                        self.update_skel()
                    imgui.same_line()
                    if imgui.radio_button("Right", self.skin_direction == "right"):
                        self.skin_direction = "right"
                        self.update_skel()
            
            if imgui.radio_button("Male", self.skel_gender == "male"):
                self.skel_gender = "male"
                self.update_skel(gender_changed=True)
            imgui.same_line()
            if imgui.radio_button("Female", self.skel_gender == "female"):
                self.skel_gender = "female"
                self.update_skel(gender_changed=True)

            changed, self.isSKELSection = imgui.checkbox("Show SKEL section", self.isSKELSection)
            if changed:
                self.update_skel()

            if self.isSKELSection:
                # clicked, self.SKEL_section_index = imgui.listbox('', self.SKEL_section_index, self.SKEL_section_names)
                # if clicked:
                #     self.update_skel()
                if imgui.tree_node("SKEL Sections"):
                    if self.skel_gender == "male":
                        SKEL_section_toggle = self.SKEL_section_toggle_male
                        SKEL_section_names = self.SKEL_section_names_male
                    else:
                        SKEL_section_toggle = self.SKEL_section_toggle_female
                        SKEL_section_names = self.SKEL_section_names_female

                    if imgui.button("Choose All"):
                        for i in range(len(SKEL_section_toggle)):
                            SKEL_section_toggle[i] = True
                    imgui.same_line()
                    if imgui.button("Choose None"):
                        for i in range(len(SKEL_section_toggle)):
                            SKEL_section_toggle[i] = False
                    imgui.same_line()
                    changed, self.SKEL_section_unique = imgui.checkbox("Choose One Section", self.SKEL_section_unique)
                    if changed:
                        if self.SKEL_section_unique == True:
                            SKEL_section_toggle[0] = True
                            for i in range(1, len(SKEL_section_toggle)):
                                SKEL_section_toggle[i] = False
                        else:
                            for i in range(len(SKEL_section_toggle)):
                                SKEL_section_toggle[i] = True

                    for i in range(len(SKEL_section_toggle)):
                        changed, SKEL_section_toggle[i] = imgui.checkbox(SKEL_section_names[i], SKEL_section_toggle[i])
                        if changed:
                            if self.SKEL_section_unique and SKEL_section_toggle[i] == True:
                                for j in range(len(SKEL_section_toggle)):
                                    if j != i:
                                        SKEL_section_toggle[j] = False
                            self.update_skel()
                    imgui.tree_pop()

            imgui.text("SKEL Betas")
            for i in range(len(self.skel_betas[0])):
                imgui.push_item_width(150)
                changed, self.skel_betas[0][i] = imgui.slider_float(f"Beta {i}", self.skel_betas[0][i], -5.0, 5.0)
                if changed:
                    self.gui_boxes = []
                    self.update_skel()
                imgui.pop_item_width()
                imgui.same_line()
                if imgui.button(f"Reset##skelbeta{i}"):
                    self.gui_boxes = []
                    self.skel_betas[0][i] = 0.0
                    self.update_skel()
            imgui.text("SKEL Pose")
            for i in range(len(self.skel_pose[0])):
                imgui.push_item_width(150)
                pose_param_name = pose_param_names[i]
                if pose_param_name in pose_limits.keys():
                    lower = pose_limits[pose_param_name][0]
                    upper = pose_limits[pose_param_name][1]
                    if lower > upper:
                        lower, upper = upper, lower
                else:
                    lower = -3.14
                    upper = 3.14
                changed, self.skel_pose[0][i] = imgui.slider_float(pose_param_name, self.skel_pose[0][i], lower, upper)
                if changed:
                    self.update_skel()
                imgui.pop_item_width()
                imgui.same_line()
                if imgui.button(f"Reset##skelpose{i}"):
                    self.skel_pose[0][i] = self.skel_pose_base[0][i].clone()
                    self.update_skel()
            
            imgui.tree_pop()

        if imgui.tree_node("Smpl Shape Parameters"):
            ## Smpl Shape Parameters
            for i in range(len(self.shape_parameters)):
                imgui.push_item_width(200)
                changed, self.shape_parameters[i] = imgui.slider_float("Beta %d" % (i), self.shape_parameters[i], -5.0, 5.0)
                if changed:
                    self.update_smpl()
                    self.fitSkel2SMPL()
                    self.setObjScale()
                    self.newSkeleton()
                    
                imgui.pop_item_width()
                imgui.same_line()
                if imgui.button(f"Reset##smplbeta{i}"):
                    self.shape_parameters[i] = 0.0
                    self.update_smpl()
                    self.fitSkel2SMPL()
                    self.setObjScale()
                    self.newSkeleton()

            imgui.push_item_width(200)
            changed, self.smpl_scale = imgui.slider_float("Scale", self.smpl_scale, 1, 1.5)
            if changed:
                self.update_smpl()
            imgui.pop_item_width()
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

        
        # # Used for selecting half skin faces
        # if imgui.button("Add this face"):
        #     if len(self.cand_skin_right_faces) >= 3:
        #         self.skin_right_faces.extend(self.cand_skin_right_faces[0:3])
        #         self.cand_skin_right_faces = self.cand_skin_right_faces[3:]
        #         self.update_skel()
        # imgui.same_line()
        # if imgui.button("Remove this face"):
        #     if len(self.cand_skin_right_faces) >= 3:
        #         self.skin_left_faces.extend(self.cand_skin_right_faces[0:3])
        #         self.cand_skin_right_faces = self.cand_skin_right_faces[3:]
        #         self.update_skel()
        # imgui.same_line()
        # if imgui.button("Print faces"):
        #     print("Right faces")
        #     print(self.skin_right_faces)
        #     if len(self.cand_skin_right_faces) > 0:
        #         print("Remaining right faces")
        #         print(self.cand_skin_right_faces)
        #     print("Left faces")
        #     print(self.skin_left_faces)

        #     f = open(f"{self.skel_gender}_face_info.py", "w")
        #     f.write(f"skin_right_faces = {self.skin_right_faces}\n")
        #     f.write(f"skin_left_faces = {self.skin_left_faces}\n")
        #     f.close()
        
        # Show selected SKEL joint
        if imgui.button("<##skeljoint"):
            self.skel_joint_index -= 1
            if self.skel_joint_index < 0:
                self.skel_joint_index = len(self.skel_joints) - 1
        imgui.same_line()
        imgui.text(f"{self.skel_joint_index}")
        imgui.same_line()
        if imgui.button(">##skeljoint"):
            self.skel_joint_index += 1
            if self.skel_joint_index >= len(self.skel_joints):
                self.skel_joint_index = 0

        if imgui.tree_node("GUI Boxes"):
            if imgui.button("Add Cand Box"):
                self.cand_gui_box = Box("box", [0, 0, 0], [0, 0, 0], [0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0, 0, 0])
            if self.cand_gui_box:
                _, self.cand_gui_box.pos[:] = imgui.slider_float3("Position", *self.cand_gui_box.pos[:], -5.0, 5.0)
                changed, self.cand_gui_box.rot[:] = imgui.slider_float3("Rotation", *self.cand_gui_box.rot[:], -5.0, 5.0)
                if changed:
                    self.cand_gui_box.updateRot()
                _, self.cand_gui_box.size[:] = imgui.slider_float3("Size", *self.cand_gui_box.size[:], 0.001, 5.0)
                _, self.cand_gui_box.joint[:] = imgui.slider_float3("Joint", *self.cand_gui_box.joint[:], -5.0, 5.0)

            if imgui.button("Print Cand Box"):
                if self.cand_gui_box is not None:
                    print(self.cand_gui_box.name)
                    print(self.cand_gui_box.pos)
                else:
                    print("No Cand Box")
            
            if imgui.button("Delete Cand Box"):
                self.cand_gui_box = None

            if imgui.button("Add Cand to GUI Box"):
                self.gui_boxes.append(self.cand_gui_box)
                self.cand_gui_box = None

            imgui.tree_pop()

        # if imgui.tree_node("Find Farthest faces in SKEL sections"):
        #     SKEL_section_index = self.SKEL_section_index_male if self.skel_gender == "male" else self.SKEL_section_index_female
        #     if imgui.button("<##face"):
        #         self.skel_farthest_section_index -= 1
        #         if self.skel_farthest_section_index < 0:
        #             self.skel_farthest_section_index = len(SKEL_section_index) - 1
        #     imgui.same_line()
        #     imgui.text(list(SKEL_section_index.keys())[self.skel_farthest_section_index])
        #     imgui.same_line()
        #     if imgui.button(">##face"):
        #         self.skel_farthest_section_index += 1
        #         if self.skel_farthest_section_index >= len(SKEL_section_index):
        #             self.skel_farthest_section_index = 0

        #     if imgui.button("Find Farthest faces"):
        #         pairs = []
        #         for section_index in range(len(SKEL_section_index)):
        #             section_name = list(SKEL_section_index.keys())[section_index]
        #             if self.skel_gender == "male":
        #                 SKEL_section_faces = self.SKEL_section_faces_male[section_index]
        #             else:
        #                 SKEL_section_faces = self.SKEL_section_faces_female[section_index]
        #             SKEL_section_vertices = self.skel_vertices

        #             len_faces = SKEL_section_faces.shape[0] // 3

        #             max_d = 0
        #             cand_i = 0
        #             cand_j = 0
        #             for i in range(len_faces):
        #                 face_i = SKEL_section_vertices[SKEL_section_faces[3*i:3*i+3]]
        #                 mid_i = np.mean(face_i, axis=0)
        #                 for j in range(i + 1, len_faces):
        #                     face_j = SKEL_section_vertices[SKEL_section_faces[3*j:3*j+3]]
        #                     mid_j = np.mean(face_j, axis=0)
        #                     d = np.linalg.norm(mid_i - mid_j)
        #                     if d > max_d:
        #                         max_d = d
        #                         cand_i = i
        #                         cand_j = j
        #                         print(f"Candidates {i}, {j} with distance {d}")
        #                 print(f"face {i+1} / {len_faces} done")
        #             print(f"Farthest faces are {cand_i}, {cand_j} in section {section_name}")

        #             pairs.append([cand_i, cand_j])
                
        #         print(pairs)


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
        self.update_smpl()
        self.update_skel()

    def zero_reset(self):
        self.env.zero_reset()
        self.reward_buffer = [self.env.get_reward()]
        self.update_smpl()
        self.update_skel()

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
            elif key == glfw.KEY_A:
                self.skel_face_index -= 1
                if self.skel_face_index < 0:
                    self.skel_face_index = len(self.skel_faces) // 3 - 1
                i = self.skel_face_index - 1
                print(i, self.skel_faces[i*3:i*3+3])
            elif key in [glfw.KEY_D, glfw.KEY_F]:
                self.skel_face_index += 1
                if self.skel_face_index >= len(self.skel_faces) // 3:
                    self.skel_face_index = 0
                i = self.skel_face_index - 1
                print(i, self.skel_faces[i*3:i*3+3])
            elif key == glfw.KEY_G:
                self.skel_face_index += 10
                if self.skel_face_index >= len(self.skel_faces) // 3:
                    self.skel_face_index = 0
                i = self.skel_face_index - 1
                print(i, self.skel_faces[i*3:i*3+3])
            elif key == glfw.KEY_H:
                self.skel_face_index += 100
                if self.skel_face_index >= len(self.skel_faces) // 3:
                    self.skel_face_index = 0
                i = self.skel_face_index - 1
                print(i, self.skel_faces[i*3:i*3+3])

            elif key == glfw.KEY_X:
                self.skel_pose[0][37] -= 0.03
                self.update_skel()
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

