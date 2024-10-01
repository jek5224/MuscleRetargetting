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

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from imgui.integrations.glfw import GlfwRenderer
from viewer.TrackBall import TrackBall
from learning.ray_model import loading_network
from numba import jit
from core.env import Env
from core.dartHelper import buildFromInfo, exportSkeleton

import time

from models.smpl import SMPL
import torch

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

        self.draw_obj = True
        self.obj_trans = 0.5
        self.obj_axis = {}

        self.draw_target_motion = False
        self.draw_pd_target = False
        self.draw_zero_pose = False
        self.draw_test_skeleton = False
        self.test_dofs = None

        self.draw_body = False
        self.body_trans = 0.5
        self.draw_muscle = True
        self.draw_line_muscle = True
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

        imgui.create_context()
        self.window = impl_glfw_init(self.name, self.width, self.height)
        self.impl = GlfwRenderer(self.window)

        self.meshes = {}

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

    def update_smpl(self):
        ## Update SMPL Vertex to character's vertex
        ## Pos 0 : Sim Character, Pos 1 : Ref Character
        pos = torch.tensor(np.tile(np.zeros(3, dtype=np.float32), (2, 24, 1)))

        with torch.no_grad():
            res = self.smpl_model(body_pose = pos[:, 1:], global_orient=pos[:, 0].unsqueeze(1), betas = torch.tensor(np.array([self.shape_parameters,self.shape_parameters], dtype=np.float32)))
            self.smpl_zero_joints = res.smpl_joints[0] * self.smpl_scale

            self.smpl_colors = np.ones((2, len(self.smpl_vertices[0]), 4)) * 0.8
            self.smpl_colors[:, :, 3] = self.smpl_trans

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

        self.fitSkel2SMPL()
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

            self.meshes[bn.getName()].draw(np.array([color[0], color[1], color[2], self.obj_trans]))

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
            idx = 0
            for m_wps in self.env.muscle_pos:
                a = self.env.muscle_activation_levels[idx]
                if color is None:
                    glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
                glBegin(GL_LINE_STRIP)
                for wp in m_wps:
                    glVertex3f(wp[0], wp[1], wp[2])
                glEnd() 

                # glColor4d(1, 0, 0, 1)
                # for i in [0, -1]:
                #     glPushMatrix()
                #     glTranslatef(m_wps[i][0], m_wps[i][1], m_wps[i][2])
                #     mygl.draw_sphere(0.003, 10, 10)
                #     glPopMatrix()
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
        if self.draw_muscle:
            self.drawMuscles()
        if self.draw_obj:
            self.drawObj(self.env.skel.getPositions())
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

        # ## Draw Shawdow
        # glPushMatrix()
        # glScalef(1,1E-3,1)
        # color = np.array([0.0, 0.0, 0.0, 1.0])
        # glColor4f(color[0], color[1], color[2], color[3])
        # glEnableClientState(GL_VERTEX_ARRAY)
        # glBindBuffer(GL_ARRAY_BUFFER, 0)
        # glVertexPointer(3, GL_FLOAT, 0, self.smpl_vertex3[0])
        # glEnableClientState(GL_NORMAL_ARRAY)
        # glNormalPointer(GL_FLOAT, 0, self.smpl_normal[0])
        # glDrawArrays(GL_TRIANGLES, 0, len(self.smpl_vertex3[0]))
        # glDisableClientState(GL_NORMAL_ARRAY)
        # glDisableClientState(GL_VERTEX_ARRAY)
        # glPopMatrix()

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
                if imgui.button(f"Reset##beta{i}"):
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

        # if imgui.tree_node("Skeletons"):
        #     if imgui.button("Update Skeletons"):
        #         self.get_skeletons(self.data_path)

        #     clicked, self.skeleton_idx = imgui.listbox('', self.skeleton_idx, self.skeleton_files)

        #     if imgui.button("Load Skeleton"):
        #         pass

        #     imgui.tree_pop()

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

    def zero_reset(self):
        self.env.zero_reset()
        self.reward_buffer = [self.env.get_reward()]
        self.update_smpl()

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

