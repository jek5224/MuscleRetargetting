#pip install imgui[glfw]
import imgui
import glfw
import numpy as np
import trimesh
import copy
import os
os.environ['DISABLE_VIEWER']='1'
from scipy.spatial.transform import Rotation as R
import dartpy as dart
import viewer.gl_function as mygl
import quaternion
from PIL import Image
from viewer.mesh_loader import MeshLoader
from viewer.arap_backends import get_backend, check_gpu_available, check_taichi_available
from viewer.contour_mesh import find_corner_indices_ray_based
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
from core.bvhparser import MyBVH
import glob
from skeleton_section import SKEL_dart_info
from viewer.zygote_mesh_ui import (draw_zygote_ui,
    _render_inspect_2d_windows, _render_neck_viz_windows,
    _render_manual_cut_windows, _render_level_select_windows,
    update_available_muscles, load_previous_muscles, save_loaded_muscles,
    draw_inter_muscle_constraint_lines, drawMuscles, drawTestMuscles,
    reset, zero_reset, _scan_motion_files,
    _motion_step_forward, _motion_bake_step)

import time

# ============================================================================
# CONSTANTS
# ============================================================================

# Window defaults
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
DEFAULT_PERSPECTIVE = 45.0

# Camera defaults
CAMERA_ZOOM_FACTOR = 1.05
CAMERA_INITIAL_DISTANCE = 10  # power of CAMERA_ZOOM_FACTOR
MIN_EYE_DISTANCE = 0.5

# Paths
ZYGOTE_MESH_DIR = 'Zygote_Meshes_251229/'
RESULT_PATH = './ray_results'
DATA_PATH = './data'

# Rendering defaults
DEFAULT_BODY_TRANSPARENCY = 0.5
DEFAULT_OBJ_TRANSPARENCY = 1.0
DEFAULT_LINE_WIDTH = 2
ACTIVATION_PLOT_SCALE = 80

# Default colors
MUSCLE_COLOR = np.array([0.75, 0.25, 0.25])
SKELETON_COLOR = np.array([0.9, 0.9, 0.9])

# UI dimensions
wide_button_width = 308
wide_button_height = 50
button_width = 150
push_width = 150

# ============================================================================
# Light Options
# ============================================================================
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

def _draw_tet_meshes_batched(app):
    """Batch all visible tet mesh surfaces into minimal GL draw calls."""
    surface_arrays = []
    normal_arrays = []
    color_arrays = []
    cap_arrays = []
    cap_normal_arrays = []
    cap_color_arrays = []
    edge_muscles = []

    for name, obj in app.zygote_muscle_meshes.items():
        viper_only = obj.viper_sim is not None and obj.viper_only_mode
        if viper_only:
            continue


        if not (obj.is_draw_tet_mesh or getattr(obj, '_tet_anim_active', False)):
            continue

        # Skip phase 0 (tet not visible yet)
        if getattr(obj, '_tet_anim_active', False) and obj._tet_anim_phase == 0:
            continue

        # Ensure arrays are prepared
        if not hasattr(obj, '_tet_surface_verts') or obj._tet_surface_verts is None:
            if not hasattr(obj, 'tet_vertices') or obj.tet_vertices is None:
                continue
            obj._prepare_tet_draw_arrays()

        # Determine alpha
        if not getattr(obj, '_tet_anim_active', False):
            alpha = app.zygote_tet_transparency
        else:
            alpha = obj.contour_mesh_transparency
            if obj._tet_anim_phase == 1:
                alpha = getattr(obj, '_tet_anim_tet_alpha', alpha)

        color = obj.contour_mesh_color

        # Surface faces
        if obj._tet_surface_verts is not None and len(obj._tet_surface_verts) > 0:
            surface_arrays.append(obj._tet_surface_verts)
            normal_arrays.append(obj._tet_surface_normals)

            heatmap_colors = getattr(obj, '_tet_surface_colors', None)
            if heatmap_colors is not None and len(heatmap_colors) == len(obj._tet_surface_verts):
                # Override baked alpha with current slider value
                if heatmap_colors[0, 3] != alpha:
                    heatmap_colors[:, 3] = alpha
                color_arrays.append(heatmap_colors)
            else:
                n = len(obj._tet_surface_verts)
                flat = np.empty((n, 4), dtype=np.float32)
                flat[:, 0] = color[0]
                flat[:, 1] = color[1]
                flat[:, 2] = color[2]
                flat[:, 3] = alpha
                color_arrays.append(flat)

        # Cap faces
        if obj._tet_cap_verts is not None and len(obj._tet_cap_verts) > 0:
            cap_arrays.append(obj._tet_cap_verts)
            cap_normal_arrays.append(obj._tet_cap_normals)
            n = len(obj._tet_cap_verts)
            cap_col = np.empty((n, 4), dtype=np.float32)
            cap_col[:, 0] = 0.2
            cap_col[:, 1] = 0.6
            cap_col[:, 2] = 0.2
            cap_col[:, 3] = alpha
            cap_color_arrays.append(cap_col)

        # Edges (rare, keep per-muscle)
        if obj.is_draw_tet_edges:
            edge_muscles.append(obj)

    # Batched surface draw
    if surface_arrays:
        all_verts = np.concatenate(surface_arrays)
        all_normals = np.concatenate(normal_arrays)
        all_colors = np.concatenate(color_arrays)

        glPushMatrix()
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, all_verts)
        glNormalPointer(GL_FLOAT, 0, all_normals)
        glColorPointer(4, GL_FLOAT, 0, all_colors)
        glDrawArrays(GL_TRIANGLES, 0, len(all_verts))

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

    # Batched cap draw
    if cap_arrays:
        all_cap_verts = np.concatenate(cap_arrays)
        all_cap_normals = np.concatenate(cap_normal_arrays)
        all_cap_colors = np.concatenate(cap_color_arrays)

        glPushMatrix()
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, all_cap_verts)
        glNormalPointer(GL_FLOAT, 0, all_cap_normals)
        glColorPointer(4, GL_FLOAT, 0, all_cap_colors)
        glDrawArrays(GL_TRIANGLES, 0, len(all_cap_verts))

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()

    # Edge drawing fallback (per-muscle, rarely enabled)
    for obj in edge_muscles:
        if obj._tet_edge_verts is not None and len(obj._tet_edge_verts) > 0:
            glPushMatrix()
            glDisable(GL_LIGHTING)
            glEnableClientState(GL_VERTEX_ARRAY)
            glColor4f(0.1, 0.1, 0.1, 0.8)
            glLineWidth(1.0)
            glVertexPointer(3, GL_FLOAT, 0, obj._tet_edge_verts)
            glDrawArrays(GL_LINES, 0, len(obj._tet_edge_verts))
            glDisableClientState(GL_VERTEX_ARRAY)
            glEnable(GL_LIGHTING)
            glPopMatrix()


def initGL():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    # glClearColor(0.0, 0.0, 0.0, 1.0)
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
def impl_glfw_init(window_name="Muscle Simulation", width=DEFAULT_WINDOW_WIDTH, height=DEFAULT_WINDOW_HEIGHT):
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
        self.width = DEFAULT_WINDOW_WIDTH
        self.height = DEFAULT_WINDOW_HEIGHT
        
        ## Camera Setting
        self.perspective = DEFAULT_PERSPECTIVE
        self.trackball = TrackBall()
        self.eye = np.array([0.0, 0.0, 1.0]) * np.power(CAMERA_ZOOM_FACTOR, CAMERA_INITIAL_DISTANCE)
        self.up = np.array([0.0, 1.0, 0.0])
        self.trans = np.array([0.0, 0.0, 0.0])
        self.zoom = 1.0

        self.trackball.set_trackball(np.array([self.width * 0.5, self.height * 0.5]), self.width * 0.5)
        self.trackball.set_quaternion(np.quaternion(1.0, 0.0, 0.0, 0.0))
        
        ## Camera transform flag
        self.mouse_down = False
        self.rotate = False
        self.translate = False

        ## Auto-rotate around focused muscle
        self.auto_rotate = False
        self.auto_rotate_speed = 0.5  # radians per second
        self._auto_rotate_paused = False

        self.mouse_x = 0
        self.mouse_y = 0
        self.motion_skel = None

        # Motion Browser state
        self.motion_bvh_files = []  # list of .bvh paths found in data/motion/
        self.motion_selected_idx = -1  # index into motion_bvh_files (-1 = none)
        self.motion_bvh = None  # loaded MyBVH instance
        self.motion_current_frame = 0
        self.motion_total_frames = 0
        self.motion_is_playing = False
        self.motion_play_speed = 0.25
        self.motion_run_tet_sim = False
        self.motion_settle_iters = 50
        self.motion_play_accumulator = 0.0
        self.motion_repeat = False
        self.motion_deform_cache = {}     # Dict: muscle_name -> {frame_idx: positions_array}
        self.motion_baking = False        # True during batch bake
        self.motion_bake_end_frame = 0    # Target end frame for batch bake
        self.motion_bake_current = 0      # Current progress during bake
        self._bake_data = {}              # Accumulated bake results
        self.motion_use_nn = False           # Toggle: use NN inference instead of cache
        self.motion_nn_error_heatmap = False  # Toggle: color muscles by NN vs GT error
        self.motion_fix_x = False            # Fix root X translation at rest
        self.motion_fix_y = False            # Fix root Y translation at rest
        self.motion_fix_z = False            # Fix root Z translation at rest
        self.motion_fix_rotation = False     # Fix root rotation at rest
        self.motion_nn_model = None          # Loaded DistillNet model
        self.motion_nn_rest_positions = None # Rest positions dict for NN inference
        self.motion_nn_checkpoint_path = None # Path to loaded checkpoint
        _scan_motion_files(self)

        ## Flag
        self.is_simulation = False

        self.draw_obj = False
        self.obj_trans = 1.0
        self.obj_axis = {}

        self.draw_target_motion = False
        self.draw_pd_target = False
        self.draw_test_skeleton = False
        self.test_dofs = None

        self.draw_body = False
        self.draw_muscle = True
        self.body_trans = DEFAULT_BODY_TRANSPARENCY
        self.draw_line_muscle = True
        self.muscle_index = 0
        self.line_width = DEFAULT_LINE_WIDTH
        self.draw_bone = False
        self.draw_joint = False
        self.draw_shadow = False

        # Joint position editor
        self.joint_edit_mode = False
        self.joint_edit_selected = None   # name of selected joint's body node
        self.joint_edit_symmetry = True
        self.joint_edit_dragging = False
        self.joint_edit_drag_depth = 0.0  # depth of dragged joint for screen-parallel plane
        
        self.reset_value = 0

        self.is_screenshot = False
        self.imagenum = 0

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
        self.skel_scale = 1

        self.skel_muscle_activation_levels = None
        self.test_skel = None
        self.test_skel_dofs = None
        self.draw_skel_dofs = False

        # OBJ Files
        self.meshes = {}

        self.zygote_muscle_meshes = {}
        self.zygote_muscle_color = MUSCLE_COLOR.copy()
        self.zygote_muscle_transparency = DEFAULT_OBJ_TRANSPARENCY
        self.zygote_tet_transparency = DEFAULT_OBJ_TRANSPARENCY
        self.zygote_fiber_transparency = 1.0
        self.is_draw_zygote_muscle = True
        self.is_draw_zygote_muscle_tet = True
        self.is_draw_zygote_muscle_fibers = True
        self.is_draw_zygote_muscle_open_edges = False
        self.is_draw_one_zygote_muscle = False
        self.zygote_muscle_dir = 'Zygote_Meshes_251229/Muscle/'
        self.available_muscle_files = []  # List of (name, path) tuples for muscles not loaded
        self.available_muscle_by_category = {}  # Dict: category -> list of (name, path)
        self.available_category_expanded = {}  # Dict: category -> bool (expanded state)
        self.available_muscle_selected = 0  # Selected index in available list
        self.available_selected_category = None  # Currently selected category
        self.available_selected_muscle = None  # Currently selected muscle name
        self.loaded_muscle_selected = 0  # Selected index in loaded list
        self.last_muscles_file = '.last_loaded_muscles.json'  # File to remember loaded muscles
        self.zygote_skeleton_meshes = {}
        self.zygote_skeleton_color = SKELETON_COLOR.copy()
        self.zygote_skeleton_transparency = DEFAULT_OBJ_TRANSPARENCY
        self.is_draw_zygote_skeleton = True
        self.is_draw_one_zygote_skeleton = False
        self.zygote_muscle_meshes_intersection_bones = {}

        # Inter-muscle distance constraints
        # List of (muscle1_name, v1_idx, v1_fixed, muscle2_name, v2_idx, v2_fixed, rest_distance)
        self.inter_muscle_constraints = []
        self.inter_muscle_constraint_threshold = 0.015  # 15mm default threshold
        self.coupled_as_unified_volume = True  # Treat all muscles as one unified system
        self.use_muscle_aware_arap = True  # Fiber-direction + volume-preserving ARAP

        # 2D Inspect window state
        self.inspect_2d_open = {}  # Dict: muscle_name -> bool (window open state)
        self.inspect_2d_stream_idx = {}  # Dict: muscle_name -> selected stream index
        self.inspect_2d_contour_idx = {}  # Dict: muscle_name -> selected contour index

        # Correspondence mode state for Inspect 2D
        self.inspect_2d_corr_mode = {}  # Dict: muscle_name -> bool (correspondence mode active)
        self.inspect_2d_corr_corner = {}  # Dict: muscle_name -> selected corner index (0-3) or -1
        self.inspect_2d_corr_vertex = {}  # Dict: muscle_name -> selected vertex index or -1

        # Display rotation for Inspect 2D right side (visual only, doesn't change data)
        self.inspect_2d_display_rot = {}  # Dict: (muscle_name, level_idx) -> int (0-3, number of 90° CCW rotations)

        # Correspondence hover preview: stores original contour_match to restore when hover ends
        self.inspect_2d_corr_backup_cm = {}  # Dict: muscle_name -> original contour_match list
        self.inspect_2d_corr_backup_wp = {}  # Dict: muscle_name -> original waypoints for preview level
        self.inspect_2d_corr_backup_mvc = {}  # Dict: muscle_name -> original mvc_weights for preview level
        self.inspect_2d_corr_preview_active = {}  # Dict: muscle_name -> bool

        # Edit fiber mode state for Inspect 2D
        self.inspect_2d_edit_fiber_mode = {}  # Dict: muscle_name -> bool (edit fiber mode active)
        self.inspect_2d_edit_fiber_selected = {}  # Dict: muscle_name -> selected fiber index or -1
        self.inspect_2d_edit_fiber_preview = {}  # Dict: muscle_name -> preview position (u, v) or None
        self.inspect_2d_edit_fiber_test = {}  # Dict: muscle_name -> test waypoints list or None

        # GPU acceleration settings
        self.use_gpu_arap = False  # Use GPU (PyTorch) for ARAP solver
        self.use_taichi_arap = True  # Use Taichi for ARAP solver (default)
        self.gpu_available = check_gpu_available()
        self.taichi_available = check_taichi_available()
        self.arap_backend = None  # Will be created when needed

        # FEM simulation settings
        self.use_fem_sim = False  # Use FEM instead of ARAP
        self.fem_youngs_modulus = 5000.0  # Young's modulus (Pa)
        self.fem_poisson_ratio = 0.49  # Poisson's ratio
        self.fem_collision_kappa = 1e4  # Collision penalty stiffness
        self.fem_volume_penalty = 100.0  # Volume preservation penalty
        self.fem_contact_threshold = 0.015  # Inter-muscle contact distance (m)
        self.fem_outer_iterations = 3  # Outer iterations for inter-muscle convergence

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
            # Always pass scroll to imgui for window scrolling
            io = imgui.get_io()
            io.mouse_wheel = yoffset
            io.mouse_wheel_horizontal = xoffset
            # Only handle in viewer if imgui doesn't want it
            if not io.want_capture_mouse:
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

    def setEnv(self, env):  
        self.env = env
        self.motion_skel = self.env.skel.clone()
        self.test_dofs = np.zeros(self.motion_skel.getNumDofs())

        for k, info in self.env.mesh_info.items():
            self.meshes[k] = MeshLoader()
            self.meshes[k].load(info)
            self.meshes[k].use_two_pass_culling = False  # OBJ meshes may have inconsistent winding

        self.env.meshes = self.meshes

        # Load skeleton meshes from Zygote directory
        zygote_dir = 'Zygote_Meshes_251229/'
        for skeleton_file in os.listdir(zygote_dir + 'Skeleton'):
            if skeleton_file.endswith('.obj'):
                skeleton_name = skeleton_file.split('.')[0]

                self.zygote_skeleton_meshes[skeleton_name] = MeshLoader()
                self.zygote_skeleton_meshes[skeleton_name].load(zygote_dir + 'Skeleton/' + skeleton_file)
                self.zygote_skeleton_meshes[skeleton_name].color = np.array([0.9, 0.9, 0.9])
                # Load trimesh and apply same scale as MeshLoader.load() uses
                skel_trimesh = trimesh.load_mesh(zygote_dir + 'Skeleton/' + skeleton_file)
                skel_trimesh.vertices *= 0.01  # MESH_SCALE
                self.zygote_skeleton_meshes[skeleton_name].trimesh = skel_trimesh

        # Sort skeleton meshes by name
        self.zygote_skeleton_meshes = dict(sorted(self.zygote_skeleton_meshes.items()))
        for i, (name, mesh) in enumerate(self.zygote_skeleton_meshes.items()):
            mesh.cand_parent_index = i

        # Auto-load previously loaded muscles
        load_previous_muscles(self)
        self.zygote_muscle_meshes = dict(sorted(self.zygote_muscle_meshes.items()))

        # Update available muscles list
        update_available_muscles(self)

        # reset(self, self.reset_value)
        zero_reset(self)

        # Mesh Setting
        for bn in self.env.skel.getBodyNodes():  # bn = body node
            transform = bn.getWorldTransform().matrix() @ bn.getParentJoint().getTransformFromChildBodyNode().matrix()
            t_parent = transform[:3, 3]

            if bn.getName() in self.meshes.keys():
                if len(self.meshes[bn.getName()].vertices_3) > 0:
                    self.meshes[bn.getName()].vertices_3 -= t_parent
                    self.meshes[bn.getName()].new_vertices_3 -= t_parent
                if len(self.meshes[bn.getName()].vertices_4) > 0:
                    self.meshes[bn.getName()].vertices_4 -= t_parent
    
    def loadNetwork(self, path):
        self.nn, mus_nn, env_str = loading_network(path)
        # if env_str != None:
        #     self.setEnv(Env(env_str))   
        self.env.muscle_nn = mus_nn

    def get_ray_from_cursor(self):
        projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        x = self.mouse_x
        y = self.height - self.mouse_y
        near = gluUnProject(x, y, 0.0, modelview_matrix, projection_matrix, viewport)
        far = gluUnProject(x, y, 1.0, modelview_matrix, projection_matrix, viewport)
        ray_origin = np.array(near)
        ray_direction = np.array(far) - ray_origin
        ray_direction /= np.linalg.norm(ray_direction)
        return ray_origin, ray_direction

    def _get_joint_world_positions(self):
        """Return dict of {body_node_name: world_pos} for all joints."""
        skel = self.env.skel
        positions = {}
        for i in range(skel.getNumBodyNodes()):
            bn = skel.getBodyNode(i)
            j = bn.getParentJoint()
            transform = bn.getWorldTransform().matrix() @ j.getTransformFromChildBodyNode().matrix()
            positions[bn.getName()] = transform[:3, 3]
        return positions

    def pick_joint(self, ray_origin, ray_direction, threshold=0.02):
        """Find closest joint to ray. Returns body node name or None."""
        joint_positions = self._get_joint_world_positions()
        best_name = None
        best_dist = threshold
        for name, pos in joint_positions.items():
            # Point-to-ray distance
            v = pos - ray_origin
            t = np.dot(v, ray_direction)
            if t < 0:
                continue
            closest = ray_origin + t * ray_direction
            dist = np.linalg.norm(pos - closest)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    @staticmethod
    def _get_mirror_name(name):
        """Return L/R mirror counterpart name, or None."""
        if name.startswith("L_"):
            mirror = "R_" + name[2:]
        elif name.startswith("R_"):
            mirror = "L_" + name[2:]
        else:
            return None
        return mirror

    def _drag_joint(self, dx, dy):
        """Move selected joint on a screen-parallel plane based on mouse delta."""
        name = self.joint_edit_selected
        info = self.env.new_skel_info.get(name)
        if info is None:
            return

        # Convert pixel delta to world-space delta at the joint's depth
        # Use the inverse projection to get world units per pixel
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)

        # Unproject two points at the joint depth to get scale
        viewport = glGetIntegerv(GL_VIEWPORT)
        depth_z = self.joint_edit_drag_depth
        # Map depth to window z: get approximate ndc
        # Use gluUnProject at two nearby screen positions
        sx, sy = self.mouse_x, self.height - self.mouse_y
        p1 = np.array(gluUnProject(sx, sy, 0.5, modelview, projection, viewport))
        p2 = np.array(gluUnProject(sx + 1, sy, 0.5, modelview, projection, viewport))
        p3 = np.array(gluUnProject(sx, sy + 1, 0.5, modelview, projection, viewport))

        # Get right and up vectors in world space
        right = p2 - p1
        up = p3 - p1

        # Scale by depth ratio (further = larger movement)
        # The 0.5 z is mid-frustum; scale by actual depth
        proj_near = 0.1
        proj_far = 100.0
        mid_depth = (proj_near + proj_far) / 2.0
        actual_depth = abs(depth_z)
        if mid_depth > 0:
            scale = actual_depth / mid_depth
        else:
            scale = 1.0

        delta = right * dx * scale + up * dy * scale
        info['joint_t'] = info['joint_t'].astype(np.float64) + delta

        # Mirror if symmetry enabled
        if self.joint_edit_symmetry:
            mirror_name = self._get_mirror_name(name)
            if mirror_name and mirror_name in self.env.new_skel_info:
                mirror_info = self.env.new_skel_info[mirror_name]
                mirror_info['joint_t'] = info['joint_t'].copy()
                mirror_info['joint_t'][0] = -mirror_info['joint_t'][0]

        self.newSkeleton()

    ## mousce button callback function
    def mousePress(self, button, action, mods):
        if action == glfw.PRESS:
            self.mouse_down = True
            if button == glfw.MOUSE_BUTTON_LEFT:
                # Joint editor: pick instead of rotate
                if self.joint_edit_mode and hasattr(self.env, 'new_skel_info'):
                    ray_origin, ray_direction = self.get_ray_from_cursor()
                    hit = self.pick_joint(ray_origin, ray_direction)
                    if hit is not None:
                        self.joint_edit_selected = hit
                        mirror = self._get_mirror_name(hit)
                        self.joint_edit_symmetry = mirror is not None and mirror in self.env.new_skel_info
                        self.joint_edit_dragging = True
                        # Store depth for screen-parallel dragging
                        joint_positions = self._get_joint_world_positions()
                        joint_pos = joint_positions[hit]
                        # Project joint position to get depth
                        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                        # depth = dot(modelview * joint_pos)
                        p = modelview @ np.append(joint_pos, 1.0)
                        self.joint_edit_drag_depth = p[2]
                        return
                    else:
                        self.joint_edit_selected = None
                if self.auto_rotate:
                    self._auto_rotate_paused = True
                self.rotate = True
                self.trackball.start_ball(self.mouse_x, self.height - self.mouse_y)
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.translate = True
        elif action == glfw.RELEASE:
            self.mouse_down = False
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.joint_edit_dragging = False
                if self.auto_rotate:
                    self._auto_rotate_paused = False
                self.rotate = False
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.translate = False

    ## mouse move callback function
    def mouseMove(self, xpos, ypos):
        dx = xpos - self.mouse_x
        dy = ypos - self.mouse_y

        self.mouse_x = xpos
        self.mouse_y = ypos

        # Joint editor dragging
        if self.joint_edit_dragging and self.joint_edit_selected is not None:
            self._drag_joint(dx, dy)
            return

        if self.rotate:
            if dx != 0 or dy != 0:
                self.trackball.update_ball(xpos, self.height - ypos)

        if self.translate:
            rot = quaternion.as_rotation_matrix(self.trackball.curr_quat)
            self.trans += (1.0 / self.zoom) * rot.transpose() @ np.array([dx, -dy, 0.0])
        
    ## mouse scroll callback function
    def mouseScroll(self, xoffset, yoffset):
        if yoffset < 0:
            self.eye *= CAMERA_ZOOM_FACTOR
        elif (yoffset > 0) and (np.linalg.norm(self.eye) > MIN_EYE_DISTANCE):
            self.eye /= CAMERA_ZOOM_FACTOR
    
    def update(self):
        if self.nn is not None:
            obs = self.env.get_obs()
            action = self.nn.get_action(obs)
            try:
                _, _, done, _ = self.env.step(action)
            except Exception:
                _, _, done, _ = self.env.step(np.zeros(self.env.num_action))
        else:
            _, _, done, _ = self.env.step(np.zeros(self.env.num_action), skel_action=self.skel_muscle_activation_levels)
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
        mygl.draw_cube(shape.getSize())
    
    def drawObj(self, pos, color = np.array([0.8, 0.8, 0.8, 0.5])):
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

            # check if bn.getName() in self.meshes.keys()
            if bn.getName() in self.meshes.keys():
                self.meshes[bn.getName()].draw_simple(np.array([color[0], color[1], color[2], self.obj_trans]))

            glPopMatrix()

        glPopMatrix()


    def drawSkeleton(self, pos, color = np.array([0.5, 0.5, 0.5, 0.5])):
        self.motion_skel.setPositions(pos)

        for bn in self.motion_skel.getBodyNodes():
            glPushMatrix()
            # self.meshes[bn.getName()].draw()
            glMultMatrixd(bn.getWorldTransform().matrix().transpose())
            
            for sn in bn.getShapeNodes():  # sn = shape node
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

            glPopMatrix()

    def _draw_joint_editor_overlay(self):
        """Draw larger colored spheres for all joints in edit mode."""
        joint_positions = self._get_joint_world_positions()
        for name, pos in joint_positions.items():
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            if name == self.joint_edit_selected:
                glColor4d(0.0, 1.0, 0.0, 0.9)  # green = selected
                mygl.draw_sphere(0.015, 12, 12)
            else:
                glColor4d(0.0, 0.5, 1.0, 0.7)  # blue = default (matches existing joint style)
                mygl.draw_sphere(0.012, 10, 10)
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

        # mygl.drawGround(-1E-3)

        if self.mouse_down:
            glLineWidth(1.5)
            mygl.draw_axis()

        # Draw muscle mesh parts (contours, bounding boxes, edges — NOT the mesh itself)
        for name, obj in self.zygote_muscle_meshes.items():
            viper_only = obj.viper_sim is not None and obj.viper_only_mode
            if obj.viper_sim is not None and obj.is_draw_viper:
                obj.draw_viper()
            if obj.viper_sim is not None and getattr(obj, 'is_draw_viper_rod_mesh', False):
                obj.draw_viper_mesh()
            if not viper_only:
                if obj.is_draw_contours:
                    obj.draw_contours()
                if obj.is_draw_open_edges:
                    obj.draw_open_edges([0.0, 0.0, 1.0, obj.transparency])
                if obj.is_draw_centroid:
                    obj.draw_centroid()
                if obj.is_draw_bounding_box:
                    obj.draw_bounding_box()
                if obj.is_draw_edges:
                    obj.draw_edges()
                if obj.is_draw_constraints:
                    obj.draw_constraints()

        # Draw inter-muscle constraints if enabled
        if getattr(self, 'draw_inter_muscle_constraints', False):
            draw_inter_muscle_constraint_lines(self)

        # Draw order: DART skeleton -> fiber structure -> tet mesh
        # GL_BLEND is enabled globally, just use alpha in glColor4f

        # Draw DART skeleton first
        if self.draw_target_motion:
            self.drawSkeleton(self.env.target_pos, np.array([1.0, 0.3, 0.3, 0.5]))
        if self.draw_bone:
            self.drawBone(self.env.skel.getPositions())
        if self.draw_joint:
            self.drawJoint(self.env.skel.getPositions())
        if self.draw_obj:
            self.drawObj(self.env.skel.getPositions())
        if self.draw_muscle:
            drawMuscles(self)
        if self.draw_body:
            self.drawSkeleton(self.env.skel.getPositions(), np.array([0.5, 0.5, 0.5, self.body_trans]))

        # Draw zygote skeleton meshes (before muscle mesh so they're visible through transparency)
        for name, obj in self.zygote_skeleton_meshes.items():
            if obj.is_draw:
                obj.draw([obj.color[0], obj.color[1], obj.color[2], obj.transparency])
            if obj.is_draw_corners:
                obj.draw_corners()
            if obj.is_draw_edges:
                obj.draw_edges()

        # Draw joint editor overlay on top of skeleton (disable depth test so joints show through)
        if self.joint_edit_mode:
            glDisable(GL_DEPTH_TEST)
            self._draw_joint_editor_overlay()
            glEnable(GL_DEPTH_TEST)

        # Draw fiber structure
        for name, obj in self.zygote_muscle_meshes.items():
            viper_only = obj.viper_sim is not None and obj.viper_only_mode
            if not viper_only and obj.is_draw_fiber_architecture:
                obj.fiber_transparency = self.zygote_fiber_transparency
                obj.draw_fiber_architecture()

        # Draw tet mesh (before contour mesh so contour mesh fades on top during animation)
        _draw_tet_meshes_batched(self)

        # Draw contour mesh (after fibers and tet so it fades cleanly on top)
        for name, obj in self.zygote_muscle_meshes.items():
            viper_only = obj.viper_sim is not None and obj.viper_only_mode
            if not viper_only and (obj.is_draw_contour_mesh or getattr(obj, '_mesh_anim_active', False)):
                obj.draw_contour_mesh()

        # Draw muscle mesh (last so transparency shows skeleton and fibers through it)
        for name, obj in self.zygote_muscle_meshes.items():
            viper_only = obj.viper_sim is not None and obj.viper_only_mode
            if not viper_only and obj.is_draw:
                obj.draw([obj.color[0], obj.color[1], obj.color[2], obj.transparency])

        # if self.draw_pd_target:
        #     self.drawSkeleton(self.env.pd_target, np.array([0.3, 0.3, 1.0, 0.5]))
        
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
                drawMuscles(self, shadow_color)
            
            if self.draw_target_motion:
                self.drawSkeleton(self.env.target_pos, np.array([1.0, 0.3, 0.3, 0.2]))
            if self.draw_body:
                self.drawSkeleton(self.env.skel.getPositions(), shadow_color)
            if self.draw_pd_target:
                self.drawSkeleton(self.env.pd_target, np.array([0.3, 0.3, 1.0, 0.2]))

            glPopMatrix()

    def newSkeleton(self):
        # current_pos = self.env.skel.getPositions()
        self.env.world.removeSkeleton(self.env.skel)
        self.env.skel = buildFromInfo(self.env.new_skel_info, self.env.root_name)
        self.env.target_skel = self.env.skel.clone()
        self.env.world.addSkeleton(self.env.skel)

        if self.env.new_muscle_info is not None:
            self.env.loading_muscle_info(self.env.new_muscle_info)
            self.env.loading_test_muscle_info(self.env.new_muscle_info)

        self.motion_skel = self.env.skel.clone()
        # self.motion_skel.setPositions(current_pos)
        self.motion_skel.setPositions(self.env.skel.getPositions())

        # reset(self, self.env.world.getTime())
        zero_reset(self)

    # ========================================================================
    # Motion Browser Methods
    # ========================================================================

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


        draw_zygote_ui(self)


        if imgui.tree_node("Rendering Mode"):
            # imgui.checkbox("Draw Mesh", self.draw_mesh)
            
            _, self.draw_target_motion = imgui.checkbox("Draw Target Motion", self.draw_target_motion)
            _, self.draw_pd_target = imgui.checkbox("Draw PD Target", self.draw_pd_target)
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

                if self.env.muscle_info is not None:
                    if imgui.button("<##muscle"):
                        self.muscle_index -= 1
                        if self.muscle_index < 0:
                            self.muscle_index = len(self.env.muscle_pos) - 1
                    imgui.same_line()
                    if imgui.button(">##muscle"):
                        self.muscle_index += 1
                        if self.muscle_index >= len(self.env.muscle_pos):
                            self.muscle_index = 0
                    imgui.same_line()
                    muscle_name = list(self.env.muscle_info.keys())[self.muscle_index]
                    imgui.text("%3d: %s" % (self.muscle_index, muscle_name))
                    
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

            imgui.tree_pop()
            
        changed, self.reset_value = imgui.slider_float("Reset Time", self.reset_value, 0.0, self.env.bvhs[self.env.bvh_idx].bvh_time)
        if changed:
            reset(self, self.reset_value)
        if imgui.button("Reset"):
            reset(self, self.reset_value)
        imgui.same_line()
        if imgui.button("Random Reset"):
            self.reset_value = np.random.random() * self.env.bvhs[self.env.bvh_idx].bvh_time
            reset(self, self.reset_value)
        imgui.same_line()
        if imgui.button("Zero Reset"):
            zero_reset(self)

        _, self.is_screenshot = imgui.checkbox("Take Screenshots", self.is_screenshot)

        if imgui.tree_node("Test rotvecs"):
            for i in range(self.motion_skel.getNumDofs()):
                imgui.push_item_width(push_width)
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
                # Bounds check for activation levels
                if idx < len(self.env.muscle_activation_levels):
                    a = self.env.muscle_activation_levels[idx]
                else:
                    a = 0.0
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

        # Render Inspect 2D windows for each muscle
        _render_inspect_2d_windows(self)

        # Render Neck Viz windows (narrowest neck visualization)
        _render_neck_viz_windows(self)

        # Render Manual Cut windows for each muscle
        _render_manual_cut_windows(self)

        # Render Level Select windows for manual level selection
        _render_level_select_windows(self)

        imgui.render()

    def keyboardPress(self, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.window, True)
            elif key == glfw.KEY_SPACE:
                self.is_simulation = not self.is_simulation
            elif key == glfw.KEY_S:
                self.update()
            elif key == glfw.KEY_R:
                reset(self, self.reset_value)
            elif key == glfw.KEY_Z:
                zero_reset(self)

    def startLoop(self):        
        while not glfw.window_should_close(self.window):
            start_time = time.time()
            glfw.poll_events()
            
            self.impl.process_inputs()

            # Motion Browser: auto-advance in play mode
            if self.motion_is_playing and self.motion_bvh is not None:
                dt = start_time - getattr(self, '_motion_last_time', start_time)
                if dt <= 0:
                    dt = 1.0 / 30.0
                self.motion_play_accumulator += dt * self.motion_play_speed
                frame_time = self.motion_bvh.frame_time
                if self.motion_play_accumulator >= frame_time:
                    # Skip to final frame directly — only apply pose+deformation once
                    frames_to_skip = int(self.motion_play_accumulator / frame_time)
                    self.motion_play_accumulator -= frames_to_skip * frame_time
                    _motion_step_forward(self, frames_to_skip)
            self._motion_last_time = start_time

            # Motion Browser: bake one frame per render loop
            if self.motion_baking:
                _motion_bake_step(self)

            # Process step animations (replay)
            for name, obj in self.zygote_muscle_meshes.items():
                if getattr(obj, '_scalar_anim_active', False):
                    obj.update_scalar_animation(1.0 / 30.0)
                if getattr(obj, '_contour_anim_active', False):
                    obj.update_contour_animation(1.0 / 30.0)
                if getattr(obj, '_fill_gaps_anim_active', False):
                    obj.update_fill_gaps_animation(1.0 / 30.0)
                if getattr(obj, '_transitions_anim_active', False):
                    obj.update_transitions_animation(1.0 / 30.0)
                if getattr(obj, '_smooth_anim_active', False):
                    obj.update_smooth_animation(1.0 / 30.0)
                if getattr(obj, '_cut_anim_active', False):
                    obj.update_cut_animation(1.0 / 30.0)
                if getattr(obj, '_stream_smooth_anim_active', False):
                    obj.update_stream_smooth_animation(1.0 / 30.0)
                if getattr(obj, '_level_select_anim_active', False):
                    obj.update_level_select_animation(1.0 / 30.0)
                if getattr(obj, '_fiber_anim_active', False):
                    obj.update_fiber_animation(1.0 / 30.0)
                if getattr(obj, '_resample_anim_active', False):
                    obj.update_resample_animation(1.0 / 30.0)
                if getattr(obj, '_mesh_anim_active', False):
                    obj.update_mesh_animation(1.0 / 30.0)
                if getattr(obj, '_tet_anim_active', False):
                    if not obj.update_tet_animation(1.0 / 30.0):
                        self.zygote_tet_transparency = obj.contour_mesh_transparency

                # Play All sequencing
                if getattr(obj, '_play_all_active', False):
                    any_active = any(getattr(obj, f, False) for f in [
                        '_scalar_anim_active', '_contour_anim_active', '_fill_gaps_anim_active',
                        '_transitions_anim_active', '_smooth_anim_active', '_cut_anim_active',
                        '_stream_smooth_anim_active', '_level_select_anim_active',
                        '_fiber_anim_active', '_resample_anim_active', '_mesh_anim_active',
                        '_tet_anim_active'])
                    if not any_active:
                        _play_all_steps = [
                            (lambda o: getattr(o, '_scalar_anim_target_colors', None) is not None,
                             lambda o: o.replay_scalar_animation(), '_scalar_replayed'),
                            (lambda o: o.contours is not None and len(o.contours) > 0,
                             lambda o: o.replay_contour_animation(), '_contour_replayed'),
                            (lambda o: len(getattr(o, '_fill_gaps_inserted_indices', None) or []) > 0,
                             lambda o: o.replay_fill_gaps_animation(), '_fill_gaps_replayed'),
                            (lambda o: len(getattr(o, '_transitions_inserted_indices', None) or []) > 0,
                             lambda o: o.replay_transitions_animation(), '_transitions_replayed'),
                            (lambda o: getattr(o, '_smooth_bp_after', None) is not None,
                             lambda o: o.replay_smooth_animation(), '_smooth_replayed'),
                            (lambda o: getattr(o, '_cut_color_after', None) is not None,
                             lambda o: o.replay_cut_animation(), '_cut_replayed'),
                            (lambda o: getattr(o, '_stream_smooth_bp_after', None) is not None,
                             lambda o: o.replay_stream_smooth_animation(), '_stream_smooth_replayed'),
                            (lambda o: getattr(o, '_level_select_anim_original', None) is not None,
                             lambda o: o.replay_level_select_animation(), '_level_select_replayed'),
                            (lambda o: getattr(o, '_fiber_anim_waypoints', None) is not None,
                             lambda o: o.replay_fiber_animation(), '_build_fibers_replayed'),
                            (lambda o: getattr(o, '_resample_anim_data', None) is not None,
                             lambda o: o.replay_resample_animation(), '_resample_replayed'),
                            (lambda o: getattr(o, '_mesh_anim_face_bands', None) is not None,
                             lambda o: o.replay_mesh_animation(), '_build_mesh_replayed'),
                            (lambda o: getattr(o, '_tet_anim_internal_edges', None) is not None,
                             lambda o: o.replay_tet_animation(), '_tetrahedralize_replayed'),
                        ]
                        started = False
                        while obj._play_all_step < 12 and not started:
                            step = obj._play_all_step
                            obj._play_all_step += 1
                            has_data, replay, replayed_flag = _play_all_steps[step]
                            if has_data(obj):
                                # Set all prior steps' replayed flags so replay buttons appear
                                for prev_i in range(step):
                                    _, _, prev_flag = _play_all_steps[prev_i]
                                    setattr(obj, prev_flag, True)
                                replay(obj); started = True
                        if not started:
                            # All remaining steps had no data — set all replayed flags
                            for _, _, flag in _play_all_steps:
                                setattr(obj, flag, True)
                            obj._play_all_active = False

            # Auto-rotate around focused muscle
            if self.auto_rotate and not self._auto_rotate_paused:
                ar_dt = start_time - getattr(self, '_auto_rotate_last_time', start_time)
                if ar_dt <= 0:
                    ar_dt = 1.0 / 30.0
                angle = self.auto_rotate_speed * ar_dt
                half = angle / 2.0
                rot_quat = np.quaternion(np.cos(half), 0.0, np.sin(half), 0.0)
                self.trackball.curr_quat = rot_quat * self.trackball.curr_quat
            self._auto_rotate_last_time = start_time

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

            while time.time() - start_time < 1.0 / 30:
                time.sleep(1E-6)

        self.impl.shutdown()
        glfw.terminate()
        return

