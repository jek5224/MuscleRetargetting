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
from skeleton_section import SKEL_dart_info

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

        self.mouse_x = 0
        self.mouse_y = 0
        self.motion_skel = None
        
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
        self.is_draw_zygote_muscle = True
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

        # 2D Inspect window state
        self.inspect_2d_open = {}  # Dict: muscle_name -> bool (window open state)
        self.inspect_2d_stream_idx = {}  # Dict: muscle_name -> selected stream index
        self.inspect_2d_contour_idx = {}  # Dict: muscle_name -> selected contour index

        # Correspondence mode state for Inspect 2D
        self.inspect_2d_corr_mode = {}  # Dict: muscle_name -> bool (correspondence mode active)
        self.inspect_2d_corr_corner = {}  # Dict: muscle_name -> selected corner index (0-3) or -1
        self.inspect_2d_corr_vertex = {}  # Dict: muscle_name -> selected vertex index or -1

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

    def update_available_muscles(self):
        """Scan muscle directory and subdirectories for available .obj files not yet loaded."""
        self.available_muscle_files = []
        self.available_muscle_by_category = {}
        loaded_names = set(self.zygote_muscle_meshes.keys())

        # Scan root directory and subdirectories
        for root, dirs, files in os.walk(self.zygote_muscle_dir):
            for file in files:
                if file.endswith('.obj'):
                    muscle_name = file.split('.')[0]
                    if muscle_name not in loaded_names:
                        full_path = os.path.join(root, file)
                        self.available_muscle_files.append((muscle_name, full_path))

                        # Determine category from relative path
                        rel_path = os.path.relpath(root, self.zygote_muscle_dir)
                        if rel_path == '.':
                            category = 'Root'
                        else:
                            category = rel_path.replace(os.sep, '/')

                        if category not in self.available_muscle_by_category:
                            self.available_muscle_by_category[category] = []
                            # Initialize expanded state (collapsed by default)
                            if category not in self.available_category_expanded:
                                self.available_category_expanded[category] = False

                        self.available_muscle_by_category[category].append((muscle_name, full_path))

        # Sort categories and muscles within each category
        self.available_muscle_by_category = dict(sorted(self.available_muscle_by_category.items()))
        for category in self.available_muscle_by_category:
            self.available_muscle_by_category[category].sort(key=lambda x: x[0])

        # Sort flat list by name (for backwards compatibility)
        self.available_muscle_files.sort(key=lambda x: x[0])

        # Reset selection if current selection is no longer valid
        if self.available_selected_muscle:
            # Check if selected muscle still exists in available list
            found = False
            for cat, muscles in self.available_muscle_by_category.items():
                for name, path in muscles:
                    if name == self.available_selected_muscle:
                        found = True
                        break
                if found:
                    break
            if not found:
                self.available_selected_category = None
                self.available_selected_muscle = None

        # Reset old index-based selection if out of bounds
        if self.available_muscle_selected >= len(self.available_muscle_files):
            self.available_muscle_selected = max(0, len(self.available_muscle_files) - 1)

    def add_muscle_mesh(self, name, path):
        """Dynamically add a muscle mesh to the simulation."""
        if name in self.zygote_muscle_meshes:
            return  # Already loaded

        # Remember current position in category for cursor maintenance
        prev_category = self.available_selected_category
        prev_index = -1
        if prev_category and prev_category in self.available_muscle_by_category:
            muscles_in_cat = self.available_muscle_by_category[prev_category]
            for i, (mname, mpath) in enumerate(muscles_in_cat):
                if mname == name:
                    prev_index = i
                    break

        self.zygote_muscle_meshes[name] = MeshLoader()
        self.zygote_muscle_meshes[name].load(path)
        self.zygote_muscle_meshes[name].color = np.array(self.zygote_muscle_color)
        self.zygote_muscle_meshes[name].transparency = self.zygote_muscle_transparency
        self.zygote_muscle_meshes[name].is_draw = self.is_draw_zygote_muscle
        # Load trimesh and apply same scale as MeshLoader.load() uses
        muscle_trimesh = trimesh.load_mesh(path)
        muscle_trimesh.vertices *= 0.01  # MESH_SCALE
        self.zygote_muscle_meshes[name].trimesh = muscle_trimesh

        # Re-sort meshes by name
        self.zygote_muscle_meshes = dict(sorted(self.zygote_muscle_meshes.items()))

        # Update available list
        self.update_available_muscles()

        # Auto-save muscle list
        self.save_loaded_muscles()

        # Maintain cursor position in the same category
        if prev_category and prev_category in self.available_muscle_by_category:
            muscles_in_cat = self.available_muscle_by_category[prev_category]
            if len(muscles_in_cat) > 0:
                # Select same index or previous if at end
                new_index = min(prev_index, len(muscles_in_cat) - 1)
                self.available_selected_category = prev_category
                self.available_selected_muscle = muscles_in_cat[new_index][0]
            else:
                # Category is now empty, clear selection
                self.available_selected_category = None
                self.available_selected_muscle = None
        else:
            self.available_selected_category = None
            self.available_selected_muscle = None

    def remove_muscle_mesh(self, name):
        """Dynamically remove a muscle mesh from the simulation."""
        if name not in self.zygote_muscle_meshes:
            return  # Not loaded

        del self.zygote_muscle_meshes[name]

        # Reset loaded selection if out of bounds
        if self.loaded_muscle_selected >= len(self.zygote_muscle_meshes):
            self.loaded_muscle_selected = max(0, len(self.zygote_muscle_meshes) - 1)

        # Update available list
        self.update_available_muscles()

        # Auto-save muscle list
        self.save_loaded_muscles()

    def save_loaded_muscles(self):
        """Save current loaded muscle names to file for later reload."""
        import json
        try:
            # Build list of (name, path) for all loaded muscles
            muscle_list = []
            for name, mobj in self.zygote_muscle_meshes.items():
                # Get path from mesh object's stored filename
                path = getattr(mobj, 'obj', None)
                if path is None:
                    # Fallback: try to reconstruct path
                    path = self.zygote_muscle_dir + name + '.obj'
                muscle_list.append({'name': name, 'path': path})

            with open(self.last_muscles_file, 'w') as f:
                json.dump(muscle_list, f, indent=2)
            print(f"Saved {len(muscle_list)} muscle names to {self.last_muscles_file}")
        except Exception as e:
            print(f"Failed to save muscle list: {e}")

    def load_previous_muscles(self):
        """Load muscles that were previously saved."""
        import json
        import os
        if not os.path.exists(self.last_muscles_file):
            print(f"No previous muscle list found at {self.last_muscles_file}")
            return 0

        try:
            with open(self.last_muscles_file, 'r') as f:
                muscle_list = json.load(f)

            loaded_count = 0
            for entry in muscle_list:
                name = entry['name']
                path = entry['path']
                if name not in self.zygote_muscle_meshes:
                    if os.path.exists(path):
                        self.add_muscle_mesh(name, path)
                        loaded_count += 1
                    else:
                        print(f"  Muscle file not found: {path}")
            print(f"Loaded {loaded_count} muscles from previous session")
            return loaded_count
        except Exception as e:
            print(f"Failed to load previous muscles: {e}")
            return 0

    def find_inter_muscle_constraints(self, threshold=None):
        """
        Find distance constraints between vertices of different muscles.
        Uses REST positions (from soft_body.rest_positions) to find constraints.
        Only considers muscles with initialized soft bodies.

        Args:
            threshold: Maximum distance to create constraint (default: self.inter_muscle_constraint_threshold)

        Returns:
            Number of constraints found
        """
        if threshold is None:
            threshold = self.inter_muscle_constraint_threshold

        self.inter_muscle_constraints = []

        # Get all muscles with soft body (need rest_positions)
        tet_muscles = {}
        for name, mobj in self.zygote_muscle_meshes.items():
            if hasattr(mobj, 'soft_body') and mobj.soft_body is not None:
                # Use REST positions for finding constraints
                tet_muscles[name] = {
                    'rest_positions': mobj.soft_body.rest_positions.copy(),
                    'fixed_mask': mobj.soft_body.fixed_mask.copy()
                }

        if len(tet_muscles) < 2:
            print(f"Inter-muscle constraints: need at least 2 muscles with soft body, found {len(tet_muscles)}")
            return 0

        muscle_names = list(tet_muscles.keys())
        print(f"Finding inter-muscle constraints for {len(muscle_names)} muscles (threshold={threshold*100:.1f}cm)...")

        # For each pair of muscles
        from scipy.spatial import cKDTree

        for i in range(len(muscle_names)):
            name1 = muscle_names[i]
            data1 = tet_muscles[name1]
            verts1 = data1['rest_positions']
            fixed1 = data1['fixed_mask']

            for j in range(i + 1, len(muscle_names)):
                name2 = muscle_names[j]
                data2 = tet_muscles[name2]
                verts2 = data2['rest_positions']
                fixed2 = data2['fixed_mask']

                # Build KD-tree for muscle2 vertices
                tree2 = cKDTree(verts2)

                # Find all pairs within threshold
                for v1_idx, v1 in enumerate(verts1):
                    # Query nearby vertices in muscle2
                    nearby_indices = tree2.query_ball_point(v1, threshold)

                    for v2_idx in nearby_indices:
                        v2 = verts2[v2_idx]
                        dist = np.linalg.norm(v1 - v2)

                        # Store constraint with fixed status
                        self.inter_muscle_constraints.append((
                            name1, v1_idx, bool(fixed1[v1_idx]),
                            name2, v2_idx, bool(fixed2[v2_idx]),
                            dist  # rest distance
                        ))

        print(f"Found {len(self.inter_muscle_constraints)} inter-muscle constraints")
        return len(self.inter_muscle_constraints)

    def run_all_tet_sim_with_constraints(self, max_iterations=100, tolerance=1e-4, outer_iterations=10):
        """
        Run tet simulation for all muscles together, respecting inter-muscle constraints.
        Uses ARAP with collision detection integrated.

        If self.coupled_as_unified_volume is True, treats all muscles as one unified system.
        """
        # Get all muscles with soft body
        active_muscles = {}
        for name, mobj in self.zygote_muscle_meshes.items():
            if mobj.soft_body is not None:
                active_muscles[name] = mobj

        if len(active_muscles) == 0:
            print("No muscles with soft body initialized")
            return

        n_constraints = len(self.inter_muscle_constraints)

        if self.coupled_as_unified_volume and n_constraints > 0:
            # Unified volume mode: treat all muscles as one system
            self._run_unified_volume_sim(active_muscles, max_iterations, tolerance)
        else:
            # Standard mode: alternating individual solve + constraint enforcement
            print(f"Running coupled tet sim for {len(active_muscles)} muscles with {n_constraints} inter-muscle constraints...")

            for outer_iter in range(outer_iterations):
                # Step 1: Run individual soft body solves
                total_residual = 0
                for name, mobj in active_muscles.items():
                    iters, residual = mobj.run_soft_body_to_convergence(
                        self.zygote_skeleton_meshes,
                        self.env.skel,
                        max_iterations=max_iterations // outer_iterations,
                        tolerance=tolerance,
                        enable_collision=mobj.soft_body_collision,
                        collision_margin=mobj.soft_body_collision_margin,
                        verbose=False,
                        use_arap=mobj.use_arap
                    )
                    total_residual += residual

                # Step 2: Enforce inter-muscle constraints
                if n_constraints > 0:
                    constraint_error = self._enforce_inter_muscle_constraints(active_muscles)
                    print(f"  Iter {outer_iter+1}/{outer_iterations}: residual={total_residual:.2e}, constraint_err={constraint_error:.6f}m")

                    if constraint_error < tolerance:
                        print(f"  Constraints satisfied (error < {tolerance}), stopping")
                        break
                else:
                    print(f"  Iter {outer_iter+1}: residual={total_residual:.2e} (no constraints)")
                    if total_residual < tolerance * len(active_muscles):
                        break

            print(f"Coupled tet sim complete ({len(active_muscles)} muscles)")

    def _run_unified_volume_sim(self, active_muscles, max_iterations=100, tolerance=1e-4):
        """
        Run simulation treating all muscles as one unified volume.
        Inter-muscle constraints become edges in a combined system.
        Supports GPU acceleration via PyTorch when self.use_gpu_arap is True.
        """
        import scipy.sparse
        import scipy.sparse.linalg
        import time

        if self.use_taichi_arap:
            backend_name = 'taichi'
        elif self.use_gpu_arap:
            backend_name = 'gpu'
        else:
            backend_name = 'cpu'
        print(f"Running UNIFIED volume sim for {len(active_muscles)} muscles... [{backend_name.upper()}]")

        # Step 1: Update each muscle's positions and fixed targets from skeleton
        print(f"  Updating positions from skeleton...")
        for name, mobj in active_muscles.items():
            # Update vertex positions based on skeleton bindings
            if hasattr(mobj, '_update_tet_positions_from_skeleton'):
                mobj._update_tet_positions_from_skeleton(self.env.skel)
            # Update fixed vertex targets (origins/insertions)
            if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
                mobj._update_fixed_targets_from_skeleton(self.zygote_skeleton_meshes, self.env.skel)

        # Build global vertex indexing
        muscle_names = list(active_muscles.keys())
        global_offset = {}  # muscle_name -> starting index in global array
        total_verts = 0
        for name in muscle_names:
            global_offset[name] = total_verts
            total_verts += active_muscles[name].soft_body.num_vertices

        print(f"  Total vertices: {total_verts}")

        # Collect all positions into one array
        global_positions = np.zeros((total_verts, 3))
        global_rest_positions = np.zeros((total_verts, 3))
        global_fixed_mask = np.zeros(total_verts, dtype=bool)
        global_fixed_targets = {}  # global_idx -> target position

        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            global_positions[offset:offset+n] = mobj.soft_body.positions
            global_rest_positions[offset:offset+n] = mobj.soft_body.rest_positions
            global_fixed_mask[offset:offset+n] = mobj.soft_body.fixed_mask

            # Store fixed targets
            if mobj.soft_body.fixed_targets is not None and len(mobj.soft_body.fixed_indices) > 0:
                for local_idx, target in zip(mobj.soft_body.fixed_indices, mobj.soft_body.fixed_targets):
                    global_fixed_targets[offset + local_idx] = target

        # Debug: check if fixed targets differ from rest
        max_fixed_diff = 0.0
        for gi, target in global_fixed_targets.items():
            diff = np.linalg.norm(target - global_rest_positions[gi])
            max_fixed_diff = max(max_fixed_diff, diff)

        n_fixed_mask = np.sum(global_fixed_mask)
        n_fixed_targets = len(global_fixed_targets)
        print(f"  Fixed: {n_fixed_mask} in mask, {n_fixed_targets} with targets, max displacement from rest: {max_fixed_diff:.4f}m")
        if n_fixed_mask != n_fixed_targets:
            print(f"  WARNING: Mismatch between fixed_mask ({n_fixed_mask}) and fixed_targets ({n_fixed_targets})")

        # Build combined edge list (internal edges + inter-muscle constraints)
        all_edges = []  # (global_i, global_j, rest_length, weight)

        # Add internal edges from each muscle
        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            sb = mobj.soft_body
            for edge_idx, (i, j) in enumerate(zip(sb.edge_i, sb.edge_j)):
                # Use stored rest_lengths if available, otherwise compute from rest positions
                if hasattr(sb, 'rest_lengths') and sb.rest_lengths is not None and edge_idx < len(sb.rest_lengths):
                    rest_len = sb.rest_lengths[edge_idx]
                else:
                    rest_len = np.linalg.norm(sb.rest_positions[j] - sb.rest_positions[i])
                all_edges.append((offset + i, offset + j, rest_len, 1.0))

        n_internal = len(all_edges)

        # Add inter-muscle constraints as edges
        for constraint in self.inter_muscle_constraints:
            name1, v1_idx, v1_fixed, name2, v2_idx, v2_fixed, rest_dist = constraint
            if name1 in active_muscles and name2 in active_muscles:
                global_i = global_offset[name1] + v1_idx
                global_j = global_offset[name2] + v2_idx
                all_edges.append((global_i, global_j, rest_dist, 1.0))  # Same weight as internal

        n_inter = len(all_edges) - n_internal
        print(f"  Edges: {n_internal} internal + {n_inter} inter-muscle = {len(all_edges)} total")

        # Build neighbor list
        neighbors = [[] for _ in range(total_verts)]
        edge_weights = {}
        rest_edge_vectors = [{} for _ in range(total_verts)]

        for gi, gj, rest_len, weight in all_edges:
            neighbors[gi].append(gj)
            neighbors[gj].append(gi)
            edge_weights[(gi, gj)] = weight
            edge_weights[(gj, gi)] = weight
            # Store rest edge vectors
            rest_edge_vectors[gi][gj] = global_rest_positions[gj] - global_rest_positions[gi]
            rest_edge_vectors[gj][gi] = global_rest_positions[gi] - global_rest_positions[gj]

        # Debug: check neighbor distribution
        neighbor_counts = [len(neighbors[i]) for i in range(total_verts)]
        n0 = sum(1 for c in neighbor_counts if c == 0)
        n1 = sum(1 for c in neighbor_counts if c == 1)
        n2 = sum(1 for c in neighbor_counts if c == 2)
        if n0 > 0 or n1 > 0:
            print(f"  WARNING: {n0} isolated, {n1} single-neighbor, {n2} two-neighbor vertices")

        # Create or get ARAP backend
        backend = get_backend(backend_name)

        # Prepare fixed targets array (ordered by fixed indices)
        fixed_indices = np.where(global_fixed_mask)[0]
        fixed_targets_array = np.array([global_fixed_targets.get(i, global_rest_positions[i]) for i in fixed_indices])

        # Build system and run ARAP
        start_time = time.time()
        backend.build_system(total_verts, neighbors, edge_weights, global_fixed_mask, regularization=1e-6)
        print(f"  System built in {time.time() - start_time:.3f}s")

        start_time = time.time()
        global_positions, iterations, max_disp = backend.solve(
            global_positions, global_rest_positions, neighbors, edge_weights, rest_edge_vectors,
            global_fixed_mask, fixed_targets_array, max_iterations=max_iterations, tolerance=tolerance,
            target_edges=None, verbose=True
        )
        print(f"  ARAP solved in {time.time() - start_time:.3f}s ({iterations} iterations)")

        # Fix isolated vertices (0 neighbors) by finding closest non-isolated vertex
        # and applying same displacement
        isolated_fixed = 0
        for i in range(total_verts):
            if len(neighbors[i]) == 0 and not global_fixed_mask[i]:
                # Find closest vertex that has neighbors
                rest_pos = global_rest_positions[i]
                best_dist = float('inf')
                best_j = -1
                for j in range(total_verts):
                    if j != i and len(neighbors[j]) > 0:
                        d = np.linalg.norm(global_rest_positions[j] - rest_pos)
                        if d < best_dist:
                            best_dist = d
                            best_j = j
                if best_j >= 0:
                    # Apply same displacement as closest connected vertex
                    disp = global_positions[best_j] - global_rest_positions[best_j]
                    global_positions[i] = rest_pos + disp
                    isolated_fixed += 1
        if isolated_fixed > 0:
            print(f"  Fixed {isolated_fixed} isolated vertices by copying nearby displacement")

        # Check for other stuck vertices and fix them
        # Only apply if there's actual deformation (fixed vertices moved from rest)
        fixed_indices = np.where(global_fixed_mask)[0]
        fixed_disp = np.linalg.norm(global_positions[fixed_indices] - global_rest_positions[fixed_indices], axis=1)
        max_fixed_disp = np.max(fixed_disp) if len(fixed_disp) > 0 else 0.0

        if max_fixed_disp > 1e-6:  # Only fix stuck vertices if there's actual deformation
            total_disp_from_rest = np.linalg.norm(global_positions - global_rest_positions, axis=1)
            stuck_threshold = 1e-6
            stuck_count = 0
            for i in range(total_verts):
                if not global_fixed_mask[i] and total_disp_from_rest[i] < stuck_threshold:
                    n_neighbors = len(neighbors[i])
                    if n_neighbors > 0:
                        neighbor_positions = [global_positions[j] for j in neighbors[i]]
                        neighbor_avg = np.mean(neighbor_positions, axis=0)
                        # Move toward neighbor average
                        global_positions[i] = 0.3 * global_positions[i] + 0.7 * neighbor_avg
                        stuck_count += 1
            if stuck_count > 0:
                print(f"  Fixed {stuck_count} stuck vertices by moving toward neighbors")

        # Distribute results back to individual muscles
        total_change = 0.0
        for name, mobj in active_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            old_pos = mobj.tet_vertices.copy() if mobj.tet_vertices is not None else mobj.soft_body.rest_positions
            mobj.soft_body.positions = global_positions[offset:offset+n].copy()
            mobj.tet_vertices = mobj.soft_body.get_positions().astype(np.float32)
            mobj._prepare_tet_draw_arrays()

            # Update waypoints/fibers from deformed tetrahedra
            if getattr(mobj, 'waypoints_from_tet_sim', True):
                if hasattr(mobj, 'waypoints') and len(mobj.waypoints) > 0:
                    if hasattr(mobj, '_update_waypoints_from_tet'):
                        mobj._update_waypoints_from_tet(self.env.skel)

            # Calculate change
            change = np.linalg.norm(mobj.tet_vertices - old_pos, axis=1).max()
            total_change = max(total_change, change)
            print(f"    {name}: max vertex change = {change:.4f}m")

        print(f"Unified volume sim complete (max change: {total_change:.4f}m)")

    def _enforce_inter_muscle_constraints(self, active_muscles, stiffness=0.5):
        """
        Enforce inter-muscle distance constraints by adjusting vertex positions.
        Respects fixed vertices - only moves free vertices.

        Returns: total constraint error
        """
        total_error = 0
        constraint_count = 0

        for constraint in self.inter_muscle_constraints:
            # Unpack constraint (new format with fixed status)
            name1, v1_idx, v1_fixed, name2, v2_idx, v2_fixed, rest_dist = constraint

            if name1 not in active_muscles or name2 not in active_muscles:
                continue

            # Skip if both vertices are fixed
            if v1_fixed and v2_fixed:
                continue

            mobj1 = active_muscles[name1]
            mobj2 = active_muscles[name2]

            # Get current positions
            pos1 = mobj1.soft_body.positions[v1_idx].copy()
            pos2 = mobj2.soft_body.positions[v2_idx].copy()

            # Current distance
            diff = pos2 - pos1
            curr_dist = np.linalg.norm(diff)

            if curr_dist < 1e-8:
                continue

            # Error
            error = curr_dist - rest_dist
            total_error += abs(error)
            constraint_count += 1

            # Correction direction
            direction = diff / curr_dist

            # Determine correction weights based on fixed status
            if v1_fixed:
                # Only v2 moves
                w1, w2 = 0.0, 1.0
            elif v2_fixed:
                # Only v1 moves
                w1, w2 = 1.0, 0.0
            else:
                # Both move equally
                w1, w2 = 0.5, 0.5

            # Apply correction
            correction = error * stiffness
            mobj1.soft_body.positions[v1_idx] += direction * correction * w1
            mobj2.soft_body.positions[v2_idx] -= direction * correction * w2

        # Update tet_vertices for rendering
        for name, mobj in active_muscles.items():
            mobj.tet_vertices = mobj.soft_body.get_positions().astype(np.float32)
            mobj._prepare_tet_draw_arrays()

        return total_error / max(1, constraint_count)

    def draw_inter_muscle_constraint_lines(self):
        """Draw lines between inter-muscle constraint vertex pairs with strain visualization."""
        if not hasattr(self, 'inter_muscle_constraints') or len(self.inter_muscle_constraints) == 0:
            return

        from OpenGL.GL import (glPushMatrix, glPopMatrix, glDisable, glEnable,
                               glBegin, glEnd, glVertex3fv, glColor4f, glLineWidth,
                               GL_LIGHTING, GL_LINES, GL_BLEND, glBlendFunc,
                               GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        import numpy as np

        glPushMatrix()
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(2.0)

        glBegin(GL_LINES)
        for constraint in self.inter_muscle_constraints:
            name1, v1_idx, v1_fixed, name2, v2_idx, v2_fixed, rest_dist = constraint

            if name1 not in self.zygote_muscle_meshes or name2 not in self.zygote_muscle_meshes:
                continue

            mobj1 = self.zygote_muscle_meshes[name1]
            mobj2 = self.zygote_muscle_meshes[name2]

            if mobj1.tet_vertices is None or mobj2.tet_vertices is None:
                continue
            if v1_idx >= len(mobj1.tet_vertices) or v2_idx >= len(mobj2.tet_vertices):
                continue

            v1 = mobj1.tet_vertices[v1_idx]
            v2 = mobj2.tet_vertices[v2_idx]
            current_dist = np.linalg.norm(v2 - v1)

            # Calculate strain: positive = stretched, negative = compressed
            strain = (current_dist - rest_dist) / rest_dist if rest_dist > 0 else 0

            # Color based on strain with transparency
            if strain > 0.01:  # Stretched
                t = min(strain * 5, 1.0)
                glColor4f(1.0, 1.0 - t * 0.7, 0.3, 0.5)  # Yellow -> Red
            elif strain < -0.01:  # Compressed
                t = min(-strain * 5, 1.0)
                glColor4f(0.3, 1.0, 0.3 + t * 0.7, 0.5)  # Green -> Cyan
            else:  # Near rest length
                glColor4f(0.9, 0.9, 0.9, 0.4)  # White/gray

            glVertex3fv(v1)
            glVertex3fv(v2)

        glEnd()
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def setEnv(self, env):  
        self.env = env
        self.motion_skel = self.env.skel.clone()
        self.test_dofs = np.zeros(self.motion_skel.getNumDofs())

        for k, info in self.env.mesh_info.items():
            self.meshes[k] = MeshLoader()
            self.meshes[k].load(info)

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
        self.load_previous_muscles()
        self.zygote_muscle_meshes = dict(sorted(self.zygote_muscle_meshes.items()))

        # Update available muscles list
        self.update_available_muscles()

        # self.reset(self.reset_value)
        self.zero_reset()

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
                self.meshes[bn.getName()].draw(np.array([color[0], color[1], color[2], self.obj_trans]))

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

    def drawTestMuscles(self, color=None):
        if color is not None:
            glColor4d(color[0], color[1], color[2], color[3])
        glLineWidth(self.line_width)
        idx = 0
        for m_wps in self.env.test_muscle_pos:  # m_wps = muscle waypoints
            glBegin(GL_LINE_STRIP)
            for wp in m_wps:
                glVertex3f(wp[0], wp[1], wp[2])
            glEnd() 
            idx += 1

    def drawMuscles(self, color=None):
        if color is not None:
            glColor4d(color[0], color[1], color[2], color[3])
        if self.draw_line_muscle:
            glDisable(GL_LIGHTING)
            glLineWidth(self.line_width)
            for idx, m_wps in enumerate(self.env.muscle_pos):
                # Bounds check for activation levels
                if idx < len(self.env.muscle_activation_levels):
                    a = self.env.muscle_activation_levels[idx]
                else:
                    a = 0.0  # Default activation if index out of bounds
                if color is None:
                    # if idx == self.muscle_index:
                    #     glColor4d(10, 0, 0, 1)
                    # else:
                    #     glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
                    glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
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
            glEnable(GL_LIGHTING)
        else:
            for idx, m_wps in enumerate(self.env.muscle_pos):
                # Bounds check for activation levels
                if idx < len(self.env.muscle_activation_levels):
                    a = self.env.muscle_activation_levels[idx]
                else:
                    a = 0.0  # Default activation if index out of bounds
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

        for name, obj in self.zygote_muscle_meshes.items():
            # Check if VIPER only mode is active
            viper_only = obj.viper_sim is not None and obj.viper_only_mode

            # Draw VIPER rods (always if available and is_draw_viper is True)
            if obj.viper_sim is not None and obj.is_draw_viper:
                obj.draw_viper()

            # Draw VIPER rod-based mesh
            if obj.viper_sim is not None and getattr(obj, 'is_draw_viper_rod_mesh', False):
                obj.draw_viper_mesh()

            # Skip other visualizations if VIPER only mode
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
                if obj.is_draw_fiber_architecture:
                    obj.draw_fiber_architecture()
                if obj.is_draw_contour_mesh:
                    obj.draw_contour_mesh()
                if obj.is_draw_tet_mesh:
                    obj.draw_tetrahedron_mesh(draw_tets=obj.is_draw_tet_edges)
                if obj.is_draw_constraints:
                    obj.draw_constraints()
                if obj.is_draw:
                    obj.draw([obj.color[0], obj.color[1], obj.color[2], obj.transparency])

        # Draw inter-muscle constraints if enabled
        if getattr(self, 'draw_inter_muscle_constraints', False):
            self.draw_inter_muscle_constraint_lines()

        for name, obj in self.zygote_skeleton_meshes.items():
            if obj.is_draw:
                obj.draw([obj.color[0], obj.color[1], obj.color[2], obj.transparency])
            if obj.is_draw_corners:
                obj.draw_corners()
            if obj.is_draw_edges:
                obj.draw_edges()

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
                self.drawMuscles(shadow_color)
            
            if self.draw_target_motion:
                self.drawSkeleton(self.env.target_pos, np.array([1.0, 0.3, 0.3, 0.2]))
            if self.draw_body:
                self.drawSkeleton(self.env.skel.getPositions(), shadow_color)
            if self.draw_pd_target:
                self.drawSkeleton(self.env.pd_target, np.array([0.3, 0.3, 1.0, 0.2]))

            glPopMatrix()

    def _build_combined_collision_mesh(self):
        """Build a combined collision mesh from all skeleton bones for batch operations."""
        import trimesh

        if not hasattr(self, 'zygote_skeleton_meshes') or self.zygote_skeleton_meshes is None:
            return None
        if self.env.skel is None:
            return None

        all_vertices = []
        all_faces = []
        vertex_offset = 0
        skeleton = self.env.skel

        for mesh_name, mesh_loader in self.zygote_skeleton_meshes.items():
            # Find the body node for this mesh
            body_node = skeleton.getBodyNode(mesh_name)
            if body_node is None:
                for i in range(skeleton.getNumBodyNodes()):
                    bn = skeleton.getBodyNode(i)
                    bn_name = bn.getName()
                    if mesh_name.lower() in bn_name.lower() or bn_name.lower() in mesh_name.lower():
                        body_node = bn
                        break

            if body_node is None:
                continue

            # Get or create trimesh
            if not hasattr(mesh_loader, 'trimesh') or mesh_loader.trimesh is None:
                if hasattr(mesh_loader, 'vertices') and hasattr(mesh_loader, 'faces'):
                    if mesh_loader.vertices is not None and mesh_loader.faces is not None and len(mesh_loader.vertices) > 0:
                        try:
                            mesh_loader.trimesh = trimesh.Trimesh(
                                vertices=np.array(mesh_loader.vertices),
                                faces=np.array(mesh_loader.faces),
                                process=False
                            )
                        except:
                            continue

            if mesh_loader.trimesh is None:
                continue

            try:
                original_verts = np.array(mesh_loader.trimesh.vertices)
                faces = np.array(mesh_loader.trimesh.faces)

                # Get current world transform
                world_transform = body_node.getWorldTransform()
                R_curr = world_transform.rotation()
                T_curr = world_transform.translation()

                # Get rest transforms if available (stored during soft body init)
                body_name = body_node.getName()
                # Check any muscle for skeleton_rest_transforms
                rest_transform = None
                for mobj in self.zygote_muscle_meshes.values():
                    if hasattr(mobj, 'skeleton_rest_transforms') and body_name in mobj.skeleton_rest_transforms:
                        rest_transform = mobj.skeleton_rest_transforms[body_name]
                        break

                if rest_transform is not None:
                    R_rest, T_rest = rest_transform
                    delta_rotation = R_curr @ R_rest.T
                    transformed_verts = (delta_rotation @ (original_verts - T_rest).T).T + T_curr
                else:
                    # No rest transform, assume OBJ is already at rest
                    transformed_verts = original_verts

                all_vertices.append(transformed_verts)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(transformed_verts)

            except Exception as e:
                continue

        if len(all_vertices) == 0:
            return None

        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)

        return trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces,
            process=False
        )

    def _apply_batched_collision(self, collision_mesh, margin=2.0, num_passes=2):
        """Apply collision to all muscles at once with a single batched query."""
        if collision_mesh is None:
            return

        # Collect all proximal vertices from all muscles
        position_arrays = []
        muscle_info = []  # (muscle_obj, proximal_indices, start_idx, end_idx)
        total_count = 0

        for mname, mobj in self.zygote_muscle_meshes.items():
            if mobj.soft_body is None:
                continue
            if not hasattr(mobj, 'bone_proximal_vertices') or len(mobj.bone_proximal_vertices) == 0:
                continue

            proximal_indices = np.array(list(mobj.bone_proximal_vertices.keys()))
            positions = mobj.soft_body.positions[proximal_indices].copy()

            start_idx = total_count
            total_count += len(positions)
            end_idx = total_count

            position_arrays.append(positions)
            muscle_info.append((mobj, proximal_indices, start_idx, end_idx))

        if len(position_arrays) == 0:
            print("No proximal vertices to check for collision")
            return

        all_positions = np.vstack(position_arrays)
        print(f"Batched collision: {len(all_positions)} vertices from {len(muscle_info)} muscles")

        # Do collision passes
        for pass_idx in range(num_passes):
            # Single batched query for all vertices
            closest_points, distances, face_ids = collision_mesh.nearest.on_surface(all_positions)

            need_push = distances < margin
            if not np.any(need_push):
                break

            push_indices = np.where(need_push)[0]
            face_normals = collision_mesh.face_normals[face_ids[push_indices]]
            all_positions[push_indices] = closest_points[push_indices] + face_normals * (margin + 1.0)

        # Distribute results back to muscles
        for mobj, proximal_indices, start_idx, end_idx in muscle_info:
            new_positions = all_positions[start_idx:end_idx]
            mobj.soft_body.positions[proximal_indices] = new_positions
            # Quick relax
            mobj.soft_body.step(5)
            # Update rendering
            mobj.tet_vertices = mobj.soft_body.get_positions().astype(np.float32)
            mobj._prepare_tet_draw_arrays()

        print(f"Batched collision done: {num_passes} passes")

    def newSkeleton(self):
        # current_pos = self.env.skel.getPositions()
        self.env.world.removeSkeleton(self.env.skel)
        self.env.skel = buildFromInfo(self.env.new_skel_info, self.env.root_name)
        self.env.target_skel = self.env.skel.clone()
        self.env.world.addSkeleton(self.env.skel)

        self.env.loading_muscle_info(self.env.new_muscle_info)
        self.env.loading_test_muscle_info(self.env.new_muscle_info)

        self.motion_skel = self.env.skel.clone()
        # self.motion_skel.setPositions(current_pos)
        self.motion_skel.setPositions(self.env.skel.getPositions())

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

        if imgui.tree_node("Zygote", imgui.TREE_NODE_DEFAULT_OPEN):
            # Scrollable child region for Zygote menu
            imgui.begin_child("ZygoteScroll", width=0, height=500, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
            if imgui.tree_node("Muscle", imgui.TREE_NODE_DEFAULT_OPEN):                
                changed, self.is_draw_zygote_muscle = imgui.checkbox("Draw", self.is_draw_zygote_muscle)
                if changed:
                    for name, obj in self.zygote_muscle_meshes.items():
                        obj.is_draw = self.is_draw_zygote_muscle
                changed, self.is_draw_zygote_muscle_open_edges = imgui.checkbox("Draw Open Edges", self.is_draw_zygote_muscle_open_edges)
                if changed:
                    for name, obj in self.zygote_muscle_meshes.items():
                        obj.is_draw_open_edges = self.is_draw_zygote_muscle_open_edges
                    
                _, self.is_draw_one_zygote_muscle = imgui.checkbox("Draw One Muscle", self.is_draw_one_zygote_muscle)
                changed, self.zygote_muscle_color = imgui.color_edit3("Color", *self.zygote_muscle_color)
                if changed:
                    for name, obj in self.zygote_muscle_meshes.items():
                        obj.color = self.zygote_muscle_color

                changed, self.zygote_muscle_transparency = imgui.slider_float("Transparency", self.zygote_muscle_transparency, 0.0, 1.0)
                if changed:
                    for name, obj in self.zygote_muscle_meshes.items():
                        obj.transparency = self.zygote_muscle_transparency
                        if obj.vertex_colors is not None:
                            obj.vertex_colors[:, 3] = obj.transparency

                # Muscle Add/Remove UI
                if imgui.tree_node("Add/Remove Muscles"):
                    imgui.text("Available:")
                    # Calculate total available count
                    total_available = sum(len(muscles) for muscles in self.available_muscle_by_category.values())

                    if total_available > 0:
                        # Draw category-based listbox using child region
                        imgui.begin_child("##available_muscles_child", width=0, height=150, border=True)

                        for category, muscles in self.available_muscle_by_category.items():
                            if len(muscles) == 0:
                                continue

                            # Category header with expand/collapse
                            is_expanded = self.available_category_expanded.get(category, False)
                            arrow = "v" if is_expanded else ">"
                            category_label = f"{arrow} {category} ({len(muscles)})"

                            # Make category clickable
                            if imgui.selectable(category_label, False)[0]:
                                self.available_category_expanded[category] = not is_expanded

                            # Show muscles if expanded
                            if is_expanded:
                                for name, path in muscles:
                                    # Indent muscle names
                                    imgui.indent(15)
                                    is_selected = (self.available_selected_muscle == name)
                                    clicked, _ = imgui.selectable(f"  {name}", is_selected)
                                    if clicked:
                                        self.available_selected_category = category
                                        self.available_selected_muscle = name
                                    # Double-click to add
                                    if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                                        self.add_muscle_mesh(name, path)
                                    imgui.unindent(15)

                        imgui.end_child()
                    else:
                        imgui.text("(none)")

                    # Arrow buttons for add/remove
                    if imgui.button("Add", width=button_width):
                        if self.available_selected_muscle and self.available_selected_category:
                            # Find the path for selected muscle
                            for name, path in self.available_muscle_by_category.get(self.available_selected_category, []):
                                if name == self.available_selected_muscle:
                                    self.add_muscle_mesh(name, path)
                                    break
                    imgui.same_line()
                    if imgui.button("Remove", width=button_width):
                        if len(self.zygote_muscle_meshes) > 0:
                            loaded_names = list(self.zygote_muscle_meshes.keys())
                            name = loaded_names[self.loaded_muscle_selected]
                            self.remove_muscle_mesh(name)

                    imgui.text("Loaded:")
                    loaded_names = list(self.zygote_muscle_meshes.keys())
                    if len(loaded_names) > 0:
                        # Use child region with selectables for double-click support
                        imgui.begin_child("##loaded_muscles_child", width=0, height=150, border=True)
                        for i, name in enumerate(loaded_names):
                            is_selected = (self.loaded_muscle_selected == i)
                            clicked, _ = imgui.selectable(name, is_selected)
                            if clicked:
                                self.loaded_muscle_selected = i
                            # Double-click to remove
                            if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                                self.remove_muscle_mesh(name)
                        imgui.end_child()
                    else:
                        imgui.text("(none)")

                    # Save/Load previous muscles buttons
                    if imgui.button("Save List", width=button_width):
                        self.save_loaded_muscles()
                    imgui.same_line()
                    if imgui.button("Load Previous", width=button_width):
                        self.load_previous_muscles()

                    imgui.tree_pop()

                if imgui.tree_node("Activation levels"):
                    if self.env.zygote_activation_levels is not None:
                        # for i in range(len(self.env.muscle_activation_levels)):
                        #     imgui.push_item_width(push_width)
                        #     _, self.env.muscle_activation_levels[i] = imgui.slider_float(f" ##fiber{i}", self.env.muscle_activation_levels[i], 0.0, 1.0)
                        #     imgui.pop_item_width()

                        # Use shorter slider width to leave room for name
                        slider_width = 120
                        for i, (name, obj) in enumerate(self.env.muscle_info.items()):
                            # Bounds check for zygote_activation_levels
                            if i >= len(self.env.zygote_activation_levels):
                                continue
                            # Show name first (truncate if too long)
                            display_name = name[:25] + "..." if len(name) > 25 else name
                            imgui.text(f"{display_name}")
                            imgui.same_line(position=180)
                            imgui.push_item_width(slider_width)
                            changed, self.env.zygote_activation_levels[i] = imgui.slider_float(f"##zygote_act{i}", self.env.zygote_activation_levels[i], 0.0, 1.0)
                            imgui.pop_item_width()
                            if changed:
                                # Bounds check for activation indices
                                if i + 1 < len(self.env.zygote_activation_indices):
                                    start_fiber = self.env.zygote_activation_indices[i]
                                    end_fiber = self.env.zygote_activation_indices[i + 1]
                                    if end_fiber <= len(self.env.muscle_activation_levels):
                                        self.env.muscle_activation_levels[start_fiber:end_fiber] = self.env.zygote_activation_levels[i]
                                # self.env.muscle_activation_levels[start_fiber:end_fiber] = self.env.zygote_activation_levels[i] / (end_fiber - start_fiber)
                    imgui.tree_pop()
                if imgui.button("Export Muscle Waypoints", width=wide_button_width):
                    from core.dartHelper import exportMuscleWaypoints
                    exportMuscleWaypoints(self.zygote_muscle_meshes, list(self.zygote_skeleton_meshes.keys()))
                if imgui.button("Import zygote_muscle", width=wide_button_width):
                    import os
                    muscle_file = "data/zygote_muscle.xml"
                    if not os.path.exists(muscle_file):
                        print(f"Error: Muscle file not found: {muscle_file}")
                        print("  Run 'Export Muscle Waypoints' first to create it.")
                    else:
                        try:
                            self.env.muscle_info = self.env.saveZygoteMuscleInfo(muscle_file)
                            if not self.env.muscle_info:
                                print("No muscles loaded from file (empty or invalid)")
                            else:
                                self.env.loading_zygote_muscle_info(self.env.muscle_info)
                                self.env.muscle_activation_levels = np.zeros(self.env.muscles.getNumMuscles())

                                self.draw_obj = True
                                # Disable skeleton drawing when importing muscle waypoints
                                self.is_draw_zygote_skeleton = False
                                for name, obj in self.zygote_skeleton_meshes.items():
                                    obj.is_draw = False
                                print(f"Imported {self.env.muscles.getNumMuscles()} muscles from {muscle_file}")
                        except Exception as e:
                            print(f"Error importing muscle waypoints: {e}")

                # Load all tet meshes and init soft bodies
                if imgui.button("Load All Tets", width=wide_button_width):
                    load_count = 0
                    init_count = 0
                    already_init_count = 0
                    for mname, mobj in self.zygote_muscle_meshes.items():
                        try:
                            # If tet already loaded, just check if soft body needs init
                            if mobj.tet_vertices is not None:
                                if mobj.soft_body is None:
                                    # Tet loaded but soft body not initialized - init it
                                    skeleton_names = list(self.zygote_skeleton_meshes.keys())
                                    mobj.resolve_skeleton_attachments(skeleton_names)
                                    mobj.init_soft_body(self.zygote_skeleton_meshes, self.env.skel, self.env.mesh_info)
                                    if mobj.soft_body is not None:
                                        init_count += 1
                                else:
                                    already_init_count += 1
                                continue
                            # Load new tet
                            mobj.soft_body = None  # Reset soft body when loading new tet
                            if mobj.load_tetrahedron_mesh(mname):
                                if mobj.tet_vertices is not None:
                                    mobj.is_draw = False  # Disable mesh draw
                                    mobj.is_draw_contours = False
                                    mobj.is_draw_tet_mesh = True
                                    mobj.is_draw_fiber_architecture = True  # Enable fiber draw
                                    load_count += 1
                                    # Resolve skeleton attachments from names to current indices
                                    skeleton_names = list(self.zygote_skeleton_meshes.keys())
                                    mobj.resolve_skeleton_attachments(skeleton_names)
                                    # Also init soft body
                                    mobj.init_soft_body(self.zygote_skeleton_meshes, self.env.skel, self.env.mesh_info)
                                    if mobj.soft_body is not None:
                                        init_count += 1
                        except Exception as e:
                            print(f"[{mname}] Load Tet error: {e}")
                    print(f"Loaded {load_count} new tets, initialized {init_count} soft bodies ({already_init_count} already initialized)")
                # Run soft body simulation for all muscles at once
                if imgui.button("Run All Tet Sim", width=wide_button_width):
                    count = 0
                    collision_count = 0
                    for mname, mobj in self.zygote_muscle_meshes.items():
                        if mobj.tet_vertices is not None:
                            if mobj.soft_body is None:
                                mobj.init_soft_body(self.zygote_skeleton_meshes, self.env.skel, self.env.mesh_info)
                            if mobj.soft_body is not None:
                                # Respect each muscle's individual collision setting
                                iterations, residual = mobj.run_soft_body_to_convergence(
                                    self.zygote_skeleton_meshes,
                                    self.env.skel,
                                    max_iterations=100,
                                    tolerance=1e-4,
                                    enable_collision=mobj.soft_body_collision,
                                    collision_margin=mobj.soft_body_collision_margin,
                                    verbose=False,
                                    use_arap=mobj.use_arap
                                )
                                count += 1
                                if mobj.soft_body_collision:
                                    collision_count += 1
                    print(f"Ran tet sim for {count} muscles ({collision_count} with collision)")

                # Inter-muscle constraints section
                imgui.separator()
                imgui.text("Inter-Muscle Constraints")

                # Threshold slider
                imgui.push_item_width(100)
                changed, self.inter_muscle_constraint_threshold = imgui.slider_float(
                    "Threshold (m)", self.inter_muscle_constraint_threshold, 0.001, 0.05
                )
                imgui.pop_item_width()
                imgui.same_line()
                imgui.text(f"({self.inter_muscle_constraint_threshold*100:.1f}cm)")

                # Find constraints button
                if imgui.button("Find Constraints", width=wide_button_width):
                    count = self.find_inter_muscle_constraints()

                imgui.text(f"{len(self.inter_muscle_constraints)} constraints")
                _, self.draw_inter_muscle_constraints = imgui.checkbox(
                    "Draw##inter_constraints", getattr(self, 'draw_inter_muscle_constraints', False)
                )

                # Unified volume checkbox
                _, self.coupled_as_unified_volume = imgui.checkbox(
                    "Unified Volume", self.coupled_as_unified_volume
                )
                if self.coupled_as_unified_volume:
                    imgui.same_line()
                    imgui.text_colored("(all muscles as one system)", 0.5, 0.8, 0.5)

                # Backend selection (radio-button style with checkboxes)
                imgui.text("Backend:")
                imgui.same_line()

                # CPU (always available, default)
                use_cpu = not self.use_gpu_arap and not self.use_taichi_arap
                if imgui.checkbox("CPU", use_cpu)[1] and not use_cpu:
                    self.use_gpu_arap = False
                    self.use_taichi_arap = False

                # GPU (PyTorch)
                if self.gpu_available:
                    imgui.same_line()
                    changed, checked = imgui.checkbox("GPU", self.use_gpu_arap)
                    if changed and checked:
                        self.use_gpu_arap = True
                        self.use_taichi_arap = False
                    elif changed and not checked:
                        self.use_gpu_arap = False

                # Taichi
                if self.taichi_available:
                    imgui.same_line()
                    changed, checked = imgui.checkbox("Taichi", self.use_taichi_arap)
                    if changed and checked:
                        self.use_taichi_arap = True
                        self.use_gpu_arap = False
                    elif changed and not checked:
                        self.use_taichi_arap = False

                # Run coupled simulation button
                if imgui.button("Run Coupled Tet Sim", width=wide_button_width):
                    self.run_all_tet_sim_with_constraints()

                imgui.separator()

                for name, obj in self.zygote_muscle_meshes.items():
                    if imgui.tree_node(name):
                        # Two-column layout: "Process All" button on left, individual buttons on right
                        imgui.columns(2, f"cols##{name}", border=False)
                        imgui.set_column_width(0, 120)

                        # Left column: Process button with vertical slider
                        num_process_buttons = 12  # Match number of buttons on right
                        process_all_height = num_process_buttons * imgui.get_frame_height() + (num_process_buttons - 1) * imgui.get_style().item_spacing[1]

                        # Initialize process step slider value
                        if not hasattr(obj, '_process_step'):
                            obj._process_step = 12

                        # Vertical slider for step selection (top=1, bottom=12)
                        # v_slider_int format: label, width, height, value, min, max
                        # To get 1 at top and 12 at bottom, we invert: display (13 - value)
                        display_step = 13 - obj._process_step  # Convert for display
                        changed, new_display = imgui.v_slider_int(
                            f"##step{name}", 20, process_all_height, display_step, 1, 12)
                        if changed:
                            obj._process_step = 13 - new_display  # Convert back
                        imgui.same_line()

                        # Step names matching button order (1=top, 12=bottom)
                        # 1:Scalar, 2:Contours, 3:FillGap, 4:Transitions, 5:Smooth, 6:Cut, 7:StreamSmooth, 8:Select, 9:Build, 10:Resample, 11:Mesh, 12:Tet
                        step_names = ['', 'Scalar', 'Contours', 'Fill Gap', 'Transitions', 'Smooth', 'Cut', 'StreamSmooth', 'Select', 'Build', 'Resample', 'Mesh', 'Tet']
                        if imgui.button(f"Process\n1 to {obj._process_step}\n({step_names[obj._process_step]})##{name}", width=75, height=process_all_height):
                            try:
                                max_step = obj._process_step
                                print(f"[{name}] Running pipeline steps 1 to {max_step}...")

                                # Step 1: Scalar Field
                                if max_step >= 1 and len(obj.edge_groups) > 0 and len(obj.edge_classes) > 0:
                                    print(f"  [1/{max_step}] Computing Scalar Field...")
                                    obj.compute_scalar_field()

                                # Step 2: Find Contours
                                if max_step >= 2 and obj.scalar_field is not None:
                                    print(f"  [2/{max_step}] Finding Contours...")
                                    obj.find_contours(skeleton_meshes=self.zygote_skeleton_meshes, spacing_scale=obj.contour_spacing_scale)
                                    obj.is_draw_bounding_box = True

                                # Step 3: Fill Gaps
                                if max_step >= 3 and obj.contours is not None and len(obj.contours) > 0:
                                    print(f"  [3/{max_step}] Filling Gaps...")
                                    obj.refine_contours(max_spacing_threshold=0.01)

                                # Step 4: Find Transitions
                                if max_step >= 4 and obj.scalar_field is not None:
                                    print(f"  [4/{max_step}] Finding Transitions...")
                                    field_min = float(obj.scalar_field.min())
                                    field_max = float(obj.scalar_field.max())
                                    scalar_min, scalar_max = field_min, field_max
                                    if hasattr(obj, 'origin_contour_value') and hasattr(obj, 'insertion_contour_value'):
                                        o_val, i_val = obj.origin_contour_value, obj.insertion_contour_value
                                        if o_val != i_val:
                                            proposed_min, proposed_max = min(o_val, i_val), max(o_val, i_val)
                                            if proposed_min >= field_min and proposed_max <= field_max:
                                                scalar_min, scalar_max = proposed_min, proposed_max
                                    exp_origin = len(obj.contours[0]) if obj.contours and len(obj.contours) > 0 else None
                                    exp_insertion = len(obj.contours[-1]) if obj.contours and len(obj.contours) > 0 else None
                                    obj.find_all_transitions(scalar_min=scalar_min, scalar_max=scalar_max, num_samples=200,
                                                            expected_origin=exp_origin, expected_insertion=exp_insertion)
                                    if obj.contours is not None and len(obj.contours) > 0:
                                        obj.add_transitions_to_contours()

                                # Step 5: Smooth (z, x, bp - before cut)
                                if max_step >= 5 and obj.contours is not None and len(obj.contours) > 0:
                                    print(f"  [5/{max_step}] Smoothening (z, x, bp)...")
                                    obj.smoothen_contours_z()
                                    obj.smoothen_contours_x()
                                    obj.smoothen_contours_bp()

                                # Step 6: Cut
                                if max_step >= 6 and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                                    print(f"  [6/{max_step}] Cutting streams...")
                                    obj.cut_streams(cut_method=obj.cutting_method, muscle_name=name)
                                    # Check if waiting for manual cut
                                    if obj._manual_cut_pending or obj._manual_cut_data is not None:
                                        print(f"  [6/{max_step}] Waiting for manual cut - pipeline paused")
                                        raise StopIteration("Manual cut pending")

                                # Step 7: Stream Smooth (z, x, bp - after cut)
                                if max_step >= 7 and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                                    print(f"  [7/{max_step}] Stream Smoothening (z, x, bp)...")
                                    obj.smoothen_contours_z()
                                    obj.smoothen_contours_x()
                                    obj.smoothen_contours_bp()

                                # Step 8: Contour Select
                                if max_step >= 8 and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                                    print(f"  [8/{max_step}] Selecting contours...")
                                    obj.select_levels()

                                # Step 9: Build Fiber
                                if max_step >= 9 and hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                                    print(f"  [9/{max_step}] Building fibers...")
                                    obj.build_fibers(skeleton_meshes=self.zygote_skeleton_meshes)

                                # Step 10: Resample Contours
                                if max_step >= 10 and obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                                    print(f"  [10/{max_step}] Resampling Contours...")
                                    obj.resample_contours(base_samples=32)

                                # Step 11: Build Contour Mesh
                                if max_step >= 11 and obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
                                    print(f"  [11/{max_step}] Building Contour Mesh...")
                                    obj.build_contour_mesh()

                                # Step 12: Tetrahedralize
                                if max_step >= 12 and obj.contour_mesh_vertices is not None:
                                    print(f"  [12/{max_step}] Tetrahedralizing...")
                                    obj.soft_body = None
                                    obj.tetrahedralize_contour_mesh()
                                    if obj.tet_vertices is not None:
                                        obj.is_draw_contours = False
                                        obj.is_draw_tet_mesh = True

                                print(f"[{name}] Pipeline complete (steps 1-{max_step})!")
                            except StopIteration:
                                pass  # Manual cut pending - pipeline paused gracefully
                            except Exception as e:
                                print(f"[{name}] Pipeline error: {e}")
                                import traceback
                                traceback.print_exc()

                        # Right column: Individual buttons (use -1 to auto-fill column width)
                        imgui.next_column()
                        col_button_width = 180  # Fits in the right column

                        # Helper for button coloring based on process step
                        def colored_button(label, step_num, width):
                            will_run = step_num <= obj._process_step
                            if will_run:
                                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.2, 1.0)
                                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.7, 0.3, 1.0)
                            clicked = imgui.button(label, width=width)
                            if will_run:
                                imgui.pop_style_color(2)
                            return clicked

                        if colored_button(f"Scalar Field##{name}", 1, col_button_width):
                            if len(obj.edge_groups) > 0 and len(obj.edge_classes) > 0:
                                try:
                                    obj.compute_scalar_field()
                                except Exception as e:
                                    print(f"[{name}] Scalar Field error: {e}")
                            else:
                                print(f"[{name}] Need edge_groups and edge_classes")

                        if colored_button(f"Find Contours##{name}", 2, col_button_width):
                            if obj.scalar_field is not None:
                                try:
                                    obj.find_contours(skeleton_meshes=self.zygote_skeleton_meshes, spacing_scale=obj.contour_spacing_scale)
                                    obj.is_draw_bounding_box = True
                                except Exception as e:
                                    print(f"[{name}] Find Contours error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Scalar Field' first")
                        if colored_button(f"Fill Gaps##{name}", 3, col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0:
                                try:
                                    obj.refine_contours(max_spacing_threshold=0.01)
                                except Exception as e:
                                    print(f"[{name}] Fill Gaps error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Contours' first")

                        # Find Transitions button - fast scan for contour count changes (step 4)
                        if colored_button(f"Find Transitions##{name}", 4, col_button_width):
                            if hasattr(obj, 'scalar_field') and obj.scalar_field is not None:
                                try:
                                    # Use actual scalar field range
                                    field_min = float(obj.scalar_field.min())
                                    field_max = float(obj.scalar_field.max())
                                    scalar_min = field_min
                                    scalar_max = field_max
                                    # Optionally narrow to origin/insertion if set AND within field range
                                    if hasattr(obj, 'origin_contour_value') and hasattr(obj, 'insertion_contour_value'):
                                        o_val = obj.origin_contour_value
                                        i_val = obj.insertion_contour_value
                                        # Only use if different AND within the actual scalar field range
                                        if o_val != i_val:
                                            proposed_min = min(o_val, i_val)
                                            proposed_max = max(o_val, i_val)
                                            if proposed_min >= field_min and proposed_max <= field_max:
                                                scalar_min = proposed_min
                                                scalar_max = proposed_max
                                            else:
                                                print(f"[{name}] Origin/insertion values ({o_val}, {i_val}) outside field range [{field_min:.4f}, {field_max:.4f}], using field range")
                                    print(f"[{name}] Scanning scalar range: {scalar_min:.4f} to {scalar_max:.4f}")
                                    # Try to get expected origin/insertion counts from existing contours
                                    exp_origin = None
                                    exp_insertion = None
                                    if hasattr(obj, 'contours') and obj.contours is not None and len(obj.contours) > 0:
                                        exp_origin = len(obj.contours[0])
                                        exp_insertion = len(obj.contours[-1])
                                        print(f"[{name}] Expected counts from contours: origin={exp_origin}, insertion={exp_insertion}")
                                    obj.find_all_transitions(scalar_min=scalar_min, scalar_max=scalar_max, num_samples=200,
                                                            expected_origin=exp_origin, expected_insertion=exp_insertion)
                                    # Auto-add transitions to contours if contours exist
                                    if obj.contours is not None and len(obj.contours) > 0:
                                        obj.add_transitions_to_contours()
                                except Exception as e:
                                    print(f"[{name}] Find Transitions error: {e}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                print(f"[{name}] Prerequisites: Run 'Scalar Field' first")

                        # Step 5: Smoothen buttons: z, x, bp (3 buttons in same row - before cut)
                        sub_button_width = (col_button_width - 8) // 3  # 3 buttons with small margins
                        if colored_button(f"z##{name}", 5, sub_button_width):
                            if obj.contours is not None and len(obj.contours) > 0:
                                try:
                                    obj.smoothen_contours_z()
                                except Exception as e:
                                    print(f"[{name}] Smoothen Z error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Contours' first")
                        imgui.same_line(spacing=4)
                        if colored_button(f"x##{name}", 5, sub_button_width):
                            if obj.contours is not None and len(obj.contours) > 0:
                                try:
                                    obj.smoothen_contours_x()
                                except Exception as e:
                                    print(f"[{name}] Smoothen X error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Contours' first")
                        imgui.same_line(spacing=4)
                        if colored_button(f"bp##{name}", 5, sub_button_width):
                            if obj.contours is not None and len(obj.contours) > 0:
                                try:
                                    obj.smoothen_contours_bp()
                                except Exception as e:
                                    print(f"[{name}] Smoothen BP error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Contours' first")

                        # Step 6: Cut (standalone button)
                        if colored_button(f"Cut##{name}", 6, col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None and len(obj.bounding_planes) > 0:
                                try:
                                    obj.cut_streams(cut_method=obj.cutting_method, muscle_name=name)
                                except Exception as e:
                                    import traceback
                                    print(f"[{name}] Cut Streams error: {e}")
                                    traceback.print_exc()
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Contours' first")

                        # Step 7: Stream Smoothen buttons: z, x, bp (3 buttons in same row - after cut)
                        if colored_button(f"z##stream{name}", 7, sub_button_width):
                            if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                                try:
                                    obj.smoothen_contours_z()
                                except Exception as e:
                                    print(f"[{name}] Stream Smoothen Z error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Cut' first")
                        imgui.same_line(spacing=4)
                        if colored_button(f"x##stream{name}", 7, sub_button_width):
                            if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                                try:
                                    obj.smoothen_contours_x()
                                except Exception as e:
                                    print(f"[{name}] Stream Smoothen X error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Cut' first")
                        imgui.same_line(spacing=4)
                        if colored_button(f"bp##stream{name}", 7, sub_button_width):
                            if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                                try:
                                    obj.smoothen_contours_bp()
                                except Exception as e:
                                    print(f"[{name}] Stream Smoothen BP error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Cut' first")

                        # Step 8: Contour Select (standalone button)
                        if colored_button(f"Contour Select##{name}", 8, col_button_width):
                            if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                                try:
                                    obj.select_levels()
                                except Exception as e:
                                    import traceback
                                    print(f"[{name}] Select Levels error: {e}")
                                    traceback.print_exc()
                            else:
                                print(f"[{name}] Prerequisites: Run 'Cut' first")

                        # Step 9: Build Fiber (standalone button)
                        if colored_button(f"Build Fiber##{name}", 9, col_button_width):
                            if hasattr(obj, 'stream_contours') and obj.stream_contours is not None:
                                try:
                                    obj.build_fibers(skeleton_meshes=self.zygote_skeleton_meshes)
                                except Exception as e:
                                    import traceback
                                    print(f"[{name}] Build Fibers error: {e}")
                                    traceback.print_exc()
                            else:
                                print(f"[{name}] Prerequisites: Run 'Cut' first")

                        # Step 10: Resample Contours
                        if colored_button(f"Resample Contours##{name}", 10, col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                                try:
                                    obj.resample_contours(base_samples=32)
                                except Exception as e:
                                    print(f"[{name}] Resample Contours error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Smoothen Contours' first")

                        # Step 11: Build Contour Mesh
                        if colored_button(f"Build Contour Mesh##{name}", 11, col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
                                try:
                                    obj.build_contour_mesh()
                                except Exception as e:
                                    print(f"[{name}] Build Contour Mesh error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Build Fiber' first")

                        # Step 12: Tetrahedralize
                        if colored_button(f"Tetrahedralize##{name}", 12, col_button_width):
                            if obj.contour_mesh_vertices is not None:
                                try:
                                    obj.soft_body = None  # Reset soft body when re-tetrahedralizing
                                    obj.tetrahedralize_contour_mesh()
                                    if obj.tet_vertices is not None:
                                        obj.is_draw_contours = False
                                        obj.is_draw_tet_mesh = True
                                except Exception as e:
                                    print(f"[{name}] Tetrahedralize error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Build Contour Mesh' first")

                        # End two-column layout - back to full width for remaining GUI elements
                        imgui.columns(1)

                        # Reset process button
                        reset_width = button_width * 2 + imgui.get_style().item_spacing[0]
                        if imgui.button(f"Reset Process##{name}", width=reset_width):
                            obj.reset_process()

                        # Save/Load contours buttons
                        contour_filepath = f"{self.zygote_muscle_dir}{name}.contours.json"
                        if imgui.button(f"Save Contour##{name}", width=button_width):
                            if obj.contours is not None and len(obj.contours) > 0:
                                try:
                                    obj.save_contours(contour_filepath)
                                except Exception as e:
                                    print(f"[{name}] Save Contours error: {e}")
                            else:
                                print(f"[{name}] No contours to save")
                        imgui.same_line()
                        if imgui.button(f"Load Contour##{name}", width=button_width):
                            try:
                                obj.load_contours(contour_filepath)
                            except Exception as e:
                                print(f"[{name}] Load Contours error: {e}")

                        if imgui.button(f"Save Tet##{name}", width=button_width):
                            if hasattr(obj, 'tet_vertices') and obj.tet_vertices is not None:
                                try:
                                    obj.save_tetrahedron_mesh(name)
                                except Exception as e:
                                    print(f"[{name}] Save Tet error: {e}")
                            else:
                                print(f"[{name}] No tetrahedron mesh to save")
                        imgui.same_line()
                        if imgui.button(f"Load Tet##{name}", width=button_width):
                            try:
                                obj.soft_body = None  # Reset soft body when loading new tet
                                obj.load_tetrahedron_mesh(name)
                                if obj.tet_vertices is not None:
                                    obj.is_draw = False  # Disable mesh draw
                                    obj.is_draw_contours = False
                                    obj.is_draw_tet_mesh = True
                                    obj.is_draw_fiber_architecture = True  # Enable fiber draw
                                    # Resolve skeleton attachments from names to current indices
                                    skeleton_names = list(self.zygote_skeleton_meshes.keys())
                                    obj.resolve_skeleton_attachments(skeleton_names)
                            except Exception as e:
                                print(f"[{name}] Load Tet error: {e}")

                        # Inspect 2D button - opens visualization window for contours (and fiber samples if available)
                        inspect_width = button_width * 2 + imgui.get_style().item_spacing[0]
                        has_contour_data = (hasattr(obj, 'contours') and obj.contours is not None and len(obj.contours) > 0)
                        if not has_contour_data:
                            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                        if imgui.button(f"Inspect 2D##{name}", width=inspect_width):
                            if has_contour_data:
                                self.inspect_2d_open[name] = True
                                if name not in self.inspect_2d_stream_idx:
                                    self.inspect_2d_stream_idx[name] = 0
                                if name not in self.inspect_2d_contour_idx:
                                    self.inspect_2d_contour_idx[name] = 0
                            else:
                                print(f"[{name}] No contour data. Run 'Find Contours' first.")
                        if not has_contour_data:
                            imgui.pop_style_var()

                        # Focus camera on muscle button
                        if imgui.button(f"Focus##{name}", width=inspect_width):
                            if obj.vertices is not None and len(obj.vertices) > 0:
                                # Compute bounding box center
                                min_pt = np.min(obj.vertices, axis=0)
                                max_pt = np.max(obj.vertices, axis=0)
                                center = (min_pt + max_pt) / 2
                                bbox_size = np.linalg.norm(max_pt - min_pt)
                                # Set camera target (trans is scaled by 0.001 in render, so multiply by 1000)
                                self.trans = -center * 1000.0
                                # Adjust eye distance based on bounding box size
                                distance = bbox_size * 2.0
                                eye_dir = self.eye / (np.linalg.norm(self.eye) + 1e-10)
                                self.eye = eye_dir * max(distance, MIN_EYE_DISTANCE)
                            else:
                                print(f"[{name}] No vertices to focus on")

                        _, obj.is_one_fiber = imgui.checkbox(f"One Fiber##{name}", obj.is_one_fiber)

                        # Bounding box method selector
                        bbox_methods = ['farthest_vertex', 'pca', 'bbox']
                        current_method = getattr(obj, 'bounding_box_method', 'farthest_vertex')
                        current_idx = bbox_methods.index(current_method) if current_method in bbox_methods else 0
                        changed, new_idx = imgui.combo(f"BBox Method##{name}", current_idx, bbox_methods)
                        if changed:
                            obj.bounding_box_method = bbox_methods[new_idx]

                        # Contour spacing scale slider (lower = more contours)
                        changed, obj.contour_spacing_scale = imgui.slider_float(
                            f"Spacing##{name}", obj.contour_spacing_scale, 0.1, 2.0, "%.2f")

                        # Min level distance for Select Levels (as % of muscle length)
                        if not hasattr(obj, 'min_level_distance'):
                            obj.min_level_distance = 0.01  # Default 1%
                        changed, obj.min_level_distance = imgui.slider_float(
                            f"Min Dist##{name}", obj.min_level_distance, 0.001, 0.1, "%.3f")

                        # Error threshold for level selection (as % of muscle length)
                        if not hasattr(obj, 'level_select_error_threshold'):
                            obj.level_select_error_threshold = 0.005  # Default 0.5%
                        changed, obj.level_select_error_threshold = imgui.slider_float(
                            f"Err Thresh##{name}", obj.level_select_error_threshold, 0.001, 0.05, "%.3f")

                        # Sampling method selector
                        sampling_methods = ['sobol_unit_square', 'sobol_min_contour']
                        current_idx = sampling_methods.index(obj.sampling_method) if obj.sampling_method in sampling_methods else 0
                        changed, new_idx = imgui.combo(f"Sampling##{name}", current_idx, sampling_methods)
                        if changed:
                            obj.sampling_method = sampling_methods[new_idx]

                        # Cutting method selector
                        cutting_methods = ['bp', 'area_based', 'voronoi', 'angular', 'gradient', 'ratio', 'cumulative_area', 'projected_area']
                        current_cut_idx = cutting_methods.index(obj.cutting_method) if obj.cutting_method in cutting_methods else 0
                        changed, new_cut_idx = imgui.combo(f"Cutting##{name}", current_cut_idx, cutting_methods)
                        if changed:
                            obj.cutting_method = cutting_methods[new_cut_idx]

                        imgui.same_line()
                        imgui.text(obj.link_mode)
                        changed1, obj.specific_contour_value = imgui.slider_float(f"Ori##{name}", obj.specific_contour_value, 1.0, obj.contour_value_min, flags=imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)
                        changed2, obj.specific_contour_value = imgui.slider_float(f"Mid##{name}", obj.specific_contour_value, obj.contour_value_min, obj.contour_value_max, flags=imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)
                        changed3, obj.specific_contour_value = imgui.slider_float(f"Ins##{name}", obj.specific_contour_value, obj.contour_value_max, 10.0, flags=imgui.SLIDER_FLAGS_NO_ROUND_TO_FORMAT)
                        if imgui.tree_node(f"MinMax##{name}"):
                            _, obj.contour_value_min = imgui.input_float(f"Min##{name}", obj.contour_value_min)
                            _, obj.contour_value_max = imgui.input_float(f"Max##{name}", obj.contour_value_max)
                            imgui.tree_pop()
                        if changed1 or changed2 or changed3:
                            obj.find_contour_with_value(obj.specific_contour_value)
                        # if imgui.button(f"Find Value Contour##{name}"):
                        #     obj.find_contour_with_value()
                        if imgui.button(f"Switch Link Mode##{name}"):
                            if obj.link_mode == 'mean':
                                obj.link_mode = 'vertex'
                            else:
                                obj.link_mode = 'mean'

                        # Minimum contour distance threshold
                        if hasattr(obj, 'min_contour_distance'):
                            _, obj.min_contour_distance = imgui.slider_float(
                                f"Min Dist##{name}", obj.min_contour_distance, 0.001, 0.02, "%.3f")

                        # if imgui.button("Find Interesecting Bones"):  
                        #     for other_name, other_obj in self.zygote_muscle_meshes.items():
                        #         other_obj.is_draw = False
                        #     obj.is_draw = True

                        #     intersecting_meshes = obj.find_intersections(self.zygote_skeleton_meshes)
                        #     # print(bb_intersect)
                        #     for skel_name, skel_obj in self.zygote_skeleton_meshes.items():
                        #         if skel_name in intersecting_meshes:
                        #             skel_obj.color = np.array([0.0, 0.0, 1.0])
                        #         else:
                        #             skel_obj.color = np.array([0.9, 0.9, 0.9])
                            
                        #     self.zygote_muscle_meshes_intersection_bones[name] = intersecting_meshes

                        changed, obj.transparency = imgui.slider_float(f"Transparency##{name}", obj.transparency, 0.0, 1.0)
                        if changed and obj.vertex_colors is not None:
                            obj.vertex_colors[:, 3] = obj.transparency

                        if imgui.tree_node("Edge Classes"):
                            for i in range(len(obj.edge_classes)):
                                # Fixed width: "insertion" is 9 chars, pad "origin" to match
                                label = f"{obj.edge_classes[i]:9s}"
                                imgui.text(label)
                                imgui.same_line()
                                if imgui.button(f"Flip class##{name}_{i}"):
                                    obj.edge_classes[i] = 'insertion' if obj.edge_classes[i] == 'origin' else 'origin'
                            imgui.tree_pop()
                        if obj.draw_contour_stream is not None:
                            if imgui.tree_node("Contour Stream"):
                                if imgui.button("All Stream Off"):
                                    for i in range(len(obj.draw_contour_stream)):
                                        obj.draw_contour_stream[i] = False
                                imgui.same_line()
                                if imgui.button(f"Auto Detect##{name}"):
                                    obj.auto_detect_attachments(self.zygote_skeleton_meshes)
                                # Ensure attach_skeletons arrays are properly sized
                                num_streams = len(obj.draw_contour_stream)
                                while len(obj.attach_skeletons) < num_streams:
                                    obj.attach_skeletons.append([0, 0])
                                while len(obj.attach_skeletons_sub) < num_streams:
                                    obj.attach_skeletons_sub.append([0, 0])
                                for i in range(num_streams):
                                    _, obj.draw_contour_stream[i] = imgui.checkbox(f"Stream {i}", obj.draw_contour_stream[i])
                                    imgui.push_item_width(100)
                                    changed, obj.attach_skeletons[i][0] = imgui.input_int(f"Origin##{name}_stream{i}_origin", obj.attach_skeletons[i][0])
                                    if changed:
                                        if obj.attach_skeletons[i][0] < 0:
                                            obj.attach_skeletons[i][0] = 0
                                        elif obj.attach_skeletons[i][0] > len(self.zygote_skeleton_meshes) - 1:
                                            obj.attach_skeletons[i][0] = len(self.zygote_skeleton_meshes) - 1
                                    changed, obj.attach_skeletons_sub[i][0] = imgui.input_int(f"Subpart##{name}_stream{i}_origin_sub", obj.attach_skeletons_sub[i][0])
                                    if changed:
                                        if obj.attach_skeletons_sub[i][0] < 0:
                                            obj.attach_skeletons_sub[i][0] = 0
                                        elif obj.attach_skeletons_sub[i][0] > 1:
                                            obj.attach_skeletons_sub[i][0] = 1

                                    imgui.text(list(self.zygote_skeleton_meshes.keys())[obj.attach_skeletons[i][0]] + f"{obj.attach_skeletons_sub[i][0]}")
                                    changed, obj.attach_skeletons[i][1] = imgui.input_int(f"Insertion##{name}_stream{i}_insertion", obj.attach_skeletons[i][1])
                                    if changed:
                                        if obj.attach_skeletons[i][1] < 0:
                                            obj.attach_skeletons[i][1] = 0
                                        elif obj.attach_skeletons[i][1] > len(self.zygote_skeleton_meshes) - 1:
                                            obj.attach_skeletons[i][1] = len(self.zygote_skeleton_meshes) - 1
                                    changed, obj.attach_skeletons_sub[i][1] = imgui.input_int(f"Subpart##{name}_stream{i}_insertion_sub", obj.attach_skeletons_sub[i][1])
                                    if changed:
                                        if obj.attach_skeletons_sub[i][1] < 0:
                                            obj.attach_skeletons_sub[i][1] = 0
                                        elif obj.attach_skeletons_sub[i][1] > 1:
                                            obj.attach_skeletons_sub[i][1] = 1
                                    imgui.text(list(self.zygote_skeleton_meshes.keys())[obj.attach_skeletons[i][1]] + f"{obj.attach_skeletons_sub[i][1]}")
                                    imgui.pop_item_width()

                                    # if imgui.button(f"Print Contour points##{i}"):
                                    #     print(f"Print {i}th contour stream")
                                    #     for j, contour in enumerate(obj.contours[i]):
                                    #         print(f"Contour {j}")
                                    #         for v in contour:
                                    #             print(v)
                                    #         print()
                                imgui.tree_pop()

                        changed, obj.is_draw = imgui.checkbox("Draw", obj.is_draw)
                        if changed and obj.is_draw and self.is_draw_one_zygote_muscle:
                            for other_name, other_obj in self.zygote_muscle_meshes.items():
                                other_obj.is_draw = False
                            obj.is_draw = True
                        _, obj.is_draw_open_edges = imgui.checkbox("Draw Open Edges", obj.is_draw_open_edges)
                        _, obj.is_draw_scalar_field = imgui.checkbox("Draw Scalar Field", obj.is_draw_scalar_field)
                        _, obj.is_draw_contours = imgui.checkbox("Draw Contours", obj.is_draw_contours)
                        imgui.same_line()
                        _, obj.is_draw_contour_vertices = imgui.checkbox("Vertices", obj.is_draw_contour_vertices)
                        imgui.same_line()
                        _, obj.is_draw_farthest_pair = imgui.checkbox("Farthest Pair", obj.is_draw_farthest_pair)
                        _, obj.is_draw_edges = imgui.checkbox("Draw Edges", obj.is_draw_edges)
                        _, obj.is_draw_centroid = imgui.checkbox("Draw Centroid", obj.is_draw_centroid)
                        _, obj.is_draw_bounding_box = imgui.checkbox("Draw Bounding Box", obj.is_draw_bounding_box)
                        _, obj.is_draw_discarded = imgui.checkbox("Draw Discarded", obj.is_draw_discarded)
                        _, obj.is_draw_fiber_architecture = imgui.checkbox("Draw Fiber Architecture", obj.is_draw_fiber_architecture)
                        _, obj.is_draw_contour_mesh = imgui.checkbox("Draw Contour Mesh", obj.is_draw_contour_mesh)
                        _, obj.is_draw_tet_mesh = imgui.checkbox("Draw Tet Mesh", obj.is_draw_tet_mesh)
                        imgui.same_line()
                        _, obj.is_draw_tet_edges = imgui.checkbox("Tet Edges", obj.is_draw_tet_edges)
                        imgui.same_line()
                        _, obj.is_draw_constraints = imgui.checkbox("Constraints", obj.is_draw_constraints)

                        # Soft body simulation controls (quasistatic, on-demand)
                        if imgui.tree_node(f"Soft Body##{name}"):
                            # Initialize soft body if tet mesh exists but soft body doesn't
                            if obj.soft_body is None and obj.tet_vertices is not None:
                                if imgui.button(f"Init Soft Body##{name}", width=wide_button_width):
                                    try:
                                        obj.init_soft_body(self.zygote_skeleton_meshes, self.env.skel, self.env.mesh_info)
                                    except Exception as e:
                                        print(f"[{name}] Init Soft Body error: {e}")
                            elif obj.tet_vertices is not None:
                                # Parameters: Volume is primary (muscle), Edge is secondary
                                _, obj.soft_body_volume_stiffness = imgui.slider_float(f"Volume##soft_{name}", obj.soft_body_volume_stiffness, 0.1, 1.0)
                                _, obj.soft_body_stiffness = imgui.slider_float(f"Edge##soft_{name}", obj.soft_body_stiffness, 0.0, 1.0)
                                _, obj.soft_body_damping = imgui.slider_float(f"Damping##soft_{name}", obj.soft_body_damping, 0.0, 0.95)
                                _, obj.soft_body_collision = imgui.checkbox(f"Collision##soft_{name}", obj.soft_body_collision)
                                _, obj.waypoints_from_tet_sim = imgui.checkbox(f"Update Waypoints##soft_{name}", obj.waypoints_from_tet_sim)
                                if obj.soft_body_collision:
                                    imgui.same_line()
                                    imgui.push_item_width(80)
                                    _, obj.soft_body_collision_margin = imgui.slider_float(f"Margin##soft_{name}", obj.soft_body_collision_margin, 0.001, 0.02)
                                    imgui.pop_item_width()

                                # Run simulation button
                                if imgui.button(f"Run Tet Sim##{name}", width=wide_button_width):
                                    try:
                                        iterations, residual = obj.run_soft_body_to_convergence(
                                            self.zygote_skeleton_meshes,
                                            self.env.skel,
                                            max_iterations=100,
                                            tolerance=1e-4,
                                            enable_collision=obj.soft_body_collision,
                                            collision_margin=obj.soft_body_collision_margin,
                                            use_arap=obj.use_arap
                                        )
                                        print(f"{name}: Converged in {iterations} iterations, residual={residual:.2e} (ARAP={obj.use_arap})")
                                    except Exception as e:
                                        print(f"[{name}] Run Tet Sim error: {e}")

                                # Test button to verify deformation works
                                if imgui.button(f"Test Deform##{name}", width=wide_button_width):
                                    try:
                                        if obj.soft_body is not None and obj.soft_body.fixed_targets is not None:
                                            test_offset = np.array([0.02, 0.0, 0.0])  # 2cm in X
                                            obj.soft_body.fixed_targets += test_offset
                                            iterations, residual = obj.soft_body.solve_to_convergence(100, 1e-4)
                                            obj.tet_vertices = obj.soft_body.get_positions().astype(np.float32)
                                            obj._prepare_tet_draw_arrays()
                                            print(f"{name}: Test deform - {iterations} iters, residual={residual:.2e}")
                                    except Exception as e:
                                        print(f"[{name}] Test Deform error: {e}")

                                if imgui.button(f"Reset Soft Body##{name}", width=wide_button_width):
                                    try:
                                        obj.reset_soft_body()
                                    except Exception as e:
                                        print(f"[{name}] Reset Soft Body error: {e}")

                            imgui.tree_pop()

                        # VIPER rod simulation controls
                        if imgui.tree_node(f"VIPER Rods##{name}"):
                            if obj.viper_available:
                                # Initialize VIPER if waypoints exist but VIPER doesn't
                                if obj.viper_sim is None and len(obj.waypoints) > 0:
                                    if imgui.button(f"Init VIPER##{name}", width=wide_button_width):
                                        try:
                                            obj.init_viper(self.zygote_skeleton_meshes, self.env.skel)
                                        except Exception as e:
                                            print(f"[{name}] Init VIPER error: {e}")
                                elif obj.viper_sim is not None:
                                    # VIPER parameters
                                    changed, obj.viper_sim.stretch_stiffness = imgui.slider_float(
                                        f"Stretch##viper_{name}", obj.viper_sim.stretch_stiffness, 0.1, 1.0)
                                    changed, obj.viper_sim.volume_stiffness = imgui.slider_float(
                                        f"Volume##viper_{name}", obj.viper_sim.volume_stiffness, 0.1, 1.0)
                                    changed, obj.viper_sim.bend_stiffness = imgui.slider_float(
                                        f"Bend##viper_{name}", obj.viper_sim.bend_stiffness, 0.1, 1.0)
                                    changed, obj.viper_sim.damping = imgui.slider_float(
                                        f"Damping##viper_{name}", obj.viper_sim.damping, 0.8, 0.999)
                                    changed, obj.viper_sim.iterations = imgui.slider_int(
                                        f"Iterations##viper_{name}", obj.viper_sim.iterations, 1, 50)

                                    # Volume preservation toggle (VIPER key feature)
                                    changed, obj.viper_sim.enable_volume_constraint = imgui.checkbox(
                                        f"Volume Preserve##viper_{name}", obj.viper_sim.enable_volume_constraint)

                                    # Collision toggle
                                    _, obj.viper_sim.enable_collision = imgui.checkbox(
                                        f"Collision##viper_{name}", obj.viper_sim.enable_collision)
                                    if obj.viper_sim.enable_collision:
                                        imgui.same_line()
                                        imgui.push_item_width(80)
                                        _, obj.viper_sim.collision_margin = imgui.slider_float(
                                            f"Margin##viper_col_{name}", obj.viper_sim.collision_margin, 0.001, 0.01, "%.3f")
                                        imgui.pop_item_width()

                                    imgui.separator()
                                    imgui.text("Visualization:")
                                    # Visualization controls
                                    _, obj.is_draw_viper = imgui.checkbox(
                                        f"Draw Rods##viper_{name}", obj.is_draw_viper)
                                    imgui.same_line()
                                    _, obj.viper_only_mode = imgui.checkbox(
                                        f"VIPER Only##viper_{name}", obj.viper_only_mode)
                                    _, obj.is_draw_viper_tubes = imgui.checkbox(
                                        f"Tube Mode##viper_{name}", obj.is_draw_viper_tubes)
                                    # Show VIPER rod-based mesh (built from rods)
                                    _, obj.is_draw_viper_rod_mesh = imgui.checkbox(
                                        f"Rod Mesh##viper_{name}", getattr(obj, 'is_draw_viper_rod_mesh', False))
                                    imgui.push_item_width(100)
                                    _, obj.viper_rod_radius = imgui.slider_float(
                                        f"Rod Radius##viper_{name}", obj.viper_rod_radius, 0.001, 0.01, "%.3f")
                                    _, obj.viper_point_size = imgui.slider_float(
                                        f"Point Size##viper_{name}", obj.viper_point_size, 2.0, 15.0)
                                    imgui.pop_item_width()

                                    imgui.separator()
                                    # Show rod info
                                    if hasattr(obj, 'viper_waypoints') and obj.viper_waypoints:
                                        num_rods = len(obj.viper_waypoints)
                                        total_pts = sum(len(rod) for rod in obj.viper_waypoints)
                                    else:
                                        num_rods = 0
                                        total_pts = 0
                                    imgui.text(f"Rods: {num_rods}, Points: {total_pts}")

                                    # Run VIPER simulation button
                                    if imgui.button(f"Run VIPER##{name}", width=wide_button_width):
                                        try:
                                            skel = self.env.skel if hasattr(self, 'env') and self.env is not None else None
                                            skel_meshes = self.zygote_skeleton_meshes if obj.viper_sim.enable_collision else None
                                            iterations = obj.run_viper_to_convergence(max_iterations=100, tolerance=1e-5, skeleton=skel, skeleton_meshes=skel_meshes)
                                            print(f"{name}: VIPER converged in {iterations} iterations")
                                        except Exception as e:
                                            print(f"[{name}] Run VIPER error: {e}")

                                    # Single step button
                                    if imgui.button(f"VIPER Step##{name}", width=wide_button_width):
                                        try:
                                            skel = self.env.skel if hasattr(self, 'env') and self.env is not None else None
                                            skel_meshes = self.zygote_skeleton_meshes if obj.viper_sim.enable_collision else None
                                            max_disp = obj.run_viper_step(skeleton=skel, skeleton_meshes=skel_meshes)
                                            print(f"{name}: VIPER step, max_disp={max_disp:.2e}")
                                        except Exception as e:
                                            print(f"[{name}] VIPER Step error: {e}")

                                    # Reset button
                                    if imgui.button(f"Reset VIPER##{name}", width=wide_button_width):
                                        try:
                                            obj.reset_viper()
                                            print(f"{name}: VIPER reset")
                                        except Exception as e:
                                            print(f"[{name}] Reset VIPER error: {e}")

                                    # Debug button - show constraint residuals
                                    if imgui.button(f"Debug Info##{name}", width=wide_button_width):
                                        try:
                                            from viewer.viper_rods import get_viper_backend
                                            info = obj.viper_sim.get_debug_info()
                                            residuals = obj.viper_sim.get_constraint_residuals()
                                            backend = get_viper_backend()
                                            print(f"\n=== {name} VIPER Debug ===")
                                            print(f"  Backend: {backend.upper() if backend else 'Not initialized'} (use_gpu={info.get('use_gpu', False)})")
                                            print(f"  Rods: {info.get('num_rods', 0)}, Verts/rod: {info.get('num_vertices_per_rod', 0)}")
                                            print(f"  Scale: min={info.get('min_scale', 0):.3f}, max={info.get('max_scale', 0):.3f}, avg={info.get('avg_scale', 0):.3f}")
                                            print(f"  Stretch error: mean={residuals.get('stretch_error_mean', 0):.4f}, max={residuals.get('stretch_error_max', 0):.4f}")
                                            print(f"  Volume error: mean={residuals.get('volume_error_mean', 0):.4f}, max={residuals.get('volume_error_max', 0):.4f}")
                                            print(f"  Bend error: mean={residuals.get('bend_error_mean', 0):.4f}, max={residuals.get('bend_error_max', 0):.4f}")
                                            print(f"  Collision: {obj.viper_sim.enable_collision} (margin={obj.viper_sim.collision_margin:.3f})")
                                        except Exception as e:
                                            print(f"[{name}] Debug error: {e}")
                                else:
                                    imgui.text("Generate waypoints first")
                            else:
                                imgui.text("VIPER requires Taichi")
                            imgui.tree_pop()

                        if imgui.button("Export Muscle Waypoints", width=wide_button_width):
                            pass
                            from core.dartHelper import exportMuscleWaypoints
                            exportMuscleWaypoints(self.zygote_muscle_meshes, list(self.zygote_skeleton_meshes.keys()))
                        if imgui.button("Import zygote_muscle", width=wide_button_width):
                            import os
                            muscle_file = "data/zygote_muscle.xml"
                            if not os.path.exists(muscle_file):
                                print(f"Error: Muscle file not found: {muscle_file}")
                                print("  Run 'Export Muscle Waypoints' first to create it.")
                            else:
                                try:
                                    self.env.muscle_info = self.env.saveZygoteMuscleInfo(muscle_file)
                                    if not self.env.muscle_info:
                                        print("No muscles loaded from file (empty or invalid)")
                                    else:
                                        self.env.loading_zygote_muscle_info(self.env.muscle_info)
                                        self.env.muscle_activation_levels = np.zeros(self.env.muscles.getNumMuscles())

                                        self.draw_obj = True
                                        # Disable skeleton drawing when importing muscle waypoints
                                        self.is_draw_zygote_skeleton = False
                                        for sname, sobj in self.zygote_skeleton_meshes.items():
                                            sobj.is_draw = False
                                        print(f"Imported {self.env.muscles.getNumMuscles()} muscles from {muscle_file}")
                                except Exception as e:
                                    print(f"Error importing muscle waypoints: {e}")

                        # End column layout
                        imgui.columns(1)
                        imgui.tree_pop()
                imgui.tree_pop()

            if imgui.tree_node("Skeleton"):
                changed, self.is_draw_zygote_skeleton = imgui.checkbox("Draw", self.is_draw_zygote_skeleton)
                if changed:
                    for name, obj in self.zygote_skeleton_meshes.items():
                        obj.is_draw = self.is_draw_zygote_skeleton
                _, self.is_draw_one_zygote_skeleton = imgui.checkbox("Draw One Skeleton", self.is_draw_one_zygote_skeleton)
                changed, self.zygote_skeleton_color = imgui.color_edit3("Color", *self.zygote_skeleton_color)
                if changed:
                    for name, obj in self.zygote_skeleton_meshes.items():
                        obj.color = self.zygote_skeleton_color
                if imgui.button("Draw All"):
                    for name, obj in self.zygote_skeleton_meshes.items():
                        obj.is_draw = True
                changed, self.zygote_skeleton_transparency = imgui.slider_float("Transparency##Skeleton", self.zygote_skeleton_transparency, 0.0, 1.0)
                if changed:
                    for name, obj in self.zygote_skeleton_meshes.items():
                        obj.transparency = self.zygote_skeleton_transparency
                    
                for i, (name, obj) in enumerate(self.zygote_skeleton_meshes.items()):
                    if imgui.tree_node(f"{i}: {name}"):
                        changed, obj.transparency = imgui.slider_float(f"Transparency##{name}", obj.transparency, 0.0, 1.0)
                        if changed and obj.vertex_colors is not None:
                            obj.vertex_colors[:, 3] = obj.transparency
                        _, obj.is_draw = imgui.checkbox("Draw", obj.is_draw)
                        _, obj.is_draw_corners = imgui.checkbox("Draw Corners", obj.is_draw_corners)
                        _, obj.is_draw_edges = imgui.checkbox("Draw Edges", obj.is_draw_edges)
                        _, obj.is_contact = imgui.checkbox("Contact", obj.is_contact)
                        if obj.is_draw and self.is_draw_one_zygote_skeleton:
                            for other_name, other_obj in self.zygote_skeleton_meshes.items():
                                other_obj.is_draw = False
                                obj.is_draw = True
                        imgui.tree_pop()

                        # Auto checkbox and num boxes slider
                        _, obj.auto_num_boxes = imgui.checkbox(f"Auto##{name}_auto", obj.auto_num_boxes)
                        imgui.same_line()
                        if obj.auto_num_boxes:
                            imgui.text(f"Boxes: {obj.num_boxes} (auto)")
                        else:
                            imgui.push_item_width(100)
                            _, obj.num_boxes = imgui.slider_int(f"Boxes##{name}", obj.num_boxes, 1, 10)
                            imgui.pop_item_width()

                        # Axis alignment checkboxes
                        imgui.text("Align:")
                        imgui.same_line()
                        _, obj.bb_align_x = imgui.checkbox(f"X##{name}_bbx", obj.bb_align_x)
                        imgui.same_line()
                        _, obj.bb_align_y = imgui.checkbox(f"Y##{name}_bby", obj.bb_align_y)
                        imgui.same_line()
                        _, obj.bb_align_z = imgui.checkbox(f"Z##{name}_bbz", obj.bb_align_z)
                        imgui.same_line()
                        _, obj.bb_enforce_symmetry = imgui.checkbox(f"Sym##{name}_sym", obj.bb_enforce_symmetry)

                        # Build axis string from checkboxes
                        axis_str = ''
                        if obj.bb_align_x:
                            axis_str += 'x'
                        if obj.bb_align_y:
                            axis_str += 'y'
                        if obj.bb_align_z:
                            axis_str += 'z'
                        axis_param = axis_str if axis_str else None

                        # Show current alignment
                        align_label = axis_str.upper() if axis_str else "PCA"
                        sym_label = "+Sym" if obj.bb_enforce_symmetry else ""
                        auto_label = "+Auto" if obj.auto_num_boxes else ""
                        if imgui.button(f"Find BB ({align_label}{sym_label}{auto_label})##{name}", width=140):
                            obj.find_bounding_box(axis=axis_param, symmetry=obj.bb_enforce_symmetry)

                        imgui.text(f"Parent: {obj.parent_name}")
                        if imgui.button(f"<##{name+'_parent'}"):
                            obj.cand_parent_index -= 1
                            if obj.cand_parent_index < 0:
                                obj.cand_parent_index = len(self.zygote_skeleton_meshes) - 1
                        imgui.same_line()
                        if imgui.button(f">##{name+'_parent'}"):
                            obj.cand_parent_index += 1
                            if obj.cand_parent_index >= len(self.zygote_skeleton_meshes):
                                obj.cand_parent_index = 0
                        imgui.same_line()
                        cand_name = list(self.zygote_skeleton_meshes.keys())[obj.cand_parent_index]
                        imgui.push_item_width(100)
                        changed, obj.cand_parent_index = imgui.input_int(f"Parent##{name}", obj.cand_parent_index)
                        imgui.pop_item_width()
                        if changed:
                            if obj.cand_parent_index > len(self.zygote_skeleton_meshes) - 1:
                                obj.cand_parent_index = len(self.zygote_skeleton_meshes) - 1
                            elif obj.cand_parent_index < 0:
                                obj.cand_parent_index = 0
                        imgui.text("%3d: %s   " % (obj.cand_parent_index, cand_name))
                        
                        if imgui.button(f"Set as root##{name}"):
                            for other_name, other_obj in self.zygote_skeleton_meshes.items():
                                other_obj.is_root = False
                            obj.is_root = True
                            print(f"{name} set as root")
                        if imgui.button(f"Connect to parent##{name}"):

                            if self.zygote_skeleton_meshes[cand_name].corners is None:
                                print("First find bounding boxes for parent mesh")
                            elif obj.corners is None:
                                print("First find bounding boxes for this mesh")
                            elif name == cand_name:
                                print("Self connection")
                            else:
                                parent_mesh = self.zygote_skeleton_meshes[cand_name]
                                parent_corners_list = parent_mesh.corners_list
                                cand_joints = []
                                for parent_corners in parent_corners_list:
                                    for corners in obj.corners_list:
                                        cand_joint = obj.find_overlap_point(parent_corners, corners)
                                        if cand_joint is not None: 
                                            cand_joints.append(cand_joint)
                                
                                if len(cand_joints) == 0:
                                    print("They can't be linked; No overlapping points")
                                else:
                                    obj.parent_mesh = parent_mesh
                                    obj.parent_name = cand_name
                                    if not name in parent_mesh.children_names:
                                        parent_mesh.children_names.append(name)
                                    print(f"{name} connected to {cand_name}")
                                    if len(cand_joints) == 1:
                                        obj.joint_to_parent = cand_joints[0]
                                    else:
                                        mean = np.mean(obj.vertices)
                                        distances = np.linalg.norm(cand_joints - mean, axis=1)
                                        obj.joint_to_parent = cand_joints[np.argmax(distances)]
                                    obj.is_weld = False
                        imgui.same_line()
                        if imgui.button(f"Connect to parent as Weld##{name}"):
                            if self.zygote_skeleton_meshes[cand_name].corners is None:
                                print("First find bounding boxes for parent mesh")
                            elif obj.corners is None:
                                print("First find bounding boxes for this mesh")
                            elif name == cand_name:
                                print("Self connection")
                            else:
                                parent_mesh = self.zygote_skeleton_meshes[cand_name]
                                parent_corners_list = parent_mesh.corners_list
                                cand_joints = []
                                for parent_corners in parent_corners_list:
                                    for corners in obj.corners_list:
                                        cand_joint = obj.find_overlap_point(parent_corners, corners)
                                        if cand_joint is not None: 
                                            cand_joints.append(cand_joint)
                                
                                if len(cand_joints) == 0:
                                    print("They can't be linked; No overlapping points")
                                else:
                                    obj.parent_mesh = parent_mesh
                                    obj.parent_name = cand_name
                                    parent_mesh.children_names.append(name)
                                    print(f"{name} connected to {cand_name} as Weld")
                                    if len(cand_joints) == 1:
                                        obj.joint_to_parent = cand_joints[0]
                                    else:
                                        mean = np.mean(obj.vertices)
                                        distances = np.linalg.norm(cand_joints - mean, axis=1)
                                        obj.joint_to_parent = cand_joints[np.argmax(distances)]
                                    obj.is_weld = True

                        # Revolute joint connection buttons
                        if imgui.button(f"Revolute (Knee)##{name}"):
                            # Knee: bends backward, axis = x
                            if self.zygote_skeleton_meshes[cand_name].corners is None:
                                print("First find bounding boxes for parent mesh")
                            elif obj.corners is None:
                                print("First find bounding boxes for this mesh")
                            elif name == cand_name:
                                print("Self connection")
                            else:
                                parent_mesh = self.zygote_skeleton_meshes[cand_name]
                                parent_corners_list = parent_mesh.corners_list
                                cand_joints = []
                                for parent_corners in parent_corners_list:
                                    for corners in obj.corners_list:
                                        cand_joint = obj.find_overlap_point(parent_corners, corners)
                                        if cand_joint is not None:
                                            cand_joints.append(cand_joint)

                                if len(cand_joints) == 0:
                                    print("They can't be linked; No overlapping points")
                                else:
                                    obj.parent_mesh = parent_mesh
                                    obj.parent_name = cand_name
                                    if name not in parent_mesh.children_names:
                                        parent_mesh.children_names.append(name)
                                    if len(cand_joints) == 1:
                                        obj.joint_to_parent = cand_joints[0]
                                    else:
                                        mean = np.mean(obj.vertices)
                                        distances = np.linalg.norm(cand_joints - mean, axis=1)
                                        obj.joint_to_parent = cand_joints[np.argmax(distances)]
                                    obj.is_weld = False
                                    obj.is_revolute = True
                                    obj.bends_backward = True  # Knee bends backward
                                    obj.revolute_axis = np.array([1.0, 0.0, 0.0])
                                    obj.revolute_lower = 0.0
                                    obj.revolute_upper = 2.5
                                    print(f"{name} connected to {cand_name} as Revolute (Knee, bends backward)")

                        imgui.same_line()
                        if imgui.button(f"Revolute (Elbow)##{name}"):
                            # Elbow/Finger/Toe: bends forward, axis = x
                            if self.zygote_skeleton_meshes[cand_name].corners is None:
                                print("First find bounding boxes for parent mesh")
                            elif obj.corners is None:
                                print("First find bounding boxes for this mesh")
                            elif name == cand_name:
                                print("Self connection")
                            else:
                                parent_mesh = self.zygote_skeleton_meshes[cand_name]
                                parent_corners_list = parent_mesh.corners_list
                                cand_joints = []
                                for parent_corners in parent_corners_list:
                                    for corners in obj.corners_list:
                                        cand_joint = obj.find_overlap_point(parent_corners, corners)
                                        if cand_joint is not None:
                                            cand_joints.append(cand_joint)

                                if len(cand_joints) == 0:
                                    print("They can't be linked; No overlapping points")
                                else:
                                    obj.parent_mesh = parent_mesh
                                    obj.parent_name = cand_name
                                    if name not in parent_mesh.children_names:
                                        parent_mesh.children_names.append(name)
                                    if len(cand_joints) == 1:
                                        obj.joint_to_parent = cand_joints[0]
                                    else:
                                        mean = np.mean(obj.vertices)
                                        distances = np.linalg.norm(cand_joints - mean, axis=1)
                                        obj.joint_to_parent = cand_joints[np.argmax(distances)]
                                    obj.is_weld = False
                                    obj.is_revolute = True
                                    obj.bends_backward = False  # Elbow/finger/toe bends forward
                                    obj.revolute_axis = np.array([1.0, 0.0, 0.0])
                                    obj.revolute_lower = -2.5
                                    obj.revolute_upper = 0.0
                                    print(f"{name} connected to {cand_name} as Revolute (Elbow/Finger/Toe, bends forward)")

                        if imgui.button(f"Revolute (Finger/Toe)##{name}"):
                            # Finger/Toe: smaller range than elbow
                            if self.zygote_skeleton_meshes[cand_name].corners is None:
                                print("First find bounding boxes for parent mesh")
                            elif obj.corners is None:
                                print("First find bounding boxes for this mesh")
                            elif name == cand_name:
                                print("Self connection")
                            else:
                                parent_mesh = self.zygote_skeleton_meshes[cand_name]
                                parent_corners_list = parent_mesh.corners_list
                                cand_joints = []
                                for parent_corners in parent_corners_list:
                                    for corners in obj.corners_list:
                                        cand_joint = obj.find_overlap_point(parent_corners, corners)
                                        if cand_joint is not None:
                                            cand_joints.append(cand_joint)

                                if len(cand_joints) == 0:
                                    print("They can't be linked; No overlapping points")
                                else:
                                    obj.parent_mesh = parent_mesh
                                    obj.parent_name = cand_name
                                    if name not in parent_mesh.children_names:
                                        parent_mesh.children_names.append(name)
                                    if len(cand_joints) == 1:
                                        obj.joint_to_parent = cand_joints[0]
                                    else:
                                        mean = np.mean(obj.vertices)
                                        distances = np.linalg.norm(cand_joints - mean, axis=1)
                                        obj.joint_to_parent = cand_joints[np.argmax(distances)]
                                    obj.is_weld = False
                                    obj.is_revolute = True
                                    obj.bends_backward = False
                                    obj.revolute_axis = np.array([1.0, 0.0, 0.0])
                                    obj.revolute_lower = -1.57  # ~90 degrees for fingers/toes
                                    obj.revolute_upper = 0.3    # Slight hyperextension allowed
                                    print(f"{name} connected to {cand_name} as Revolute (Finger/Toe)")

                        # Show revolute joint settings if connected as revolute
                        if obj.is_revolute:
                            imgui.text(f"Revolute: {'Knee (backward)' if obj.bends_backward else 'Elbow (forward)'}")
                            imgui.push_item_width(80)
                            changed, obj.revolute_lower = imgui.input_float(f"Lower##{name}_rev", obj.revolute_lower, 0.1)
                            imgui.same_line()
                            changed, obj.revolute_upper = imgui.input_float(f"Upper##{name}_rev", obj.revolute_upper, 0.1)
                            imgui.pop_item_width()

                        if imgui.button("Export Bounding boxes", width=wide_button_width):
                            from core.dartHelper import exportBoundingBoxes
                            exportBoundingBoxes(self.zygote_skeleton_meshes)
                        if imgui.button("Import zygote_skel", width=wide_button_width):
                            if self.env.skel is not None:
                                self.env.world.removeSkeleton(self.env.skel)

                            from core.dartHelper import saveSkeletonInfo
                            skel_info, root_name, _, _, _, _ = saveSkeletonInfo("data/zygote_skel.xml")
                            self.env.skel_info = skel_info
                            self.env.skel = buildFromInfo(skel_info, "zygote")
                            self.env.world.addSkeleton(self.env.skel)
                            self.env.kp = 300.0 * np.ones(self.env.skel.getNumDofs())
                            self.env.kv = 20.0 * np.ones(self.env.skel.getNumDofs())
                            self.env.kp[:6] = 0.0
                            self.env.kv[:6] = 0.0
                            self.env.num_action = len(self.env.get_zero_action()) * (3 if self.env.learning_gain else 1)
                            self.motion_skel = self.env.skel.clone()
                    
                imgui.tree_pop()
            imgui.end_child()  # End ZygoteScroll
            imgui.tree_pop()

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
        self._render_inspect_2d_windows()

        # Render Neck Viz windows (narrowest neck visualization)
        self._render_neck_viz_windows()

        # Render Manual Cut windows for each muscle
        self._render_manual_cut_windows()

        # Render Level Select windows for manual level selection
        self._render_level_select_windows()

        imgui.render()

    def _render_inspect_2d_windows(self):
        """Render 2D inspection windows for fiber samples and contour waypoints."""
        muscles_to_close = []

        for name, is_open in list(self.inspect_2d_open.items()):
            if not is_open:
                continue

            if name not in self.zygote_muscle_meshes:
                muscles_to_close.append(name)
                continue

            obj = self.zygote_muscle_meshes[name]

            # Check if data is available (only contours required, fiber_architecture optional)
            if (not hasattr(obj, 'contours') or obj.contours is None or len(obj.contours) == 0):
                muscles_to_close.append(name)
                continue

            # Check if fiber_architecture is available (optional - for fiber sample display)
            has_fiber = (hasattr(obj, 'fiber_architecture') and obj.fiber_architecture is not None and
                        len(obj.fiber_architecture) > 0)

            # Window setup - size to fit two 280px canvases with padding
            # Width: 2 * (280 + 2*20 + 20) + window padding = ~720
            # Height: sliders(~75) + labels(~40) + canvas(280+2*20) + margins = ~450
            imgui.set_next_window_size(720, 480, imgui.FIRST_USE_EVER)
            expanded, opened = imgui.begin(f"Inspect 2D: {name}", True)

            if not opened:
                muscles_to_close.append(name)
                imgui.end()
                continue

            # Detect data structure format:
            # - Pre-stream (before cutting): contours[level_idx][stream_idx], bounding_planes[level_idx][stream_idx]
            # - Post-stream (after build_fibers): contours[stream_idx][level_idx], bounding_planes[stream_idx][level_idx]
            #
            # Detection: If stream_contours exists and has data, we're in post-stream mode.
            # Also check if contours structure matches stream_contours (indicating build_fibers was called).
            has_stream_contours = (hasattr(obj, 'stream_contours') and
                                   obj.stream_contours is not None and
                                   len(obj.stream_contours) > 0)

            # If contours == stream_contours (same object or same structure), it's post-stream
            is_post_stream = False
            if has_stream_contours:
                # Check if contours is the same as stream_contours (build_fibers assigns directly)
                if obj.contours is obj.stream_contours:
                    is_post_stream = True
                # Also check if structure matches: outer dim is small (num streams), inner is larger (num levels)
                elif len(obj.contours) > 0 and len(obj.contours) <= 10:  # Typically few streams
                    # In post-stream, contours[stream][level], so inner should have many elements
                    if isinstance(obj.contours[0], (list, np.ndarray)) and len(obj.contours[0]) > 0:
                        # Check if inner elements are contour arrays (have 3D points)
                        inner = obj.contours[0][0]
                        if isinstance(inner, np.ndarray) and inner.ndim == 2 and inner.shape[1] == 3:
                            is_post_stream = True

            is_pre_stream = not is_post_stream

            # Helper functions to access data in correct format
            def get_num_levels():
                """Get number of levels."""
                if is_pre_stream:
                    return len(obj.contours)
                else:
                    # Post-stream: levels are the inner index
                    if len(obj.contours) > 0:
                        return len(obj.contours[0])
                    return 0

            def get_max_contours_at_level():
                """Get maximum number of contours at any level (for pre-stream)."""
                if is_pre_stream:
                    if len(obj.contours) == 0:
                        return 0
                    return max(len(level) for level in obj.contours)
                else:
                    # Post-stream: streams are the outer index
                    return len(obj.contours)

            def get_num_contours_at_level(level_idx):
                """Get number of contours at a specific level (for pre-stream)."""
                if is_pre_stream:
                    if level_idx < len(obj.contours):
                        return len(obj.contours[level_idx])
                    return 0
                else:
                    return len(obj.contours)

            def get_contour(s_idx, level_idx):
                if is_pre_stream:
                    # contours[level_idx][contour_idx] - s_idx is contour within level
                    if level_idx < len(obj.contours) and s_idx < len(obj.contours[level_idx]):
                        return obj.contours[level_idx][s_idx]
                    return None
                else:
                    # contours[stream_idx][level_idx]
                    if s_idx < len(obj.contours) and level_idx < len(obj.contours[s_idx]):
                        return obj.contours[s_idx][level_idx]
                    return None

            def get_bounding_plane(s_idx, level_idx):
                if is_pre_stream:
                    # bounding_planes[level_idx][contour_idx] - s_idx is contour within level
                    if level_idx < len(obj.bounding_planes) and s_idx < len(obj.bounding_planes[level_idx]):
                        return obj.bounding_planes[level_idx][s_idx]
                    return None
                else:
                    # bounding_planes[stream_idx][level_idx]
                    if s_idx < len(obj.bounding_planes) and level_idx < len(obj.bounding_planes[s_idx]):
                        return obj.bounding_planes[s_idx][level_idx]
                    return None

            # Different UI for pre-stream vs post-stream
            if is_pre_stream:
                # Pre-stream: Level slider (contour) + Contour-within-level slider (stream)
                num_levels = get_num_levels()
                max_contours = get_max_contours_at_level()

                # Level slider (labeled as "Level")
                level_idx = self.inspect_2d_contour_idx.get(name, 0)
                level_idx = min(level_idx, max(0, num_levels - 1))

                changed, new_level_idx = imgui.slider_int(f"Level##{name}_inspect", level_idx, 0, max(0, num_levels - 1))
                if changed:
                    self.inspect_2d_contour_idx[name] = new_level_idx
                    level_idx = new_level_idx

                # Get number of contours at current level for slider boundary
                num_at_level = get_num_contours_at_level(level_idx)

                # Contour-within-level slider (labeled as "Contour")
                # Slider max is based on contours at this level, not global max
                contour_in_level = self.inspect_2d_stream_idx.get(name, 0)
                contour_in_level = min(contour_in_level, max(0, num_at_level - 1))

                changed, new_contour_in_level = imgui.slider_int(f"Contour##{name}_inspect", contour_in_level, 0, max(0, num_at_level - 1))
                if changed:
                    self.inspect_2d_stream_idx[name] = new_contour_in_level
                    contour_in_level = new_contour_in_level

                # Show info about contours at this level
                imgui.text(f"Level {level_idx}: {num_at_level} contour(s)")

                imgui.separator()

                # Determine if we should show anything
                # Show nothing if contour_in_level >= num_at_level
                if contour_in_level < num_at_level:
                    contour_indices = [(contour_in_level, level_idx)]  # (stream_idx, level_idx)
                else:
                    contour_indices = []  # Show nothing

                # Pre-stream doesn't have show_all feature
                show_all = False

            else:
                # Post-stream: Stream slider + Contour (level) slider
                num_streams = get_max_contours_at_level()
                stream_idx = self.inspect_2d_stream_idx.get(name, 0)
                stream_idx = min(stream_idx, max(0, num_streams - 1))

                changed, new_stream_idx = imgui.slider_int(f"Stream##{name}_inspect", stream_idx, 0, max(0, num_streams - 1))
                if changed:
                    self.inspect_2d_stream_idx[name] = new_stream_idx
                    stream_idx = new_stream_idx

                num_levels = get_num_levels()
                contour_idx = self.inspect_2d_contour_idx.get(name, 0)
                contour_idx = min(contour_idx, max(0, num_levels - 1))

                # Show All checkbox
                if not hasattr(self, 'inspect_2d_show_all'):
                    self.inspect_2d_show_all = {}
                show_all = self.inspect_2d_show_all.get(name, False)
                changed_show_all, show_all = imgui.checkbox(f"Show All##{name}", show_all)
                if changed_show_all:
                    self.inspect_2d_show_all[name] = show_all
                    # Exit Edit Fiber mode when switching to "Show All"
                    if show_all and name in self.inspect_2d_edit_fiber_mode and self.inspect_2d_edit_fiber_mode[name]:
                        self.inspect_2d_edit_fiber_mode[name] = False
                        self.inspect_2d_edit_fiber_selected[name] = -1
                        self.inspect_2d_edit_fiber_preview[name] = None
                        self.inspect_2d_edit_fiber_test[name] = None
                        # Clear test fiber from object
                        obj.test_fiber_waypoints = None
                        obj.test_fiber_stream_idx = None

                if not show_all:
                    changed, new_contour_idx = imgui.slider_int(f"Contour##{name}_inspect", contour_idx, 0, max(0, num_levels - 1))
                    if changed:
                        self.inspect_2d_contour_idx[name] = new_contour_idx
                        contour_idx = new_contour_idx

                imgui.separator()

                # Determine which contours to draw
                if show_all:
                    contour_indices = [(stream_idx, i) for i in range(num_levels)]
                else:
                    contour_indices = [(stream_idx, contour_idx)]

            # Always use child region for consistent layout (scrollbar space reserved)
            imgui.begin_child(f"contours_scroll##{name}", 0, 0, border=False)

            # Initialize correspondence mode state for this muscle
            if name not in self.inspect_2d_corr_mode:
                self.inspect_2d_corr_mode[name] = False
                self.inspect_2d_corr_corner[name] = -1
                self.inspect_2d_corr_vertex[name] = -1

            corr_mode = self.inspect_2d_corr_mode[name]
            corr_corner = self.inspect_2d_corr_corner[name]
            corr_vertex = self.inspect_2d_corr_vertex[name]

            # Initialize edit fiber mode state for this muscle
            if name not in self.inspect_2d_edit_fiber_mode:
                self.inspect_2d_edit_fiber_mode[name] = False
                self.inspect_2d_edit_fiber_selected[name] = -1
                self.inspect_2d_edit_fiber_preview[name] = None
                self.inspect_2d_edit_fiber_test[name] = None

            edit_fiber_mode = self.inspect_2d_edit_fiber_mode[name]
            edit_fiber_selected = self.inspect_2d_edit_fiber_selected[name]
            edit_fiber_preview = self.inspect_2d_edit_fiber_preview[name]
            edit_fiber_test = self.inspect_2d_edit_fiber_test[name]

            # Set inspector highlight on the object for 3D visualization
            # This will highlight the contour being inspected in the 3D view
            if is_post_stream:
                highlight_stream = self.inspect_2d_stream_idx.get(name, 0)
                highlight_level = self.inspect_2d_contour_idx.get(name, 0)
            else:
                highlight_stream = self.inspect_2d_stream_idx.get(name, 0)
                highlight_level = self.inspect_2d_contour_idx.get(name, 0)
            obj.inspector_highlight_stream = highlight_stream
            obj.inspector_highlight_level = highlight_level

            # Initialize hover state (may not be set if contour_indices is empty)
            hovered_idx = -1
            hovered_type = None

            for stream_idx, level_idx in contour_indices:
                contour_idx = level_idx  # For display purposes

                imgui.text(f"=== Level {level_idx}, Contour {stream_idx} ===")

                # Show bounding plane corner angles (debug info)
                bp_info = get_bounding_plane(stream_idx, level_idx)
                if bp_info is not None:
                    bp_corners = bp_info.get('bounding_plane', None)
                    if bp_corners is not None and len(bp_corners) >= 4:
                        # Compute angles at each corner
                        angles = []
                        for i in range(4):
                            p0 = np.array(bp_corners[(i - 1) % 4])
                            p1 = np.array(bp_corners[i])
                            p2 = np.array(bp_corners[(i + 1) % 4])
                            v1 = p0 - p1
                            v2 = p2 - p1
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                            angles.append(angle_deg)
                        imgui.text(f"BP Angles: {angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}, {angles[3]:.1f}")

                imgui.separator()

                # Get canvas dimensions
                canvas_size = 280
                padding = 20
                column_width = canvas_size + 2 * padding + 20  # Fixed column width

                # Two columns: unit square (left) and contour (right)
                imgui.columns(2, f"inspect_cols##{name}_{level_idx}_{stream_idx}", border=True)
                imgui.set_column_width(0, column_width)
                imgui.set_column_width(1, column_width)

                draw_list = imgui.get_window_draw_list()
                mouse_pos = imgui.get_mouse_pos()
                hovered_idx = -1
                hovered_type = None  # 'vertex', 'fiber', or 'waypoint'
                hover_radius = 8.0

                # Get contour data first (needed for both columns)
                contour_match = None
                p_screen_points = []
                q_screen_points = []
                fiber_screen_points = []
                waypoint_screen_points = []
                corner_screen_points_left = []  # Bounding plane corners on unit square
                corner_screen_points_right = []  # Bounding plane corners on contour
                corner_to_closest_vertex = []  # (corner_idx, closest_vertex_idx) pairs
                contour_2d_norm = None

                # Helper function to compute normalized [0,1] coordinates by solving linear system
                def point_to_unit_square_2d(point_3d, mean, basis_x, basis_y, bp_corners):
                    """Convert 3D point to [0,1]x[0,1] by inverting the bounding plane formula.

                    Q was created as: Q = bp[0] + u * (bp[1]-bp[0]) + v * (bp[3]-bp[0])
                    So we solve: Q - bp[0] = u * edge_x + v * edge_y
                    This gives proper [0,1] coordinates regardless of basis orthogonality.
                    """
                    v0 = bp_corners[0]
                    edge_x = bp_corners[1] - bp_corners[0]  # horizontal edge
                    edge_y = bp_corners[3] - bp_corners[0]  # vertical edge

                    # Solve 2x2 system using least squares (works in 3D)
                    # [edge_x | edge_y] * [u; v] = point - v0
                    rel_p = point_3d - v0

                    # Build matrix A = [edge_x, edge_y] as columns (3x2)
                    A = np.column_stack([edge_x, edge_y])

                    # Solve using least squares
                    result, _, _, _ = np.linalg.lstsq(A, rel_p, rcond=None)
                    u, v = result[0], result[1]

                    return np.array([np.clip(u, 0, 1), np.clip(v, 0, 1)])

                plane_info = get_bounding_plane(stream_idx, level_idx)
                if plane_info is not None:
                    contour_match = plane_info.get('contour_match', None)

                    if contour_match is not None and len(contour_match) > 0 and 'basis_x' in plane_info:
                        mean = plane_info['mean']
                        basis_x = plane_info['basis_x']
                        basis_y = plane_info['basis_y']
                        bp = plane_info.get('bounding_plane', None)

                # Left column: Unit square with Q points
                imgui.text("Unit Square (Q points)")
                cursor_pos_left = imgui.get_cursor_screen_pos()

                left_x0, left_y0 = cursor_pos_left[0] + padding, cursor_pos_left[1] + padding
                left_x1, left_y1 = left_x0 + canvas_size, left_y0 + canvas_size

                # Background
                draw_list.add_rect_filled(left_x0, left_y0, left_x1, left_y1, imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))
                draw_list.add_rect(left_x0, left_y0, left_x1, left_y1, imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 1.0), thickness=2.0)

                # Draw fiber samples (green) and check hover
                fiber_samples = []
                if has_fiber and stream_idx < len(obj.fiber_architecture):
                    fiber_samples = obj.fiber_architecture[stream_idx]
                    for i, sample in enumerate(fiber_samples):
                        if len(sample) >= 2:
                            sx = left_x0 + sample[0] * canvas_size
                            sy = left_y0 + (1 - sample[1]) * canvas_size
                            fiber_screen_points.append((sx, sy))
                            draw_list.add_circle_filled(sx, sy, 4, imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 1.0))
                            # Check hover on fiber samples
                            dist = np.sqrt((mouse_pos[0] - sx)**2 + (mouse_pos[1] - sy)**2)
                            if dist < hover_radius and hovered_idx < 0:
                                hovered_idx = i
                                hovered_type = 'fiber'

                # Compute and draw Q points from contour_match (cyan) - draw immediately so they're not covered
                # Use bounding plane parametric coordinates to match MVC computation in find_waypoints()
                if contour_match is not None and bp is not None and len(bp) >= 4:
                    for i, (p, q) in enumerate(contour_match):
                        p = np.array(p)
                        q = np.array(q)
                        # Compute Q's position on unit square using bounding plane parametric coords
                        q_norm = point_to_unit_square_2d(q, mean, basis_x, basis_y, bp)
                        qx = left_x0 + q_norm[0] * canvas_size
                        qy = left_y0 + (1 - q_norm[1]) * canvas_size
                        q_screen_points.append((qx, qy))
                        # Draw Q point immediately (cyan)
                        draw_list.add_circle_filled(qx, qy, 3, imgui.get_color_u32_rgba(0.0, 0.8, 0.8, 1.0))

                        # Check hover on Q points
                        dist = np.sqrt((mouse_pos[0] - qx)**2 + (mouse_pos[1] - qy)**2)
                        if dist < hover_radius and hovered_idx < 0:
                            hovered_idx = i
                            hovered_type = 'vertex'

                    # Draw bounding plane corners on unit square (corners are at (0,0), (1,0), (1,1), (0,1))
                    corner_uv = [(0, 0), (1, 0), (1, 1), (0, 1)]  # Unit square corners
                    for ci, (cu, cv) in enumerate(corner_uv):
                        cx = left_x0 + cu * canvas_size
                        cy = left_y0 + (1 - cv) * canvas_size
                        corner_screen_points_left.append((cx, cy))
                        # Draw corner (purple/magenta diamond)
                        draw_list.add_quad_filled(cx, cy - 5, cx + 5, cy, cx, cy + 5, cx - 5, cy,
                                                 imgui.get_color_u32_rgba(0.8, 0.2, 0.8, 1.0))
                        # Check hover on corners
                        dist = np.sqrt((mouse_pos[0] - cx)**2 + (mouse_pos[1] - cy)**2)
                        if dist < hover_radius and hovered_idx < 0:
                            hovered_idx = ci
                            hovered_type = 'corner'

                # Reserve space
                imgui.dummy(canvas_size + 2 * padding, canvas_size + 2 * padding)

                # Right column: Contour with P points
                imgui.next_column()
                imgui.text(f"Level {level_idx} (P points)")
                cursor_pos_right = imgui.get_cursor_screen_pos()

                right_x0, right_y0 = cursor_pos_right[0] + padding, cursor_pos_right[1] + padding
                right_x1, right_y1 = right_x0 + canvas_size, right_y0 + canvas_size

                # Background
                draw_list.add_rect_filled(right_x0, right_y0, right_x1, right_y1, imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))

                # Draw contour and P points
                if contour_match is not None and 'basis_x' in plane_info:
                    # Project P points to 2D
                    p_2d_list = []
                    for p, q in contour_match:
                        p = np.array(p)
                        p_2d = np.array([np.dot(p - mean, basis_x), np.dot(p - mean, basis_y)])
                        p_2d_list.append(p_2d)
                    p_2d_arr = np.array(p_2d_list)

                    # Normalization (preserve aspect ratio)
                    min_xy = p_2d_arr.min(axis=0)
                    max_xy = p_2d_arr.max(axis=0)
                    range_xy = max_xy - min_xy
                    range_xy[range_xy < 1e-10] = 1.0
                    max_range = max(range_xy[0], range_xy[1])
                    margin = 0.1
                    scale = (1 - 2 * margin) / max_range
                    center_xy = (min_xy + max_xy) / 2

                    # Compute screen points for contour
                    p_screen_points = []
                    for p_2d in p_2d_list:
                        p_norm = (p_2d - center_xy) * scale + 0.5
                        px = right_x0 + p_norm[0] * canvas_size
                        py = right_y0 + (1 - p_norm[1]) * canvas_size
                        p_screen_points.append((px, py))

                    # Draw contour lines (yellow)
                    for i in range(len(p_screen_points)):
                        p1 = p_screen_points[i]
                        p2 = p_screen_points[(i + 1) % len(p_screen_points)]
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                          imgui.get_color_u32_rgba(0.8, 0.8, 0.2, 1.0), thickness=2.0)

                    # Check hover on P points
                    for i, (px, py) in enumerate(p_screen_points):
                        dist = np.sqrt((mouse_pos[0] - px)**2 + (mouse_pos[1] - py)**2)
                        if dist < hover_radius and hovered_idx < 0:
                            hovered_idx = i
                            hovered_type = 'vertex'

                    # Draw actual bounding plane (blue) - project 3D bounding plane corners to 2D
                    if bp is not None and len(bp) >= 4:
                        bp_screen = []
                        for corner_3d in bp[:4]:
                            corner_3d = np.array(corner_3d)
                            corner_2d = np.array([np.dot(corner_3d - mean, basis_x), np.dot(corner_3d - mean, basis_y)])
                            corner_norm = (corner_2d - center_xy) * scale + 0.5
                            cx = right_x0 + corner_norm[0] * canvas_size
                            cy = right_y0 + (1 - corner_norm[1]) * canvas_size
                            bp_screen.append((cx, cy))
                        # Draw bounding plane edges
                        for i in range(4):
                            p1 = bp_screen[i]
                            p2 = bp_screen[(i + 1) % 4]
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                              imgui.get_color_u32_rgba(0.3, 0.5, 0.9, 1.0), thickness=1.5)

                    # Draw bounding plane corners and lines to closest contour vertices
                    if bp is not None and len(bp) >= 4:
                        # Find corner-vertex correspondence based on Q positions (not ray-based)
                        # This reflects the actual correspondence used in fiber computation
                        bp_corners_3d = [np.array(c) for c in bp[:4]]
                        q_based_corner_indices = []
                        for ci, corner_3d in enumerate(bp_corners_3d):
                            # Find vertex whose Q is closest to this corner
                            min_dist = float('inf')
                            closest_vi = 0
                            for vi, (p, q) in enumerate(contour_match):
                                q_arr = np.array(q)
                                dist = np.linalg.norm(q_arr - corner_3d)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_vi = vi
                            q_based_corner_indices.append(closest_vi)

                        # Project actual bounding plane corners to 2D
                        for ci, corner_3d in enumerate(bp[:4]):
                            corner_3d = np.array(corner_3d)
                            corner_2d = np.array([np.dot(corner_3d - mean, basis_x), np.dot(corner_3d - mean, basis_y)])
                            corner_norm = (corner_2d - center_xy) * scale + 0.5
                            cx = right_x0 + corner_norm[0] * canvas_size
                            cy = right_y0 + (1 - corner_norm[1]) * canvas_size
                            corner_screen_points_right.append((cx, cy))

                            # Use Q-based corner correspondence (reflects actual fiber computation)
                            if len(p_screen_points) > 0 and ci < len(q_based_corner_indices):
                                closest_vi = q_based_corner_indices[ci]
                                corner_to_closest_vertex.append((ci, closest_vi))

                                # Draw line from corner to corresponding vertex (purple, thin)
                                if closest_vi < len(p_screen_points):
                                    px, py = p_screen_points[closest_vi]
                                    draw_list.add_line(cx, cy, px, py,
                                                     imgui.get_color_u32_rgba(0.7, 0.3, 0.7, 0.6), thickness=1.0)

                            # Draw corner (purple/magenta diamond)
                            draw_list.add_quad_filled(cx, cy - 5, cx + 5, cy, cx, cy + 5, cx - 5, cy,
                                                     imgui.get_color_u32_rgba(0.8, 0.2, 0.8, 1.0))

                            # Check hover on corners (right side)
                            dist = np.sqrt((mouse_pos[0] - cx)**2 + (mouse_pos[1] - cy)**2)
                            if dist < hover_radius and hovered_idx < 0:
                                hovered_idx = ci
                                hovered_type = 'corner'

                    # Draw waypoints (red) and check hover
                    if (hasattr(obj, 'waypoints') and obj.waypoints is not None and
                        stream_idx < len(obj.waypoints) and level_idx < len(obj.waypoints[stream_idx])):
                        waypoints_3d = obj.waypoints[stream_idx][level_idx]
                        if waypoints_3d is not None and len(waypoints_3d) > 0:
                            for wi, wp in enumerate(waypoints_3d):
                                wp = np.array(wp)
                                wp_2d = np.array([np.dot(wp - mean, basis_x), np.dot(wp - mean, basis_y)])
                                wp_norm = (wp_2d - center_xy) * scale + 0.5
                                wpx = right_x0 + wp_norm[0] * canvas_size
                                wpy = right_y0 + (1 - wp_norm[1]) * canvas_size
                                waypoint_screen_points.append((wpx, wpy))
                                draw_list.add_circle_filled(wpx, wpy, 5, imgui.get_color_u32_rgba(0.9, 0.3, 0.3, 1.0))
                                # Check hover on waypoints
                                dist = np.sqrt((mouse_pos[0] - wpx)**2 + (mouse_pos[1] - wpy)**2)
                                if dist < hover_radius and hovered_idx < 0:
                                    hovered_idx = wi
                                    hovered_type = 'waypoint'

                # Draw P vertex points (non-highlighted)
                for i in range(len(p_screen_points)):
                    px, py = p_screen_points[i]
                    if not (hovered_type == 'vertex' and i == hovered_idx):
                        draw_list.add_circle_filled(px, py, 3, imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0))

                # Draw resampled contour vertices and edges (cyan) if available
                if (hasattr(obj, 'contours_resampled') and obj.contours_resampled is not None and
                    stream_idx < len(obj.contours_resampled) and
                    level_idx < len(obj.contours_resampled[stream_idx])):
                    resampled = obj.contours_resampled[stream_idx][level_idx]
                    if resampled is not None and len(resampled) > 0:
                        # Project resampled vertices to 2D using same transformation
                        resampled_screen = []
                        for rv in resampled:
                            rv = np.array(rv)
                            rv_2d = np.array([np.dot(rv - mean, basis_x), np.dot(rv - mean, basis_y)])
                            rv_norm = (rv_2d - center_xy) * scale + 0.5
                            rvx = right_x0 + rv_norm[0] * canvas_size
                            rvy = right_y0 + (1 - rv_norm[1]) * canvas_size
                            resampled_screen.append((rvx, rvy))
                        # Draw resampled contour edges (cyan)
                        for i in range(len(resampled_screen)):
                            p1 = resampled_screen[i]
                            p2 = resampled_screen[(i + 1) % len(resampled_screen)]
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                              imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 0.9), thickness=1.5)
                        # Draw resampled vertices
                        for rvx, rvy in resampled_screen:
                            draw_list.add_circle_filled(rvx, rvy, 3, imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0))

                # Reserve space
                imgui.dummy(canvas_size + 2 * padding, canvas_size + 2 * padding)

                imgui.columns(1)

                # Draw all highlights AFTER columns are done (ensures they're on top)
                # Re-get draw list to ensure we're drawing on top layer
                draw_list = imgui.get_window_draw_list()

                if hovered_type == 'vertex' and hovered_idx >= 0:
                    # Highlight P on contour (orange with white border)
                    if hovered_idx < len(p_screen_points):
                        px, py = p_screen_points[hovered_idx]
                        draw_list.add_circle_filled(px, py, 7, imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0))
                        draw_list.add_circle(px, py, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                    # Highlight corresponding Q on unit square (magenta with white border)
                    if hovered_idx < len(q_screen_points):
                        qx, qy = q_screen_points[hovered_idx]
                        draw_list.add_circle_filled(qx, qy, 7, imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0))
                        draw_list.add_circle(qx, qy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)

                elif hovered_type == 'fiber' and hovered_idx >= 0:
                    # Highlight fiber sample on unit square (bright green with white border)
                    if hovered_idx < len(fiber_screen_points):
                        fx, fy = fiber_screen_points[hovered_idx]
                        draw_list.add_circle_filled(fx, fy, 7, imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 1.0))
                        draw_list.add_circle(fx, fy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                    # Highlight corresponding waypoint on contour (bright red with white border)
                    if hovered_idx < len(waypoint_screen_points):
                        wpx, wpy = waypoint_screen_points[hovered_idx]
                        draw_list.add_circle_filled(wpx, wpy, 7, imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 1.0))
                        draw_list.add_circle(wpx, wpy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                    # Draw MVC weight-proportional vertices
                    mvc_w = None
                    if (hasattr(obj, 'mvc_weights') and stream_idx < len(obj.mvc_weights) and
                        level_idx < len(obj.mvc_weights[stream_idx])):
                        mvc_w = obj.mvc_weights[stream_idx][level_idx]
                    # Fallback: compute on-the-fly if mvc_weights not available
                    if mvc_w is None and contour_match is not None and len(fiber_samples) > 0:
                        try:
                            # Check if contour_match has valid 3D points (not 2D)
                            if len(contour_match) > 0 and len(contour_match[0]) >= 2:
                                first_q = np.array(contour_match[0][1])
                                if len(first_q) == 3:  # Valid 3D point
                                    _, _, mvc_w = obj.find_waypoints(plane_info, fiber_samples)
                        except Exception:
                            mvc_w = None
                    if mvc_w is not None and len(mvc_w) > hovered_idx:
                        weights = np.array(mvc_w[hovered_idx])
                        if len(weights) > 0 and np.isfinite(weights).all():
                            max_w = weights.max()
                            if max_w > 1e-8:
                                max_radius = 7.0  # Same as hover emphasis
                                # Sort by weight descending (draw largest first, smallest on top)
                                sorted_indices = np.argsort(weights)[::-1]
                                for vi in sorted_indices:
                                    w = weights[vi]
                                    if w > 1e-8:  # Only draw non-zero weights
                                        rel_size = w / max_w
                                        radius = max(1.0, max_radius * rel_size)  # Minimum radius of 1
                                        # Draw on P (contour) side - yellow
                                        if vi < len(p_screen_points):
                                            px, py = p_screen_points[vi]
                                            draw_list.add_circle_filled(px, py, radius, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.8))
                                        # Draw on Q (unit square) side - yellow
                                        if vi < len(q_screen_points):
                                            qx, qy = q_screen_points[vi]
                                            draw_list.add_circle_filled(qx, qy, radius, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.8))

                elif hovered_type == 'waypoint' and hovered_idx >= 0:
                    # Highlight waypoint on contour (bright red with white border)
                    if hovered_idx < len(waypoint_screen_points):
                        wpx, wpy = waypoint_screen_points[hovered_idx]
                        draw_list.add_circle_filled(wpx, wpy, 7, imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 1.0))
                        draw_list.add_circle(wpx, wpy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                    # Highlight corresponding fiber sample on unit square (bright green with white border)
                    if hovered_idx < len(fiber_screen_points):
                        fx, fy = fiber_screen_points[hovered_idx]
                        draw_list.add_circle_filled(fx, fy, 7, imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 1.0))
                        draw_list.add_circle(fx, fy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                    # Draw MVC weight-proportional vertices
                    mvc_w = None
                    if (hasattr(obj, 'mvc_weights') and stream_idx < len(obj.mvc_weights) and
                        level_idx < len(obj.mvc_weights[stream_idx])):
                        mvc_w = obj.mvc_weights[stream_idx][level_idx]
                    # Fallback: compute on-the-fly if mvc_weights not available
                    if mvc_w is None and contour_match is not None and len(fiber_samples) > 0:
                        try:
                            # Check if contour_match has valid 3D points (not 2D)
                            if len(contour_match) > 0 and len(contour_match[0]) >= 2:
                                first_q = np.array(contour_match[0][1])
                                if len(first_q) == 3:  # Valid 3D point
                                    _, _, mvc_w = obj.find_waypoints(plane_info, fiber_samples)
                        except Exception:
                            mvc_w = None
                    if mvc_w is not None and len(mvc_w) > hovered_idx:
                        weights = np.array(mvc_w[hovered_idx])
                        if len(weights) > 0 and np.isfinite(weights).all():
                            max_w = weights.max()
                            if max_w > 1e-8:
                                max_radius = 7.0  # Same as hover emphasis
                                # Sort by weight descending (draw largest first, smallest on top)
                                sorted_indices = np.argsort(weights)[::-1]
                                for vi in sorted_indices:
                                    w = weights[vi]
                                    if w > 1e-8:  # Only draw non-zero weights
                                        rel_size = w / max_w
                                        radius = max(1.0, max_radius * rel_size)  # Minimum radius of 1
                                        # Draw on P (contour) side - yellow
                                        if vi < len(p_screen_points):
                                            px, py = p_screen_points[vi]
                                            draw_list.add_circle_filled(px, py, radius, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.8))
                                        # Draw on Q (unit square) side - yellow
                                        if vi < len(q_screen_points):
                                            qx, qy = q_screen_points[vi]
                                            draw_list.add_circle_filled(qx, qy, radius, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.8))

                elif hovered_type == 'corner' and hovered_idx >= 0:
                    # Highlight corner on unit square (left side) - bright magenta with white border
                    if hovered_idx < len(corner_screen_points_left):
                        cx, cy = corner_screen_points_left[hovered_idx]
                        draw_list.add_quad_filled(cx, cy - 8, cx + 8, cy, cx, cy + 8, cx - 8, cy,
                                                 imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0))
                        draw_list.add_quad(cx, cy - 10, cx + 10, cy, cx, cy + 10, cx - 10, cy,
                                          imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                    # Highlight corner on contour (right side) - bright magenta with white border
                    if hovered_idx < len(corner_screen_points_right):
                        cx, cy = corner_screen_points_right[hovered_idx]
                        draw_list.add_quad_filled(cx, cy - 8, cx + 8, cy, cx, cy + 8, cx - 8, cy,
                                                 imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0))
                        draw_list.add_quad(cx, cy - 10, cx + 10, cy, cx, cy + 10, cx - 10, cy,
                                          imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                    # Highlight the closest contour vertex and draw emphasized line
                    for corner_ci, closest_vi in corner_to_closest_vertex:
                        if corner_ci == hovered_idx:
                            # Draw emphasized line from corner to closest vertex
                            if hovered_idx < len(corner_screen_points_right) and closest_vi < len(p_screen_points):
                                cx, cy = corner_screen_points_right[hovered_idx]
                                px, py = p_screen_points[closest_vi]
                                draw_list.add_line(cx, cy, px, py,
                                                 imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0), thickness=3.0)
                            # Highlight the closest P vertex
                            if closest_vi < len(p_screen_points):
                                px, py = p_screen_points[closest_vi]
                                draw_list.add_circle_filled(px, py, 7, imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0))
                                draw_list.add_circle(px, py, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                            # Highlight the corresponding Q vertex
                            if closest_vi < len(q_screen_points):
                                qx, qy = q_screen_points[closest_vi]
                            draw_list.add_circle_filled(qx, qy, 7, imgui.get_color_u32_rgba(0.0, 0.8, 0.8, 1.0))
                            draw_list.add_circle(qx, qy, 9, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                        break

            # Show tooltip
            if hovered_idx >= 0:
                if hovered_type == 'vertex':
                    imgui.set_tooltip(f"Vertex {hovered_idx}")
                elif hovered_type == 'fiber':
                    imgui.set_tooltip(f"Fiber {hovered_idx}")
                elif hovered_type == 'waypoint':
                    imgui.set_tooltip(f"Waypoint {hovered_idx}")
                elif hovered_type == 'corner':
                    corner_names = ['Bottom-Left', 'Bottom-Right', 'Top-Right', 'Top-Left']
                    corner_name = corner_names[hovered_idx] if hovered_idx < 4 else f"Corner {hovered_idx}"
                    # Find closest vertex for this corner
                    closest_vi = -1
                    for corner_ci, vi in corner_to_closest_vertex:
                        if corner_ci == hovered_idx:
                            closest_vi = vi
                            break
                    if closest_vi >= 0:
                        imgui.set_tooltip(f"{corner_name} Corner -> Vertex {closest_vi}")

                if show_all:
                    imgui.separator()

            # Handle click for correspondence mode
            mouse_clicked = imgui.is_mouse_clicked(0)  # Left click
            if mouse_clicked and hovered_idx >= 0:
                if hovered_type == 'corner':
                    # Clicking corner enters/updates correspondence mode
                    self.inspect_2d_corr_mode[name] = True
                    self.inspect_2d_corr_corner[name] = hovered_idx
                    self.inspect_2d_corr_vertex[name] = -1  # Reset vertex selection
                    corr_mode = True
                    corr_corner = hovered_idx
                    corr_vertex = -1
                elif hovered_type == 'vertex' and corr_mode and corr_corner >= 0:
                    # Clicking vertex when in correspondence mode with corner selected
                    self.inspect_2d_corr_vertex[name] = hovered_idx
                    corr_vertex = hovered_idx
                elif hovered_type == 'fiber' and edit_fiber_mode:
                    # Clicking on existing fiber in edit mode - select it
                    self.inspect_2d_edit_fiber_selected[name] = hovered_idx
                    self.inspect_2d_edit_fiber_preview[name] = None
                    edit_fiber_selected = hovered_idx
                    edit_fiber_preview = None

            # Handle click on empty space in unit square for edit fiber mode
            if mouse_clicked and edit_fiber_mode and hovered_idx < 0:
                # Check if click is within the left canvas (unit square)
                if (left_x0 <= mouse_pos[0] <= left_x0 + canvas_size and
                    left_y0 <= mouse_pos[1] <= left_y0 + canvas_size):
                    # Convert screen position to unit square coordinates
                    u = (mouse_pos[0] - left_x0) / canvas_size
                    v = 1 - (mouse_pos[1] - left_y0) / canvas_size  # Flip Y
                    # Clamp to [0, 1]
                    u = max(0.0, min(1.0, u))
                    v = max(0.0, min(1.0, v))
                    # Set preview position and clear test data
                    self.inspect_2d_edit_fiber_preview[name] = (u, v)
                    self.inspect_2d_edit_fiber_selected[name] = -1
                    self.inspect_2d_edit_fiber_test[name] = None  # Clear previous test
                    # Clear test fiber from object
                    obj.test_fiber_waypoints = None
                    obj.test_fiber_stream_idx = None
                    edit_fiber_preview = (u, v)
                    edit_fiber_selected = -1
                    edit_fiber_test = None

            # Draw correspondence mode visual feedback
            if corr_mode and corr_corner >= 0:
                draw_list = imgui.get_window_draw_list()
                # Highlight selected corner with green
                if corr_corner < len(corner_screen_points_right):
                    cx, cy = corner_screen_points_right[corr_corner]
                    draw_list.add_quad_filled(cx, cy - 10, cx + 10, cy, cx, cy + 10, cx - 10, cy,
                                             imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                    draw_list.add_quad(cx, cy - 12, cx + 12, cy, cx, cy + 12, cx - 12, cy,
                                      imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                if corr_corner < len(corner_screen_points_left):
                    cx, cy = corner_screen_points_left[corr_corner]
                    draw_list.add_quad_filled(cx, cy - 10, cx + 10, cy, cx, cy + 10, cx - 10, cy,
                                             imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                    draw_list.add_quad(cx, cy - 12, cx + 12, cy, cx, cy + 12, cx - 12, cy,
                                      imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)

                # Highlight selected vertex with green if selected
                if corr_vertex >= 0 and corr_vertex < len(p_screen_points):
                    px, py = p_screen_points[corr_vertex]
                    draw_list.add_circle_filled(px, py, 9, imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                    draw_list.add_circle(px, py, 11, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                    # Also highlight on Q side
                    if corr_vertex < len(q_screen_points):
                        qx, qy = q_screen_points[corr_vertex]
                        draw_list.add_circle_filled(qx, qy, 9, imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                        draw_list.add_circle(qx, qy, 11, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                    # Draw line from corner to selected vertex
                    if corr_corner < len(corner_screen_points_right):
                        cx, cy = corner_screen_points_right[corr_corner]
                        draw_list.add_line(cx, cy, px, py,
                                          imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0), thickness=3.0)

            # Draw edit fiber mode visual feedback
            if edit_fiber_mode:
                draw_list = imgui.get_window_draw_list()
                # Highlight selected fiber with yellow ring
                if edit_fiber_selected >= 0 and edit_fiber_selected < len(fiber_screen_points):
                    fx, fy = fiber_screen_points[edit_fiber_selected]
                    draw_list.add_circle_filled(fx, fy, 8, imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0))
                    draw_list.add_circle(fx, fy, 10, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.5)
                # Draw preview marker (cyan) for new fiber position
                if edit_fiber_preview is not None:
                    px = left_x0 + edit_fiber_preview[0] * canvas_size
                    py = left_y0 + (1 - edit_fiber_preview[1]) * canvas_size
                    draw_list.add_circle_filled(px, py, 6, imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 0.7))
                    draw_list.add_circle(px, py, 8, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), thickness=2.0)
                    # Draw crosshair
                    draw_list.add_line(px - 12, py, px + 12, py, imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 0.5), 1.5)
                    draw_list.add_line(px, py - 12, px, py + 12, imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 0.5), 1.5)

            # Correspondence mode UI (Cancel/Confirm buttons)
            if corr_mode:
                imgui.separator()
                corner_names = ['Bottom-Left', 'Bottom-Right', 'Top-Right', 'Top-Left']
                corner_name = corner_names[corr_corner] if 0 <= corr_corner < 4 else f"Corner {corr_corner}"
                if corr_vertex >= 0:
                    imgui.text(f"Set {corner_name} -> Vertex {corr_vertex}")
                else:
                    imgui.text(f"Selected {corner_name}, click a contour vertex")

                # Cancel button - always enabled
                if imgui.button(f"Cancel##{name}_corr"):
                    self.inspect_2d_corr_mode[name] = False
                    self.inspect_2d_corr_corner[name] = -1
                    self.inspect_2d_corr_vertex[name] = -1

                imgui.same_line()

                # Confirm button - only enabled when vertex is selected
                if corr_vertex < 0:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                    imgui.button(f"Confirm##{name}_corr")
                    imgui.pop_style_var()
                else:
                    if imgui.button(f"Confirm##{name}_corr"):
                        # Apply the correspondence change
                        self._apply_corner_correspondence(name, obj, stream_idx, level_idx,
                                                          corr_corner, corr_vertex, is_post_stream)
                        # Reset correspondence mode
                        self.inspect_2d_corr_mode[name] = False
                        self.inspect_2d_corr_corner[name] = -1
                        self.inspect_2d_corr_vertex[name] = -1

            # Edit Fiber mode UI (only available when not in "Show All" mode)
            if has_fiber and not corr_mode and not show_all:
                imgui.separator()
                if not edit_fiber_mode:
                    # Show Edit Fiber button only when fiber architecture exists
                    if imgui.button(f"Edit Fiber##{name}"):
                        self.inspect_2d_edit_fiber_mode[name] = True
                        self.inspect_2d_edit_fiber_selected[name] = -1
                        self.inspect_2d_edit_fiber_preview[name] = None
                else:
                    # In edit fiber mode
                    imgui.text("Edit Fiber Mode (click unit square)")

                    # Exit button
                    if imgui.button(f"Exit Edit Mode##{name}"):
                        self.inspect_2d_edit_fiber_mode[name] = False
                        self.inspect_2d_edit_fiber_selected[name] = -1
                        self.inspect_2d_edit_fiber_preview[name] = None
                        self.inspect_2d_edit_fiber_test[name] = None
                        # Clear test fiber from object
                        obj.test_fiber_waypoints = None
                        obj.test_fiber_stream_idx = None

                    imgui.same_line()

                    # Delete button - enabled when existing fiber is selected
                    if edit_fiber_selected >= 0:
                        if imgui.button(f"Delete Fiber {edit_fiber_selected}##{name}"):
                            self._delete_fiber(name, obj, stream_idx, edit_fiber_selected, is_post_stream)
                            self.inspect_2d_edit_fiber_selected[name] = -1
                            self.inspect_2d_edit_fiber_preview[name] = None
                    else:
                        imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                        imgui.button(f"Delete##{name}")
                        imgui.pop_style_var()

                    imgui.same_line()

                    # Add button - enabled when preview position is set
                    if edit_fiber_preview is not None:
                        if imgui.button(f"Add Fiber##{name}"):
                            self._add_fiber(name, obj, stream_idx, edit_fiber_preview, is_post_stream)
                            self.inspect_2d_edit_fiber_selected[name] = -1
                            self.inspect_2d_edit_fiber_preview[name] = None
                            self.inspect_2d_edit_fiber_test[name] = None  # Clear test data
                            # Clear test fiber from object
                            obj.test_fiber_waypoints = None
                            obj.test_fiber_stream_idx = None
                    else:
                        imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                        imgui.button(f"Add##{name}")
                        imgui.pop_style_var()

                    imgui.same_line()

                    # Test button - enabled when preview position is set
                    if edit_fiber_preview is not None:
                        if imgui.button(f"Test##{name}"):
                            self._test_fiber(name, obj, stream_idx, edit_fiber_preview, is_post_stream)
                    else:
                        imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                        imgui.button(f"Test##{name}")
                        imgui.pop_style_var()

                    # Show info text
                    if edit_fiber_selected >= 0:
                        imgui.text(f"Selected: Fiber {edit_fiber_selected}")
                    elif edit_fiber_preview is not None:
                        test_status = " (testing)" if edit_fiber_test is not None else ""
                        imgui.text(f"Preview: ({edit_fiber_preview[0]:.3f}, {edit_fiber_preview[1]:.3f}){test_status}")
                    else:
                        imgui.text("Click on fiber to select, or empty space to add")

            # End scrollable region (always used now)
            imgui.end_child()

            imgui.end()

        # Close windows that were marked for closing
        for name in muscles_to_close:
            self.inspect_2d_open[name] = False
            # Clear inspector highlight when window closes
            if name in self.zygote_muscle_meshes:
                self.zygote_muscle_meshes[name].inspector_highlight_stream = None
                self.zygote_muscle_meshes[name].inspector_highlight_level = None

    def _apply_corner_correspondence(self, name, obj, stream_idx, level_idx, corner_idx, vertex_idx, is_post_stream):
        """
        Apply manual corner-to-vertex correspondence.

        When a corner assignment changes, recompute Q positions based on arc-length
        interpolation between the new corner vertices.

        Args:
            name: Muscle name
            obj: ContourMesh object
            stream_idx: Stream index
            level_idx: Level index
            corner_idx: Selected corner (0-3)
            vertex_idx: Selected vertex index
            is_post_stream: Whether data is in post-stream format
        """
        print(f"Applying correspondence: Corner {corner_idx} -> Vertex {vertex_idx}")
        print(f"  Stream: {stream_idx}, Level: {level_idx}, Post-stream: {is_post_stream}")

        # Get current bounding plane info
        if is_post_stream:
            bp_info = obj.bounding_planes[stream_idx][level_idx]
        else:
            bp_info = obj.bounding_planes[level_idx][stream_idx]

        bp_corners = bp_info.get('bounding_plane', None)
        contour_match = bp_info.get('contour_match', None)

        if bp_corners is None or contour_match is None:
            print("  Error: No bounding plane or contour_match found")
            return

        # Extract P vertices from contour_match
        P_vertices = [np.array(p) for p, q in contour_match]
        n = len(P_vertices)

        # Get current corner-to-vertex mapping using Q-based method (reads from saved contour_match)
        # This preserves any previous manual corner changes
        bp_corners_3d = np.array(bp_corners[:4])
        current_corner_indices = []
        for ci, corner_3d in enumerate(bp_corners_3d):
            min_dist = float('inf')
            closest_vi = 0
            for vi, (p, q) in enumerate(contour_match):
                q_arr = np.array(q)
                dist = np.linalg.norm(q_arr - corner_3d)
                if dist < min_dist:
                    min_dist = dist
                    closest_vi = vi
            current_corner_indices.append(closest_vi)

        print(f"  Current corner indices (Q-based): {current_corner_indices}")

        # Update corner assignment
        new_corner_indices = current_corner_indices.copy()
        new_corner_indices[corner_idx] = vertex_idx

        print(f"  New corner indices: {new_corner_indices}")

        # Recompute Q positions based on arc-length interpolation between new corner vertices
        # BP corners: 0 (bottom-left), 1 (bottom-right), 2 (top-right), 3 (top-left)
        bp_corners_3d = [np.array(c) for c in bp_corners[:4]]

        new_contour_match = []

        # For each edge between consecutive corners, interpolate Q positions
        for edge_idx in range(4):
            corner_start = edge_idx
            corner_end = (edge_idx + 1) % 4

            vertex_start = new_corner_indices[corner_start]
            vertex_end = new_corner_indices[corner_end]

            q_start = bp_corners_3d[corner_start]
            q_end = bp_corners_3d[corner_end]

            # Get vertices between corner_start and corner_end (inclusive of start, exclusive of end)
            if vertex_end > vertex_start:
                vertex_list = list(range(vertex_start, vertex_end))
            elif vertex_end < vertex_start:
                # Wrap around
                vertex_list = list(range(vertex_start, n)) + list(range(0, vertex_end))
            else:
                # Same vertex (shouldn't happen normally)
                vertex_list = [vertex_start]

            if len(vertex_list) == 0:
                continue

            # Compute arc-length along this segment
            arc_lengths = [0.0]
            for i in range(1, len(vertex_list)):
                prev_v = P_vertices[vertex_list[i - 1]]
                curr_v = P_vertices[vertex_list[i]]
                arc_lengths.append(arc_lengths[-1] + np.linalg.norm(curr_v - prev_v))

            total_arc = arc_lengths[-1]
            if total_arc < 1e-10:
                total_arc = 1.0

            # Assign Q positions based on arc-length interpolation
            for i, vi in enumerate(vertex_list):
                t = arc_lengths[i] / total_arc
                q_interp = (1 - t) * q_start + t * q_end
                new_contour_match.append((P_vertices[vi], q_interp))

        # Update bounding plane info with new contour_match
        bp_info['contour_match'] = new_contour_match

        print(f"  Updated contour_match with {len(new_contour_match)} pairs (was {n})")

        # Debug: verify the change
        if len(new_contour_match) > 0:
            # Check Q at the new corner vertex
            for ci in range(4):
                vi = new_corner_indices[ci]
                # Find which entry in new_contour_match has this vertex
                for idx, (p, q) in enumerate(new_contour_match):
                    if np.allclose(p, P_vertices[vi]):
                        q_arr = np.array(q)
                        corner_arr = np.array(bp_corners[ci])
                        dist = np.linalg.norm(q_arr - corner_arr)
                        print(f"    Corner {ci} -> vertex {vi}: Q dist to corner = {dist:.6f}")
                        break

        # Verify bp_info is modified
        print(f"  bp_info id: {id(bp_info)}, contour_match id: {id(bp_info.get('contour_match'))}")

        # Recompute fiber structure if available
        if hasattr(obj, 'fiber_architecture') and obj.fiber_architecture is not None:
            if is_post_stream and stream_idx < len(obj.fiber_architecture):
                fiber_samples = obj.fiber_architecture[stream_idx]
                if fiber_samples is not None and len(fiber_samples) > 0:
                    print(f"  Recomputing waypoints for {len(fiber_samples)} fiber samples")

                    # Recompute waypoints and MVC weights for this level
                    # find_waypoints returns (Qs_normalized_2d, waypoints_3d, mvc_weights)
                    _, waypoints_3d, mvc_weights = obj.find_waypoints(bp_info, fiber_samples)

                    # Update waypoints
                    if hasattr(obj, 'waypoints') and obj.waypoints is not None:
                        if stream_idx < len(obj.waypoints) and level_idx < len(obj.waypoints[stream_idx]):
                            obj.waypoints[stream_idx][level_idx] = waypoints_3d
                            print(f"  Updated waypoints: {len(waypoints_3d)} points")

                    # Update MVC weights
                    if hasattr(obj, 'mvc_weights') and obj.mvc_weights is not None:
                        if stream_idx < len(obj.mvc_weights) and level_idx < len(obj.mvc_weights[stream_idx]):
                            obj.mvc_weights[stream_idx][level_idx] = mvc_weights
                            print(f"  Updated MVC weights")

        print("  Correspondence applied successfully")

    def _delete_fiber(self, name, obj, stream_idx, fiber_idx, is_post_stream):
        """
        Delete a fiber from the fiber architecture for a given stream.
        Removes the fiber from all levels and recomputes waypoints/MVC weights.

        Args:
            name: Muscle name
            obj: ContourMesh object
            stream_idx: Stream index
            fiber_idx: Index of fiber to delete
            is_post_stream: Whether data is in post-stream format
        """
        print(f"Deleting fiber {fiber_idx} from stream {stream_idx}")

        # Check if fiber_architecture exists
        if not hasattr(obj, 'fiber_architecture') or obj.fiber_architecture is None:
            print("  Error: No fiber architecture found")
            return

        if stream_idx >= len(obj.fiber_architecture):
            print(f"  Error: stream_idx {stream_idx} out of range")
            return

        fiber_samples = obj.fiber_architecture[stream_idx]
        if fiber_samples is None or fiber_idx >= len(fiber_samples):
            print(f"  Error: fiber_idx {fiber_idx} out of range")
            return

        # Remove the fiber sample (handle both list and numpy array)
        if isinstance(fiber_samples, np.ndarray):
            fiber_samples = np.delete(fiber_samples, fiber_idx, axis=0)
            obj.fiber_architecture[stream_idx] = fiber_samples
        else:
            fiber_samples.pop(fiber_idx)
        print(f"  Removed fiber sample, {len(fiber_samples)} remaining")

        # Recompute waypoints and MVC weights for all levels in this stream
        self._recompute_fiber_data_for_stream(obj, stream_idx, is_post_stream)

        print("  Fiber deleted successfully")

    def _add_fiber(self, name, obj, stream_idx, position, is_post_stream):
        """
        Add a new fiber to the fiber architecture for a given stream.
        Adds the fiber to all levels and recomputes waypoints/MVC weights.

        Args:
            name: Muscle name
            obj: ContourMesh object
            stream_idx: Stream index
            position: (u, v) position on unit square
            is_post_stream: Whether data is in post-stream format
        """
        print(f"Adding fiber at ({position[0]:.3f}, {position[1]:.3f}) to stream {stream_idx}")

        # Check if fiber_architecture exists
        if not hasattr(obj, 'fiber_architecture') or obj.fiber_architecture is None:
            print("  Error: No fiber architecture found")
            return

        if stream_idx >= len(obj.fiber_architecture):
            print(f"  Error: stream_idx {stream_idx} out of range")
            return

        fiber_samples = obj.fiber_architecture[stream_idx]
        if fiber_samples is None:
            fiber_samples = np.array([[position[0], position[1]]])
            obj.fiber_architecture[stream_idx] = fiber_samples
        elif isinstance(fiber_samples, np.ndarray):
            # Append to numpy array
            new_sample = np.array([[position[0], position[1]]])
            fiber_samples = np.vstack([fiber_samples, new_sample])
            obj.fiber_architecture[stream_idx] = fiber_samples
        else:
            # Append to list
            fiber_samples.append([position[0], position[1]])
        print(f"  Added fiber sample, {len(fiber_samples)} total")

        # Recompute waypoints and MVC weights for all levels in this stream
        self._recompute_fiber_data_for_stream(obj, stream_idx, is_post_stream)

        print("  Fiber added successfully")

    def _recompute_fiber_data_for_stream(self, obj, stream_idx, is_post_stream):
        """
        Recompute waypoints and MVC weights for all levels in a stream after fiber changes.

        Args:
            obj: ContourMesh object
            stream_idx: Stream index
            is_post_stream: Whether data is in post-stream format
        """
        fiber_samples = obj.fiber_architecture[stream_idx]
        if fiber_samples is None or len(fiber_samples) == 0:
            print("  No fiber samples to recompute")
            return

        # Get number of levels for this stream
        if is_post_stream:
            num_levels = len(obj.bounding_planes[stream_idx]) if stream_idx < len(obj.bounding_planes) else 0
        else:
            num_levels = len(obj.bounding_planes) if obj.bounding_planes else 0

        # Ensure waypoints and mvc_weights lists are properly sized
        if not hasattr(obj, 'waypoints') or obj.waypoints is None:
            obj.waypoints = []
        if not hasattr(obj, 'mvc_weights') or obj.mvc_weights is None:
            obj.mvc_weights = []

        # Extend lists if needed
        while len(obj.waypoints) <= stream_idx:
            obj.waypoints.append([])
        while len(obj.mvc_weights) <= stream_idx:
            obj.mvc_weights.append([])

        # Recompute for each level
        for level_idx in range(num_levels):
            # Get bounding plane info
            if is_post_stream:
                bp_info = obj.bounding_planes[stream_idx][level_idx]
            else:
                bp_info = obj.bounding_planes[level_idx][stream_idx]

            if bp_info is None:
                continue

            contour_match = bp_info.get('contour_match', None)
            if contour_match is None:
                continue

            # Recompute waypoints and MVC weights
            try:
                _, waypoints_3d, mvc_weights = obj.find_waypoints(bp_info, fiber_samples)

                # Ensure level lists are properly sized
                while len(obj.waypoints[stream_idx]) <= level_idx:
                    obj.waypoints[stream_idx].append([])
                while len(obj.mvc_weights[stream_idx]) <= level_idx:
                    obj.mvc_weights[stream_idx].append([])

                # Update data
                obj.waypoints[stream_idx][level_idx] = waypoints_3d
                obj.mvc_weights[stream_idx][level_idx] = mvc_weights
                print(f"  Recomputed level {level_idx}: {len(waypoints_3d)} waypoints")
            except Exception as e:
                print(f"  Error recomputing level {level_idx}: {e}")

    def _test_fiber(self, name, obj, stream_idx, position, is_post_stream):
        """
        Compute test waypoints for a fiber at the given position.
        Stores the waypoints for visualization in 3D (shown in blue).

        Args:
            name: Muscle name
            obj: ContourMesh object
            stream_idx: Stream index
            position: (u, v) position on unit square
            is_post_stream: Whether data is in post-stream format
        """
        print(f"Testing fiber at ({position[0]:.3f}, {position[1]:.3f}) for stream {stream_idx}")

        # Create a single fiber sample for testing
        test_fiber_sample = np.array([[position[0], position[1]]])

        # Get number of levels for this stream
        if is_post_stream:
            num_levels = len(obj.bounding_planes[stream_idx]) if stream_idx < len(obj.bounding_planes) else 0
        else:
            num_levels = len(obj.bounding_planes) if obj.bounding_planes else 0

        # Compute waypoints for each level
        test_waypoints = []
        for level_idx in range(num_levels):
            # Get bounding plane info
            if is_post_stream:
                bp_info = obj.bounding_planes[stream_idx][level_idx]
            else:
                bp_info = obj.bounding_planes[level_idx][stream_idx]

            if bp_info is None:
                test_waypoints.append(None)
                continue

            contour_match = bp_info.get('contour_match', None)
            if contour_match is None:
                test_waypoints.append(None)
                continue

            # Compute waypoint for this level
            try:
                _, waypoints_3d, _ = obj.find_waypoints(bp_info, test_fiber_sample)
                if len(waypoints_3d) > 0:
                    test_waypoints.append(waypoints_3d[0])  # Single waypoint
                    print(f"  Level {level_idx}: waypoint at [{waypoints_3d[0][0]:.4f}, {waypoints_3d[0][1]:.4f}, {waypoints_3d[0][2]:.4f}]")
                else:
                    test_waypoints.append(None)
            except Exception as e:
                print(f"  Error computing waypoint for level {level_idx}: {e}")
                test_waypoints.append(None)

        # Store test waypoints for visualization (both locally and on object for 3D drawing)
        self.inspect_2d_edit_fiber_test[name] = {
            'stream_idx': stream_idx,
            'waypoints': test_waypoints
        }
        # Also store on object for draw_fiber_architecture to access
        obj.test_fiber_waypoints = test_waypoints
        obj.test_fiber_stream_idx = stream_idx
        print(f"  Test fiber computed with {sum(1 for w in test_waypoints if w is not None)} valid waypoints")

    def _render_neck_viz_windows(self):
        """Render transition visualization windows showing source and target contours."""
        if not hasattr(self, 'neck_viz_open'):
            return

        muscles_to_close = []

        for name, is_open in list(self.neck_viz_open.items()):
            if not is_open:
                continue

            if name not in self.zygote_muscle_meshes:
                muscles_to_close.append(name)
                continue

            obj = self.zygote_muscle_meshes[name]

            if not hasattr(obj, '_neck_viz_data') or not obj._neck_viz_data:
                muscles_to_close.append(name)
                continue

            viz_data = obj._neck_viz_data
            num_viz = len(viz_data)

            imgui.set_next_window_size(450, 520, imgui.FIRST_USE_EVER)
            expanded, opened = imgui.begin(f"Neck Viz: {name}", True)

            if not opened:
                muscles_to_close.append(name)
                imgui.end()
                continue

            # Slider to select visualization
            viz_idx = self.neck_viz_idx.get(name, 0)
            viz_idx = min(viz_idx, max(0, num_viz - 1))
            changed, new_idx = imgui.slider_int(f"Transition##{name}", viz_idx, 0, max(0, num_viz - 1))
            if changed:
                self.neck_viz_idx[name] = new_idx
                viz_idx = new_idx

            # Get current visualization data
            data = viz_data[viz_idx]
            large_count = data.get('large_count', 0)
            small_count = data.get('small_count', 0)
            transition_str = f"{large_count}→{small_count}"
            imgui.text(f"Transition {viz_idx + 1}/{num_viz}: {transition_str}")

            scalar_large = data.get('scalar_large', 0)
            scalar_small = data.get('scalar_small', 0)
            imgui.text(f"Source (after split): {large_count} contours @ scalar {scalar_large:.4f}")
            imgui.text(f"Target (before split): {small_count} contours @ scalar {scalar_small:.4f}")
            imgui.separator()

            # Get contour data
            target_contours = data.get('target_contours_2d', [])
            source_contours = data.get('source_contours_2d', [])

            if not target_contours and not source_contours:
                imgui.text("No contour data available")
                imgui.end()
                continue

            # Collect all points to compute bounds
            all_points = []
            for c in target_contours:
                if c is not None and len(c) > 0:
                    all_points.extend(c)
            for c in source_contours:
                if c is not None and len(c) > 0:
                    all_points.extend(c)

            if len(all_points) < 3:
                imgui.text("Insufficient contour data")
                imgui.end()
                continue

            all_points = np.array(all_points)
            min_xy = all_points.min(axis=0)
            max_xy = all_points.max(axis=0)
            range_xy = max_xy - min_xy
            range_xy[range_xy < 1e-10] = 1.0
            max_range = max(range_xy[0], range_xy[1])
            margin = 0.1
            scale_factor = (1 - 2 * margin) / max_range
            center_xy = (min_xy + max_xy) / 2

            def to_screen(pt_2d, x0, y0, canvas_size):
                norm = (pt_2d - center_xy) * scale_factor + 0.5
                sx = x0 + norm[0] * canvas_size
                sy = y0 + (1 - norm[1]) * canvas_size  # Flip Y to match matplotlib
                return (sx, sy)

            # Canvas setup
            canvas_size = 380
            padding = 15

            draw_list = imgui.get_window_draw_list()
            cursor_pos = imgui.get_cursor_screen_pos()
            x0, y0 = cursor_pos[0] + padding, cursor_pos[1] + padding

            # Background
            draw_list.add_rect_filled(x0, y0, x0 + canvas_size, y0 + canvas_size,
                                     imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 1.0))

            # Draw TARGET contours (merged, before division) in BLUE
            target_color = imgui.get_color_u32_rgba(0.3, 0.5, 1.0, 1.0)
            target_fill = imgui.get_color_u32_rgba(0.2, 0.3, 0.6, 0.3)
            for contour_2d in target_contours:
                if contour_2d is None or len(contour_2d) < 3:
                    continue
                contour_2d = np.array(contour_2d)
                contour_screen = [to_screen(p, x0, y0, canvas_size) for p in contour_2d]
                # Fill
                for i in range(1, len(contour_screen) - 1):
                    draw_list.add_triangle_filled(
                        contour_screen[0][0], contour_screen[0][1],
                        contour_screen[i][0], contour_screen[i][1],
                        contour_screen[i+1][0], contour_screen[i+1][1],
                        target_fill)
                # Outline
                for i in range(len(contour_screen)):
                    p1, p2 = contour_screen[i], contour_screen[(i+1) % len(contour_screen)]
                    draw_list.add_line(p1[0], p1[1], p2[0], p2[1], target_color, 2.5)

            # Draw SOURCE contours (split, after division) in ORANGE/RED
            source_colors = [
                imgui.get_color_u32_rgba(1.0, 0.4, 0.1, 1.0),  # Orange
                imgui.get_color_u32_rgba(1.0, 0.8, 0.0, 1.0),  # Yellow
                imgui.get_color_u32_rgba(0.0, 1.0, 0.4, 1.0),  # Green
                imgui.get_color_u32_rgba(1.0, 0.0, 0.5, 1.0),  # Pink
                imgui.get_color_u32_rgba(0.6, 0.2, 1.0, 1.0),  # Purple
            ]
            for ci, contour_2d in enumerate(source_contours):
                if contour_2d is None or len(contour_2d) < 3:
                    continue
                contour_2d = np.array(contour_2d)
                contour_screen = [to_screen(p, x0, y0, canvas_size) for p in contour_2d]
                color = source_colors[ci % len(source_colors)]
                # Outline only (no fill to see overlap with target)
                for i in range(len(contour_screen)):
                    p1, p2 = contour_screen[i], contour_screen[(i+1) % len(contour_screen)]
                    draw_list.add_line(p1[0], p1[1], p2[0], p2[1], color, 2.0)

            imgui.dummy(canvas_size + 2 * padding, canvas_size + 2 * padding)

            # Legend
            imgui.separator()
            imgui.text_colored("Target (before split)", 0.3, 0.5, 1.0, 1.0)
            imgui.same_line()
            imgui.text(" | ")
            imgui.same_line()
            imgui.text_colored("Source (after split)", 1.0, 0.4, 0.1, 1.0)

            imgui.end()

        for name in muscles_to_close:
            self.neck_viz_open[name] = False

    def _render_manual_cut_windows(self):
        """Render manual cutting windows for muscles that need user input."""
        for name, obj in self.zygote_muscle_meshes.items():
            if not hasattr(obj, '_manual_cut_pending') or not obj._manual_cut_pending:
                continue

            if obj._manual_cut_data is None:
                continue

            # Initialize mouse state for this window
            if not hasattr(self, '_manual_cut_mouse'):
                self._manual_cut_mouse = {}
            if name not in self._manual_cut_mouse:
                self._manual_cut_mouse[name] = {
                    'dragging': False,
                    'start_pos': None,
                    'end_pos': None,
                    'zoom': 1.0,
                    'pan': [0.0, 0.0],
                    'panning': False,
                }

            mouse_state = self._manual_cut_mouse[name]

            # Window setup
            muscle_name = obj._manual_cut_data.get('muscle_name', name)
            target_i = obj._manual_cut_data.get('target_i', 0)
            source_indices = obj._manual_cut_data.get('source_indices', [])
            required_pieces_display = obj._manual_cut_data.get('required_pieces', 2)
            # Window size: main canvas (550+40) + gap (30) + source panel (280+30) + margins = ~960
            imgui.set_next_window_size(960, 780, imgui.FIRST_USE_EVER)
            # Show target index and source info in title for M->N cases
            title_suffix = f" (Target {target_i}, {len(source_indices)}->1)" if len(source_indices) > 0 else ""
            expanded, opened = imgui.begin(f"Manual Cut: {muscle_name}{title_suffix}", True)

            if not opened:
                obj._cancel_manual_cut()
                imgui.end()
                continue

            # Get data
            target_2d = obj._manual_cut_data['target_2d']
            source_2d_list = obj._manual_cut_data['source_2d_list']
            target_level = obj._manual_cut_data['target_level']
            source_level = obj._manual_cut_data['source_level']

            # Find initial line using obj._find_neck_in_contour (same as find_transitions)
            # Only for SEPARATE mode - COMMON mode uses source boundary line
            # Skip for sub-windows (original_source_indices set) - they don't need neck finding
            is_common_mode = obj._manual_cut_data.get('is_common_mode', False)
            original_source_indices = obj._manual_cut_data.get('original_source_indices', None)
            is_subwindow = original_source_indices is not None
            subcut_level = obj._manual_cut_data.get('subcut_level', 0)
            need_new_recommendation = 'initial_line' not in obj._manual_cut_data

            # Neck finding for SEPARATE mode only (multiple separate sources → one target)
            # Skip for: COMMON mode (sources share boundary), sub-windows, or sub-cuts
            skip_neck = is_common_mode or is_subwindow or subcut_level > 0
            if need_new_recommendation and not skip_neck:
                current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
                contour_range = np.max(target_2d.max(axis=0) - target_2d.min(axis=0))

                # Find ALL neck candidates with distance < 3% of perimeter
                # Only print debug once when actually computing candidates
                if 'neck_candidates' not in obj._manual_cut_data:
                    print(f"[NECK] Looking for neck candidates in {len(current_pieces)} pieces, target has {len(target_2d)} vertices")
                    all_candidates = []
                    for piece_idx, piece_2d in enumerate(current_pieces):
                        n = len(piece_2d)
                        if n < 10:
                            continue

                        # Compute perimeter
                        perimeter = 0
                        for i in range(n):
                            perimeter += np.linalg.norm(piece_2d[(i+1) % n] - piece_2d[i])

                        # Threshold: 3% of perimeter (wider to catch merge point necks)
                        neck_threshold = perimeter * 0.03
                        # Use smaller index separation (10%) to catch pinch points
                        min_sep_idx = int(n * 0.10)

                        # Find all pairs below threshold
                        for i in range(n):
                            for j in range(i + min_sep_idx, i + n - min_sep_idx):
                                j_mod = j % n
                                # Only add if i < j_mod to avoid duplicates (i,j) and (j,i)
                                if i >= j_mod:
                                    continue
                                dist = np.linalg.norm(piece_2d[i] - piece_2d[j_mod])
                                if dist < neck_threshold:
                                    all_candidates.append({
                                        'piece_idx': piece_idx,
                                        'idx_a': i,
                                        'idx_b': j_mod,
                                        'width': dist,
                                        'point': (piece_2d[i] + piece_2d[j_mod]) / 2,
                                        'pos_a': piece_2d[i].copy(),
                                        'pos_b': piece_2d[j_mod].copy(),
                                    })

                    # Sort by width (narrowest first)
                    all_candidates.sort(key=lambda x: x['width'])

                    # Remove near-duplicates (candidates within 5 vertices of each other)
                    filtered = []
                    for cand in all_candidates:
                        is_dup = False
                        for existing in filtered:
                            if existing['piece_idx'] == cand['piece_idx']:
                                # Check if indices are close (in same order or swapped)
                                n = len(current_pieces[cand['piece_idx']])
                                # Same order check
                                dist_a = min(abs(cand['idx_a'] - existing['idx_a']),
                                           n - abs(cand['idx_a'] - existing['idx_a']))
                                dist_b = min(abs(cand['idx_b'] - existing['idx_b']),
                                           n - abs(cand['idx_b'] - existing['idx_b']))
                                # Swapped order check
                                dist_a_swap = min(abs(cand['idx_a'] - existing['idx_b']),
                                                n - abs(cand['idx_a'] - existing['idx_b']))
                                dist_b_swap = min(abs(cand['idx_b'] - existing['idx_a']),
                                                n - abs(cand['idx_b'] - existing['idx_a']))
                                if (dist_a < 5 and dist_b < 5) or (dist_a_swap < 5 and dist_b_swap < 5):
                                    is_dup = True
                                    break
                        if not is_dup:
                            filtered.append(cand)

                    obj._manual_cut_data['neck_candidates'] = filtered
                    obj._manual_cut_data['selected_neck_idx'] = 0 if len(filtered) > 0 else -1
                    print(f"[NECK] Found {len(filtered)} neck candidates (from {len(all_candidates)} raw)")

                # Get current selection
                candidates = obj._manual_cut_data.get('neck_candidates', [])
                selected_idx = obj._manual_cut_data.get('selected_neck_idx', 0)

                # Helper to find line-contour intersection points
                def find_contour_intersections(line_start, line_end, contour):
                    """Find where a line intersects the contour, return the two intersection points."""
                    intersections = []
                    p1 = np.array(line_start)
                    p2 = np.array(line_end)
                    d = p2 - p1

                    for i in range(len(contour)):
                        q1 = contour[i]
                        q2 = contour[(i + 1) % len(contour)]
                        e = q2 - q1

                        denom = d[0] * e[1] - d[1] * e[0]
                        if abs(denom) < 1e-10:
                            continue

                        t = ((q1[0] - p1[0]) * e[1] - (q1[1] - p1[1]) * e[0]) / denom
                        s = ((q1[0] - p1[0]) * d[1] - (q1[1] - p1[1]) * d[0]) / denom

                        if 0 <= s <= 1:  # Intersection on contour edge
                            pt = p1 + t * d
                            intersections.append((t, pt, i))

                    intersections.sort(key=lambda x: x[0])
                    return intersections

                if len(candidates) > 0 and selected_idx >= 0 and selected_idx < len(candidates):
                    cand = candidates[selected_idx]
                    piece_idx = cand['piece_idx']
                    i0, i1 = cand['idx_a'], cand['idx_b']
                    piece_2d = current_pieces[piece_idx]
                    n = len(piece_2d)

                    # Get neck vertices
                    neck_a = piece_2d[i0]
                    neck_b = piece_2d[i1]

                    # Use neck vertices directly as cutting line endpoints
                    # They are already on the contour (vertices of piece_2d)
                    line_start = tuple(neck_a)
                    line_end = tuple(neck_b)

                    # Store neck info for zoomed view
                    obj._manual_cut_data['current_neck_info'] = {
                        'neck_a': neck_a.copy(),
                        'neck_b': neck_b.copy(),
                        'line_start': neck_a.copy(),
                        'line_end': neck_b.copy(),
                        'piece_idx': piece_idx,
                        'idx_a': i0,
                        'idx_b': i1,
                    }

                    obj._manual_cut_data['initial_line'] = (line_start, line_end)
                    # Only set is_neck_line = True if user hasn't drawn their own line
                    # (i.e., if _manual_cut_line is None or equals initial_line)
                    if obj._manual_cut_line is None:
                        obj._manual_cut_data['is_neck_line'] = True  # Flag for vertex-to-vertex cutting
                        obj._manual_cut_line = (line_start, line_end)
                    elif obj._manual_cut_line == obj._manual_cut_data.get('initial_line'):
                        # User hasn't changed the line from initial neck recommendation
                        obj._manual_cut_data['is_neck_line'] = True
                else:
                    # No candidates found - set flag to prevent re-computation every frame
                    if 'neck_search_done' not in obj._manual_cut_data:
                        print(f"[NECK] No neck candidates found")
                        obj._manual_cut_data['neck_search_done'] = True

            current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
            required_pieces = obj._manual_cut_data.get('required_pieces', 2)
            imgui.text(f"Target level: {target_level} | Source level: {source_level}")
            imgui.text(f"Pieces: {len(current_pieces)} / {required_pieces} required")
            imgui.text(f"Target verts: {len(target_2d)}")
            imgui.text_colored("WHITE = Target contour (cut this)", 1.0, 1.0, 1.0, 1.0)
            imgui.same_line()
            imgui.text_colored(" | FADED = Source ref (guide)", 0.6, 0.6, 0.6, 1.0)
            imgui.text("Draw a line to cut. Scroll to zoom, middle-drag to pan.")
            imgui.text(f"Zoom: {mouse_state['zoom']:.1f}x")

            # Neck candidate selector slider
            candidates = obj._manual_cut_data.get('neck_candidates', [])
            if len(candidates) > 0:
                selected_idx = obj._manual_cut_data.get('selected_neck_idx', 0)
                cand = candidates[selected_idx] if selected_idx < len(candidates) else None
                width_str = f"{cand['width']:.6f}" if cand else "N/A"
                imgui.text(f"Neck candidates: {len(candidates)} | Selected: {selected_idx} (width={width_str})")
                imgui.push_item_width(200)
                changed, new_idx = imgui.slider_int(f"##neck_slider_{name}", selected_idx, 0, len(candidates) - 1)
                if changed:
                    obj._manual_cut_data['selected_neck_idx'] = new_idx
                    # Update the cutting line to match selected candidate
                    if new_idx < len(candidates):
                        cand = candidates[new_idx]
                        piece_idx = cand['piece_idx']
                        i0, i1 = cand['idx_a'], cand['idx_b']
                        current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
                        piece_2d = current_pieces[piece_idx]
                        n = len(piece_2d)
                        contour_range = np.max(target_2d.max(axis=0) - target_2d.min(axis=0))

                        # Get neck vertices
                        neck_a = piece_2d[i0]
                        neck_b = piece_2d[i1]

                        # Determine line direction
                        direction = neck_b - neck_a
                        dir_len = np.linalg.norm(direction)
                        if dir_len > 1e-10:
                            direction = direction / dir_len
                        else:
                            # Zero-length (pinch): use direction from prev to prev
                            prev_i0 = (i0 - 1) % n
                            prev_i1 = (i1 - 1) % n
                            direction = piece_2d[prev_i1] - piece_2d[prev_i0]
                            d_len = np.linalg.norm(direction)
                            if d_len > 1e-10:
                                direction = direction / d_len
                            else:
                                direction = np.array([1.0, 0.0])

                        # Create extended line to find edge intersections
                        neck_width = np.linalg.norm(neck_b - neck_a)
                        extension = max(neck_width * 0.5, contour_range * 0.10)
                        ext_start = neck_a - direction * extension
                        ext_end = neck_b + direction * extension

                        # Find actual contour edge intersections
                        def find_inters_inline(ls, le, contour):
                            inters = []
                            p1, p2 = np.array(ls), np.array(le)
                            d = p2 - p1
                            for ci in range(len(contour)):
                                q1, q2 = contour[ci], contour[(ci + 1) % len(contour)]
                                e = q2 - q1
                                denom = d[0] * e[1] - d[1] * e[0]
                                if abs(denom) < 1e-10:
                                    continue
                                t = ((q1[0] - p1[0]) * e[1] - (q1[1] - p1[1]) * e[0]) / denom
                                s = ((q1[0] - p1[0]) * d[1] - (q1[1] - p1[1]) * d[0]) / denom
                                if 0 <= s <= 1:
                                    inters.append((t, p1 + t * d, ci))
                            return inters

                        intersections = find_inters_inline(ext_start, ext_end, piece_2d)

                        if len(intersections) >= 2:
                            # Find two intersections closest to neck center
                            neck_center = (neck_a + neck_b) / 2
                            inters_with_dist = [(np.linalg.norm(pt - neck_center), t, pt, ei)
                                               for t, pt, ei in intersections]
                            inters_with_dist.sort(key=lambda x: x[0])
                            if len(inters_with_dist) >= 2:
                                _, _, p_start, _ = inters_with_dist[0]
                                _, _, p_end, _ = inters_with_dist[1]
                            else:
                                p_start, p_end = ext_start, ext_end
                        else:
                            p_start, p_end = ext_start, ext_end

                        # Update cutting line
                        obj._manual_cut_line = (tuple(p_start), tuple(p_end))
                        obj._manual_cut_data['initial_line'] = obj._manual_cut_line

                        # Update neck info for zoomed view
                        obj._manual_cut_data['current_neck_info'] = {
                            'neck_a': neck_a.copy(),
                            'neck_b': neck_b.copy(),
                            'line_start': np.array(p_start).copy(),
                            'line_end': np.array(p_end).copy(),
                            'piece_idx': piece_idx,
                            'idx_a': i0,
                            'idx_b': i1,
                        }
                        # User selected a neck via slider - use neck-based cutting
                        obj._manual_cut_data['is_neck_line'] = True
                imgui.pop_item_width()
            imgui.separator()

            # Compute bounds for normalization (include target AND projected sources)
            all_points_2d = [target_2d]

            # Project all source contours onto target plane and include in bounds
            source_contours_3d_for_bounds = obj._manual_cut_data.get('source_contours', [])
            source_bps_for_bounds = obj._manual_cut_data.get('source_bps', [])
            target_bp = obj._manual_cut_data['target_bp']
            target_mean_bounds = target_bp['mean']
            # Negate basis vectors to rotate 180° - match cutting window orientation
            target_x_bounds = -target_bp['basis_x']
            target_y_bounds = -target_bp['basis_y']
            target_z_bounds = target_bp['basis_z']

            for src_contour_3d in source_contours_3d_for_bounds:
                src_projected = []
                for pt in src_contour_3d:
                    diff = pt - target_mean_bounds
                    proj_pt = pt - np.dot(diff, target_z_bounds) * target_z_bounds
                    diff_proj = proj_pt - target_mean_bounds
                    x_coord = np.dot(diff_proj, target_x_bounds)
                    y_coord = np.dot(diff_proj, target_y_bounds)
                    src_projected.append([x_coord, y_coord])
                if len(src_projected) > 0:
                    all_points_2d.append(np.array(src_projected))

            # Include transformed sources from optimization in bounds (if available)
            transformed_sources = obj._manual_cut_data.get('transformed_sources_2d', None) or []
            for src_2d in transformed_sources:
                if src_2d is not None and len(src_2d) >= 3:
                    all_points_2d.append(np.array(src_2d))

            # Compute combined bounds
            all_points_combined = np.vstack(all_points_2d)
            min_xy = all_points_combined.min(axis=0)
            max_xy = all_points_combined.max(axis=0)
            range_xy = max_xy - min_xy
            max_range = max(range_xy) * 1.1  # Small padding

            # Canvas setup
            canvas_size = 550
            padding = 20
            draw_list = imgui.get_window_draw_list()
            cursor_pos = imgui.get_cursor_screen_pos()
            x0, y0 = cursor_pos[0] + padding, cursor_pos[1] + padding

            # Get zoom and pan from mouse state
            zoom = mouse_state['zoom']
            pan = mouse_state['pan']

            # Coordinate transform functions (with zoom and pan)
            # Y flip required: matplotlib Y goes up, screen Y goes down
            def to_screen(p, x0, y0, canvas_size):
                center = (min_xy + max_xy) / 2
                normalized = (np.array(p) - center) / max_range + 0.5
                # Apply zoom and pan
                normalized = (normalized - 0.5) * zoom + 0.5 + np.array(pan)
                return (x0 + normalized[0] * canvas_size,
                        y0 + (1.0 - normalized[1]) * canvas_size)  # Flip Y to match matplotlib

            def from_screen(sx, sy, x0, y0, canvas_size):
                normalized_x = (sx - x0) / canvas_size
                normalized_y = 1.0 - (sy - y0) / canvas_size  # Flip Y back
                # Reverse zoom and pan
                normalized = np.array([normalized_x, normalized_y])
                normalized = (normalized - 0.5 - np.array(pan)) / zoom + 0.5
                center = (min_xy + max_xy) / 2
                return center + (normalized - 0.5) * max_range

            def find_line_contour_intersections(line_start, line_end, contour_2d):
                """Find intersection points of a line with a contour polygon."""
                intersections = []
                p1 = np.array(line_start)
                p2 = np.array(line_end)
                d = p2 - p1

                for i in range(len(contour_2d)):
                    q1 = contour_2d[i]
                    q2 = contour_2d[(i + 1) % len(contour_2d)]
                    e = q2 - q1

                    # Solve p1 + t*d = q1 + s*e
                    denom = d[0] * e[1] - d[1] * e[0]
                    if abs(denom) < 1e-10:
                        continue

                    t = ((q1[0] - p1[0]) * e[1] - (q1[1] - p1[1]) * e[0]) / denom
                    s = ((q1[0] - p1[0]) * d[1] - (q1[1] - p1[1]) * d[0]) / denom

                    if 0 <= s <= 1:  # Intersection on contour edge
                        pt = p1 + t * d
                        intersections.append((t, pt))

                # Sort by t parameter and return points
                intersections.sort(key=lambda x: x[0])
                return [pt for _, pt in intersections]

            # Draw canvas background
            draw_list.add_rect_filled(x0 - padding, y0 - padding,
                                      x0 + canvas_size + padding, y0 + canvas_size + padding,
                                      imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))
            draw_list.add_rect(x0 - padding, y0 - padding,
                              x0 + canvas_size + padding, y0 + canvas_size + padding,
                              imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 1.0))

            # Clip drawing to canvas area
            draw_list.push_clip_rect(x0 - padding, y0 - padding,
                                     x0 + canvas_size + padding, y0 + canvas_size + padding)

            # Draw all current pieces
            # Colors for different pieces: cycle through these
            piece_colors = [
                (1.0, 1.0, 1.0),  # White (first piece / uncut)
                (0.2, 0.6, 1.0),  # Blue
                (1.0, 0.4, 0.2),  # Orange
                (0.2, 1.0, 0.4),  # Green
                (1.0, 0.8, 0.2),  # Yellow
                (0.8, 0.2, 1.0),  # Purple
            ]
            current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
            required_pieces = obj._manual_cut_data.get('required_pieces', 2)

            for piece_idx, piece_2d in enumerate(current_pieces):
                if len(piece_2d) >= 3:
                    piece_screen = [to_screen(p, x0, y0, canvas_size) for p in piece_2d]
                    color = piece_colors[piece_idx % len(piece_colors)]
                    for i in range(len(piece_screen)):
                        p1, p2 = piece_screen[i], piece_screen[(i+1) % len(piece_screen)]
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                          imgui.get_color_u32_rgba(*color, 1.0), 3.0)

                    # Add label at centroid
                    centroid = np.mean(piece_2d, axis=0)
                    cx, cy = to_screen(centroid, x0, y0, canvas_size)
                    if len(current_pieces) > 1:
                        # Multiple pieces - show piece index with colored background
                        label = f"P{piece_idx}"
                        # Draw background rect for visibility
                        draw_list.add_rect_filled(cx - 12, cy - 10, cx + 12, cy + 10,
                                                  imgui.get_color_u32_rgba(*color, 0.8))
                        draw_list.add_text(cx - 8, cy - 7,
                                          imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 1.0), label)
                    else:
                        # Single piece - show as target
                        draw_list.add_text(cx + 10, cy - 10,
                                          imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0),
                                          f"Target (Lv.{target_level})")

            # Draw transformed source contours after optimization (semi-transparent dashed outline)
            transformed_sources = obj._manual_cut_data.get('transformed_sources_2d', None) or []
            if len(transformed_sources) > 0:
                # Debug: print once when drawing
                if not hasattr(obj, '_debug_transformed_printed') or obj._debug_transformed_printed != len(transformed_sources):
                    obj._debug_transformed_printed = len(transformed_sources)
                    print(f"[Draw DEBUG] transformed_sources: {len(transformed_sources)}")
                    for si, src in enumerate(transformed_sources):
                        if src is not None and len(src) > 0:
                            src_arr = np.array(src)
                            print(f"  S{si}: {len(src)} verts, first=[{src_arr[0,0]:.4f},{src_arr[0,1]:.4f}], centroid=[{src_arr.mean(axis=0)[0]:.4f},{src_arr.mean(axis=0)[1]:.4f}]")
                for src_idx, src_2d in enumerate(transformed_sources):
                    if src_2d is not None and len(src_2d) >= 3:
                        src_color = piece_colors[src_idx % len(piece_colors)]
                        # Draw as dashed lines (semi-transparent)
                        for i in range(len(src_2d)):
                            p1 = to_screen(src_2d[i], x0, y0, canvas_size)
                            p2 = to_screen(src_2d[(i + 1) % len(src_2d)], x0, y0, canvas_size)
                            # Draw every other segment for dashed effect
                            if i % 2 == 0:
                                draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                                  imgui.get_color_u32_rgba(*src_color, 0.5), 1.5)
                        # Draw centroid marker with S label
                        src_centroid = np.mean(src_2d, axis=0)
                        scx, scy = to_screen(src_centroid, x0, y0, canvas_size)
                        draw_list.add_circle(scx, scy, 8, imgui.get_color_u32_rgba(*src_color, 0.7), thickness=2.0)
                        draw_list.add_text(scx - 5, scy - 6,
                                          imgui.get_color_u32_rgba(*src_color, 0.9), f"S{src_idx}")

                # Draw actual cut points on target contour (where cutting lines cross target)
                # These are the real cut locations, not the source shared boundaries
                target_crossings = getattr(obj, '_target_cut_crossings_2d', [])

                line_colors = [
                    imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0),  # Magenta
                    imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0),  # Cyan
                    imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0),  # Yellow
                    imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0),  # Orange
                ]

                if target_crossings:
                    # Group crossings by pair indices
                    crossings_by_pair = {}
                    for pt_2d, pair_indices in target_crossings:
                        if pair_indices not in crossings_by_pair:
                            crossings_by_pair[pair_indices] = []
                        crossings_by_pair[pair_indices].append(pt_2d)

                    # Draw crossings for each pair
                    for idx, (pair_indices, pts) in enumerate(crossings_by_pair.items()):
                        line_color = line_colors[idx % len(line_colors)]
                        # Draw each crossing point
                        for pt_2d in pts:
                            sp = to_screen(pt_2d, x0, y0, canvas_size)
                            draw_list.add_circle_filled(sp[0], sp[1], 6, line_color)
                        # If we have 2 crossings, draw line between them
                        if len(pts) >= 2:
                            p1 = to_screen(pts[0], x0, y0, canvas_size)
                            p2 = to_screen(pts[-1], x0, y0, canvas_size)
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1], line_color, 2.0)
                            mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                            draw_list.add_text(mid_x + 5, mid_y - 15, line_color, f"Cut {pair_indices[0]}-{pair_indices[1]}")
                else:
                    # Fallback to old method using shared edges
                    shared_edges_list = []
                    if hasattr(obj, '_shared_cut_edges_2d') and obj._shared_cut_edges_2d:
                        shared_edges_list = obj._shared_cut_edges_2d
                    elif hasattr(obj, '_shared_cut_edge_2d') and obj._shared_cut_edge_2d is not None:
                        shared_edges_list = [((0, 1), obj._shared_cut_edge_2d)]

                    for idx, (pair_indices, shared_edge) in enumerate(shared_edges_list):
                        if shared_edge is not None and len(shared_edge) >= 2:
                            line_color = line_colors[idx % len(line_colors)]
                            p1 = to_screen(shared_edge[0], x0, y0, canvas_size)
                            p2 = to_screen(shared_edge[-1], x0, y0, canvas_size)
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1], line_color, 2.0)
                            draw_list.add_circle_filled(p1[0], p1[1], 4, line_color)
                            draw_list.add_circle_filled(p2[0], p2[1], 4, line_color)
                            mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                            draw_list.add_text(mid_x + 5, mid_y - 15, line_color, f"Cut {pair_indices[0]}-{pair_indices[1]}")

            # Draw all neck candidates (faded) and highlight selected one
            candidates = obj._manual_cut_data.get('neck_candidates', [])
            selected_neck_idx = obj._manual_cut_data.get('selected_neck_idx', 0)
            for cand_idx, cand in enumerate(candidates):
                pos_a = cand['pos_a']
                pos_b = cand['pos_b']
                screen_a = to_screen(pos_a, x0, y0, canvas_size)
                screen_b = to_screen(pos_b, x0, y0, canvas_size)
                mid_screen = ((screen_a[0] + screen_b[0]) / 2, (screen_a[1] + screen_b[1]) / 2)

                if cand_idx == selected_neck_idx:
                    # Selected: bright cyan, thicker line
                    color = imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0)
                    draw_list.add_line(screen_a[0], screen_a[1], screen_b[0], screen_b[1], color, 3.0)
                    draw_list.add_circle_filled(screen_a[0], screen_a[1], 5, color)
                    draw_list.add_circle_filled(screen_b[0], screen_b[1], 5, color)
                    # Label
                    draw_list.add_text(mid_screen[0] + 5, mid_screen[1] - 10,
                                      color, f"#{cand_idx} w={cand['width']:.4f}")
                else:
                    # Non-selected: faded yellow
                    color = imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 0.3)
                    draw_list.add_line(screen_a[0], screen_a[1], screen_b[0], screen_b[1], color, 1.5)
                    draw_list.add_circle_filled(screen_a[0], screen_a[1], 3, color)
                    draw_list.add_circle_filled(screen_b[0], screen_b[1], 3, color)

            # Draw zoomed view indicator rectangle on main canvas
            neck_info_indicator = obj._manual_cut_data.get('current_neck_info', None)
            if neck_info_indicator is not None:
                line_start_ind = neck_info_indicator['line_start']
                line_end_ind = neck_info_indicator['line_end']
                piece_idx_ind = neck_info_indicator['piece_idx']
                current_pieces_ind = obj._manual_cut_data.get('current_pieces', [target_2d])

                if piece_idx_ind < len(current_pieces_ind):
                    piece_2d_ind = current_pieces_ind[piece_idx_ind]

                    # Calculate the same extent as the zoomed view
                    target_line_pixels = 120.0
                    zoom_canvas_size_ind = 200
                    actual_line_length_ind = np.linalg.norm(line_end_ind - line_start_ind)
                    if actual_line_length_ind > 1e-10:
                        world_extent_ind = actual_line_length_ind * (zoom_canvas_size_ind / target_line_pixels)
                    else:
                        world_extent_ind = np.max(piece_2d_ind.max(axis=0) - piece_2d_ind.min(axis=0)) * 0.3

                    neck_center_ind = (line_start_ind + line_end_ind) / 2
                    half_extent = world_extent_ind / 2

                    # Get the corners of the zoomed view in world coordinates
                    corners_2d = [
                        neck_center_ind + np.array([-half_extent, -half_extent]),
                        neck_center_ind + np.array([half_extent, -half_extent]),
                        neck_center_ind + np.array([half_extent, half_extent]),
                        neck_center_ind + np.array([-half_extent, half_extent]),
                    ]

                    # Convert to screen coordinates and draw
                    corners_screen = [to_screen(c, x0, y0, canvas_size) for c in corners_2d]
                    indicator_color = imgui.get_color_u32_rgba(0.0, 1.0, 0.5, 0.7)
                    for i in range(4):
                        p1 = corners_screen[i]
                        p2 = corners_screen[(i + 1) % 4]
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1], indicator_color, 2.0)

            # Draw selected source contour transparently on the main canvas
            source_contours_3d = obj._manual_cut_data.get('source_contours', [])
            source_bps = obj._manual_cut_data.get('source_bps', [])
            selected_sources = obj._manual_cut_data.get('selected_sources', list(range(len(source_contours_3d))))
            source_labels = obj._manual_cut_data.get('source_labels', list(range(len(source_contours_3d))))

            # Use pre-computed source_2d_list for drawing (same as PNG visualization)
            source_2d_list = obj._manual_cut_data.get('source_2d_list', [])

            if 'source_view_idx' in mouse_state and len(source_2d_list) > 0:
                src_idx = mouse_state.get('source_view_idx', 0)
                if src_idx < len(source_2d_list) and src_idx < len(source_bps):
                    src_bp = source_bps[src_idx]
                    # Use pre-computed 2D source contour (matches PNG visualization)
                    src_on_target_2d = np.array(source_2d_list[src_idx])

                    # Draw source contour transparently on main canvas
                    src_color = piece_colors[src_idx % len(piece_colors)]
                    is_selected = src_idx in selected_sources
                    alpha = 0.6 if is_selected else 0.25
                    thickness = 2.5 if is_selected else 1.5

                    if len(src_on_target_2d) >= 3:
                        # Convert all points to screen coordinates
                        screen_pts = [to_screen(src_on_target_2d[i], x0, y0, canvas_size) for i in range(len(src_on_target_2d))]

                        # Fill polygon first (like PNG visualization does)
                        # Use triangulation for non-convex polygons
                        fill_alpha = 0.15 if is_selected else 0.05
                        try:
                            # Simple triangle fan from centroid for fill
                            src_centroid_screen = to_screen(src_on_target_2d.mean(axis=0), x0, y0, canvas_size)
                            for i in range(len(screen_pts)):
                                p1 = screen_pts[i]
                                p2 = screen_pts[(i + 1) % len(screen_pts)]
                                draw_list.add_triangle_filled(
                                    src_centroid_screen[0], src_centroid_screen[1],
                                    p1[0], p1[1], p2[0], p2[1],
                                    imgui.get_color_u32_rgba(*src_color, fill_alpha))
                        except:
                            pass  # Skip fill if it fails

                        # Draw outline
                        for i in range(len(screen_pts)):
                            p1 = screen_pts[i]
                            p2 = screen_pts[(i + 1) % len(screen_pts)]
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                              imgui.get_color_u32_rgba(*src_color, alpha), thickness)

                        # Draw centroid marker
                        src_centroid = src_on_target_2d.mean(axis=0)
                        cx, cy = to_screen(src_centroid, x0, y0, canvas_size)
                        draw_list.add_circle_filled(cx, cy, 6, imgui.get_color_u32_rgba(*src_color, alpha))
                        # Add X marker if not selected
                        if not is_selected:
                            draw_list.add_line(cx - 4, cy - 4, cx + 4, cy + 4, imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 0.8), 2)
                            draw_list.add_line(cx - 4, cy + 4, cx + 4, cy - 4, imgui.get_color_u32_rgba(1.0, 0.3, 0.3, 0.8), 2)
                        # Add source label near centroid (use original index from source_labels)
                        src_label = source_labels[src_idx] if src_idx < len(source_labels) else src_idx
                        draw_list.add_text(cx + 10, cy + 5,
                                          imgui.get_color_u32_rgba(*src_color, 1.0),
                                          f"Source {src_label} (Lv.{source_level})")

            # End canvas clipping
            draw_list.pop_clip_rect()

            # Create invisible button to capture mouse input (prevents window dragging)
            imgui.set_cursor_screen_pos((x0 - padding, y0 - padding))
            imgui.invisible_button(f"canvas##{name}", canvas_size + 2 * padding, canvas_size + 2 * padding)
            canvas_hovered = imgui.is_item_hovered()
            canvas_active = imgui.is_item_active()

            # Handle mouse interaction for drawing line
            mouse_pos = imgui.get_mouse_pos()

            # Mouse wheel zoom
            if canvas_hovered:
                io = imgui.get_io()
                if io.mouse_wheel != 0:
                    zoom_factor = 1.1 if io.mouse_wheel > 0 else 0.9
                    # Zoom towards mouse position
                    mouse_norm_x = (mouse_pos[0] - x0) / canvas_size
                    mouse_norm_y = 1.0 - (mouse_pos[1] - y0) / canvas_size
                    # Adjust pan to zoom towards mouse
                    old_zoom = mouse_state['zoom']
                    new_zoom = old_zoom * zoom_factor
                    new_zoom = max(0.5, min(10.0, new_zoom))  # Clamp zoom
                    if new_zoom != old_zoom:
                        # Pan adjustment to keep mouse point fixed
                        mouse_state['pan'][0] += (mouse_norm_x - 0.5) * (1 - zoom_factor / old_zoom * new_zoom) / new_zoom
                        mouse_state['pan'][1] += (mouse_norm_y - 0.5) * (1 - zoom_factor / old_zoom * new_zoom) / new_zoom
                        mouse_state['zoom'] = new_zoom

            # Middle mouse or right mouse pan
            if canvas_hovered and (imgui.is_mouse_clicked(2) or imgui.is_mouse_clicked(1)):  # Middle or right button
                mouse_state['panning'] = True
                mouse_state['pan_button'] = 2 if imgui.is_mouse_clicked(2) else 1  # Track which button
                mouse_state['pan_start'] = (mouse_pos[0], mouse_pos[1])
                mouse_state['pan_orig'] = mouse_state['pan'].copy()

            if mouse_state.get('panning', False):
                dx = (mouse_pos[0] - mouse_state['pan_start'][0]) / canvas_size
                dy = -(mouse_pos[1] - mouse_state['pan_start'][1]) / canvas_size  # Flip Y
                mouse_state['pan'][0] = mouse_state['pan_orig'][0] + dx
                mouse_state['pan'][1] = mouse_state['pan_orig'][1] + dy
                pan_button = mouse_state.get('pan_button', 2)
                if imgui.is_mouse_released(pan_button):
                    mouse_state['panning'] = False

            # Left click for drawing cut line (disabled in optimization preview mode)
            is_preview_mode = obj._manual_cut_data.get('optimization_preview', False)
            if canvas_hovered and imgui.is_mouse_clicked(0) and not is_preview_mode:
                mouse_state['dragging'] = True
                mouse_state['start_pos'] = (mouse_pos[0], mouse_pos[1])
                mouse_state['end_pos'] = (mouse_pos[0], mouse_pos[1])

            if mouse_state['dragging'] and not is_preview_mode:
                end_x, end_y = mouse_pos[0], mouse_pos[1]
                # Check if shift is pressed for axis-aligned line
                shift_pressed = (glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                                 glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
                if shift_pressed:
                    start_x, start_y = mouse_state['start_pos']
                    dx = abs(end_x - start_x)
                    dy = abs(end_y - start_y)
                    # Snap to horizontal or vertical based on which direction is dominant
                    if dx > dy:
                        end_y = start_y  # Horizontal line
                    else:
                        end_x = start_x  # Vertical line
                mouse_state['end_pos'] = (end_x, end_y)
                if imgui.is_mouse_released(0):
                    mouse_state['dragging'] = False
                    # Convert to 2D coordinates and store the line
                    start_2d = from_screen(mouse_state['start_pos'][0], mouse_state['start_pos'][1],
                                          x0, y0, canvas_size)
                    end_2d = from_screen(mouse_state['end_pos'][0], mouse_state['end_pos'][1],
                                        x0, y0, canvas_size)
                    obj._manual_cut_line = (tuple(start_2d), tuple(end_2d))
                    obj._manual_cut_data['is_neck_line'] = False  # User drew this line manually

            # Clip cutting line drawing to canvas area
            draw_list.push_clip_rect(x0 - padding, y0 - padding,
                                     x0 + canvas_size + padding, y0 + canvas_size + padding)

            # Draw the cutting line (from click/drag points)
            if mouse_state['dragging'] and mouse_state['start_pos'] and mouse_state['end_pos']:
                # Draw line from drag start to current position
                draw_list.add_line(mouse_state['start_pos'][0], mouse_state['start_pos'][1],
                                  mouse_state['end_pos'][0], mouse_state['end_pos'][1],
                                  imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 0.9), 3.0)
            elif obj._manual_cut_line is not None:
                # Draw stored line
                start_2d, end_2d = obj._manual_cut_line
                start_screen = to_screen(start_2d, x0, y0, canvas_size)
                end_screen = to_screen(end_2d, x0, y0, canvas_size)
                draw_list.add_line(start_screen[0], start_screen[1],
                                  end_screen[0], end_screen[1],
                                  imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 0.9), 3.0)

                # Draw cut preview if we have a valid line
                piece0_2d, piece1_2d = obj._compute_cut_preview()
                if piece0_2d is not None and piece1_2d is not None:
                    # Draw piece 0 (color 0)
                    if len(piece0_2d) >= 3:
                        p0_screen = [to_screen(p, x0, y0, canvas_size) for p in piece0_2d]
                        for i in range(len(p0_screen)):
                            p1, p2 = p0_screen[i], p0_screen[(i+1) % len(p0_screen)]
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                              imgui.get_color_u32_rgba(piece_colors[0][0], piece_colors[0][1], piece_colors[0][2], 1.0), 2.5)
                    # Draw piece 1 (color 1)
                    if len(piece1_2d) >= 3:
                        p1_screen = [to_screen(p, x0, y0, canvas_size) for p in piece1_2d]
                        for i in range(len(p1_screen)):
                            pt1, pt2 = p1_screen[i], p1_screen[(i+1) % len(p1_screen)]
                            draw_list.add_line(pt1[0], pt1[1], pt2[0], pt2[1],
                                              imgui.get_color_u32_rgba(piece_colors[1][0], piece_colors[1][1], piece_colors[1][2], 1.0), 2.5)

            # End cutting line clipping
            draw_list.pop_clip_rect()

            # ===== Source Contour Viewer Panel (right side) =====
            source_contours_3d = obj._manual_cut_data.get('source_contours', [])
            source_labels = obj._manual_cut_data.get('source_labels', list(range(len(source_contours_3d))))
            if len(source_contours_3d) > 0:
                # Initialize source viewer state
                if 'source_view_idx' not in mouse_state:
                    mouse_state['source_view_idx'] = 0

                # Source viewer panel position (to the right of main canvas)
                src_panel_x = x0 + canvas_size + padding + 30
                src_panel_y = y0 - padding
                src_canvas_size = 280
                src_padding = 15

                # Draw panel background (taller to fit checkbox and summary)
                draw_list.add_rect_filled(
                    src_panel_x - src_padding, src_panel_y - src_padding,
                    src_panel_x + src_canvas_size + src_padding, src_panel_y + src_canvas_size + src_padding + 120,
                    imgui.get_color_u32_rgba(0.12, 0.12, 0.12, 1.0))
                draw_list.add_rect(
                    src_panel_x - src_padding, src_panel_y - src_padding,
                    src_panel_x + src_canvas_size + src_padding, src_panel_y + src_canvas_size + src_padding + 120,
                    imgui.get_color_u32_rgba(0.4, 0.4, 0.4, 1.0))

                # Title with level info
                draw_list.add_text(src_panel_x, src_panel_y - src_padding + 5,
                                  imgui.get_color_u32_rgba(0.9, 0.9, 0.9, 1.0),
                                  f"Source Contours (Lv.{source_level}, {len(source_contours_3d)} total)")

                # Adjust canvas position for title
                src_canvas_y = src_panel_y + 20

                # Get current source contour to display
                src_idx = mouse_state['source_view_idx']
                src_idx = max(0, min(src_idx, len(source_contours_3d) - 1))
                mouse_state['source_view_idx'] = src_idx

                # Project source contour to 2D using its bounding plane
                source_bps = obj._manual_cut_data.get('source_bps', [])
                if src_idx < len(source_bps):
                    src_bp = source_bps[src_idx]
                    src_contour_3d = source_contours_3d[src_idx]
                    src_mean = src_bp['mean']
                    src_basis_x = src_bp['basis_x']
                    src_basis_y = src_bp['basis_y']

                    # Project to 2D using TARGET's basis for consistent rotation with cutting panel
                    target_bp_for_src = obj._manual_cut_data.get('target_bp')
                    if target_bp_for_src is not None:
                        tgt_mean = target_bp_for_src['mean']
                        tgt_basis_x = target_bp_for_src['basis_x']
                        tgt_basis_y = target_bp_for_src['basis_y']
                        tgt_basis_z = target_bp_for_src['basis_z']
                        # Project source onto target plane, then to 2D
                        src_2d = []
                        for pt in src_contour_3d:
                            diff = pt - tgt_mean
                            dist_along_normal = np.dot(diff, tgt_basis_z)
                            pt_on_plane = pt - dist_along_normal * tgt_basis_z
                            diff_on_plane = pt_on_plane - tgt_mean
                            x = np.dot(diff_on_plane, tgt_basis_x)
                            y = np.dot(diff_on_plane, tgt_basis_y)
                            src_2d.append([x, y])
                        src_2d = np.array(src_2d)
                    else:
                        # Fallback to source's own basis
                        src_2d = []
                        for pt in src_contour_3d:
                            diff = pt - src_mean
                            x = np.dot(diff, src_basis_x)
                            y = np.dot(diff, src_basis_y)
                            src_2d.append([x, y])
                        src_2d = np.array(src_2d)

                    # Compute bounds for source contour
                    src_min = src_2d.min(axis=0)
                    src_max = src_2d.max(axis=0)
                    src_range = src_max - src_min
                    src_max_range = max(src_range) * 1.2 if max(src_range) > 0 else 1.0

                    # Transform function for source canvas
                    def src_to_screen(p):
                        center = (src_min + src_max) / 2
                        normalized = (np.array(p) - center) / src_max_range + 0.5
                        return (src_panel_x + normalized[0] * src_canvas_size,
                                src_canvas_y + (1.0 - normalized[1]) * src_canvas_size)

                    # Draw source canvas background
                    draw_list.add_rect_filled(
                        src_panel_x, src_canvas_y,
                        src_panel_x + src_canvas_size, src_canvas_y + src_canvas_size,
                        imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0))

                    # Draw source contour with matching color
                    src_color = piece_colors[src_idx % len(piece_colors)]
                    if len(src_2d) >= 3:
                        for i in range(len(src_2d)):
                            p1 = src_to_screen(src_2d[i])
                            p2 = src_to_screen(src_2d[(i + 1) % len(src_2d)])
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                              imgui.get_color_u32_rgba(*src_color, 1.0), 2.5)

                    # Draw centroid marker
                    src_centroid = src_2d.mean(axis=0)
                    cx, cy = src_to_screen(src_centroid)
                    draw_list.add_circle_filled(cx, cy, 5, imgui.get_color_u32_rgba(*src_color, 0.8))

                # Navigation controls below the source canvas
                nav_y = src_canvas_y + src_canvas_size + 10
                imgui.set_cursor_screen_pos((src_panel_x, nav_y))

                # Left/Right buttons and slider
                imgui.push_item_width(src_canvas_size)
                if len(source_contours_3d) > 1:
                    # Left button
                    if imgui.button(f"<##{name}_src_prev", 30, 25):
                        mouse_state['source_view_idx'] = max(0, src_idx - 1)
                    imgui.same_line()

                    # Slider
                    imgui.push_item_width(src_canvas_size - 80)
                    changed, new_idx = imgui.slider_int(f"##{name}_src_slider", src_idx, 0, len(source_contours_3d) - 1)
                    if changed:
                        mouse_state['source_view_idx'] = new_idx
                    imgui.pop_item_width()
                    imgui.same_line()

                    # Right button
                    if imgui.button(f">##{name}_src_next", 30, 25):
                        mouse_state['source_view_idx'] = min(len(source_contours_3d) - 1, src_idx + 1)

                # Show source index label (use original index from source_labels)
                imgui.set_cursor_screen_pos((src_panel_x, nav_y + 30))
                src_label_color = piece_colors[src_idx % len(piece_colors)]
                src_orig_label = source_labels[src_idx] if src_idx < len(source_labels) else src_idx
                imgui.text_colored(f"Source {src_orig_label} ({src_idx + 1}/{len(source_contours_3d)})", *src_label_color, 1.0)

                # Checkbox to include/exclude this source
                imgui.set_cursor_screen_pos((src_panel_x, nav_y + 50))

                # Initialize selected_sources if not present
                if 'selected_sources' not in obj._manual_cut_data:
                    obj._manual_cut_data['selected_sources'] = list(range(len(source_contours_3d)))

                selected_sources = obj._manual_cut_data['selected_sources']
                is_selected = src_idx in selected_sources

                changed, new_selected = imgui.checkbox(f"Include S{src_orig_label}##{name}", is_selected)
                if changed:
                    if new_selected and src_idx not in selected_sources:
                        selected_sources.append(src_idx)
                        selected_sources.sort()
                    elif not new_selected and src_idx in selected_sources:
                        # Don't allow deselecting if only 1 source remains
                        if len(selected_sources) > 1:
                            selected_sources.remove(src_idx)
                        else:
                            print("Cannot deselect: at least 1 source required")

                    # Update required_pieces and reset cuts
                    obj._manual_cut_data['selected_sources'] = selected_sources
                    new_required = len(selected_sources)
                    obj._manual_cut_data['required_pieces'] = new_required

                    # Reset current pieces to original target
                    obj._manual_cut_data['current_pieces'] = [target_2d.copy()]
                    obj._manual_cut_data['current_pieces_3d'] = [obj._manual_cut_data['target_contour'].copy()]
                    obj._manual_cut_data['cut_lines'] = []
                    if 'initial_line' in obj._manual_cut_data:
                        del obj._manual_cut_data['initial_line']
                    obj._manual_cut_line = None

                    print(f"Source selection changed: {len(selected_sources)} sources selected, need {new_required} pieces")

                # Show selection summary
                imgui.set_cursor_screen_pos((src_panel_x, nav_y + 70))
                num_selected = len(obj._manual_cut_data.get('selected_sources', []))
                imgui.text(f"Selected: {num_selected} / {len(source_contours_3d)}")

                imgui.pop_item_width()

                # ===== Zoomed Neck View Panel (below source panel) =====
                neck_info = obj._manual_cut_data.get('current_neck_info', None)
                if neck_info is not None:
                    # Panel position (below source panel)
                    zoom_panel_x = src_panel_x - src_padding
                    zoom_panel_y = src_panel_y + src_canvas_size + src_padding + 130
                    zoom_canvas_size = 200
                    zoom_padding = 10

                    # Draw panel background
                    draw_list.add_rect_filled(
                        zoom_panel_x, zoom_panel_y,
                        zoom_panel_x + src_canvas_size + src_padding * 2, zoom_panel_y + zoom_canvas_size + 50,
                        imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 1.0))
                    draw_list.add_rect(
                        zoom_panel_x, zoom_panel_y,
                        zoom_panel_x + src_canvas_size + src_padding * 2, zoom_panel_y + zoom_canvas_size + 50,
                        imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 1.0))

                    # Title
                    draw_list.add_text(zoom_panel_x + zoom_padding, zoom_panel_y + 5,
                                      imgui.get_color_u32_rgba(0.9, 0.9, 0.9, 1.0),
                                      "Neck Detail (Zoomed)")

                    # Get neck line info
                    line_start = neck_info['line_start']
                    line_end = neck_info['line_end']
                    piece_idx = neck_info['piece_idx']
                    current_pieces_zoom = obj._manual_cut_data.get('current_pieces', [target_2d])

                    if piece_idx < len(current_pieces_zoom):
                        piece_2d = current_pieces_zoom[piece_idx]

                        # Calculate zoom level to keep line at fixed pixel length (e.g., 120 pixels)
                        target_line_pixels = 120.0
                        actual_line_length = np.linalg.norm(line_end - line_start)
                        if actual_line_length > 1e-10:
                            # Calculate the world space extent that should fit in target_line_pixels
                            world_extent = actual_line_length * (zoom_canvas_size / target_line_pixels)
                        else:
                            world_extent = np.max(piece_2d.max(axis=0) - piece_2d.min(axis=0)) * 0.3

                        # Center on neck midpoint
                        neck_center = (line_start + line_end) / 2

                        # Zoomed canvas position
                        zcanvas_x = zoom_panel_x + zoom_padding
                        zcanvas_y = zoom_panel_y + 25
                        zcanvas_size = zoom_canvas_size

                        # Draw zoom canvas background
                        draw_list.add_rect_filled(
                            zcanvas_x, zcanvas_y,
                            zcanvas_x + zcanvas_size, zcanvas_y + zcanvas_size,
                            imgui.get_color_u32_rgba(0.05, 0.05, 0.05, 1.0))

                        # Clip to zoom canvas
                        draw_list.push_clip_rect(zcanvas_x, zcanvas_y,
                                                 zcanvas_x + zcanvas_size, zcanvas_y + zcanvas_size)

                        # Transform function for zoomed view
                        def to_zoom_screen(p):
                            offset = np.array(p) - neck_center
                            normalized = offset / world_extent + 0.5
                            return (zcanvas_x + normalized[0] * zcanvas_size,
                                    zcanvas_y + (1.0 - normalized[1]) * zcanvas_size)

                        # Draw the piece contour (faded white)
                        if len(piece_2d) >= 3:
                            for i in range(len(piece_2d)):
                                p1 = to_zoom_screen(piece_2d[i])
                                p2 = to_zoom_screen(piece_2d[(i + 1) % len(piece_2d)])
                                draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                                  imgui.get_color_u32_rgba(0.6, 0.6, 0.6, 0.8), 1.5)

                        # Draw cutting line (magenta)
                        zstart = to_zoom_screen(line_start)
                        zend = to_zoom_screen(line_end)
                        draw_list.add_line(zstart[0], zstart[1], zend[0], zend[1],
                                          imgui.get_color_u32_rgba(1.0, 0.0, 1.0, 1.0), 3.0)

                        # Draw endpoints (on contour)
                        draw_list.add_circle_filled(zstart[0], zstart[1], 5,
                                                   imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))
                        draw_list.add_circle_filled(zend[0], zend[1], 5,
                                                   imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0))

                        # Draw neck vertices (cyan)
                        neck_a = neck_info['neck_a']
                        neck_b = neck_info['neck_b']
                        za = to_zoom_screen(neck_a)
                        zb = to_zoom_screen(neck_b)
                        draw_list.add_circle_filled(za[0], za[1], 4,
                                                   imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0))
                        draw_list.add_circle_filled(zb[0], zb[1], 4,
                                                   imgui.get_color_u32_rgba(0.0, 1.0, 1.0, 1.0))

                        # End clip
                        draw_list.pop_clip_rect()

                        # Show info below zoomed canvas
                        info_y = zcanvas_y + zcanvas_size + 5
                        draw_list.add_text(zcanvas_x, info_y,
                                          imgui.get_color_u32_rgba(0.7, 0.7, 0.7, 1.0),
                                          f"Line len: {actual_line_length:.4f}")

            # Get piece info (use selected_sources count for required_pieces)
            current_pieces = obj._manual_cut_data.get('current_pieces', [target_2d])
            selected_sources = obj._manual_cut_data.get('selected_sources', list(range(len(obj._manual_cut_data.get('source_contours', [])))))
            required_pieces = len(selected_sources) if len(selected_sources) > 0 else obj._manual_cut_data.get('required_pieces', 2)
            source_indices_display = obj._manual_cut_data.get('source_indices', [])

            # Show source->target info
            imgui.set_cursor_screen_pos((cursor_pos[0], y0 + canvas_size + padding + 10))
            imgui.separator()
            num_selected = len(selected_sources)
            imgui.text(f"Selected: {num_selected} sources -> 1 target")
            imgui.text(f"Pieces: {len(current_pieces)} / {required_pieces} needed")

            # Show preview mode indicator
            in_preview_mode = obj._manual_cut_data.get('optimization_preview', False)
            if in_preview_mode:
                edit_count = len(obj._manual_cut_data.get('edit_history', []))
                if edit_count > 0:
                    imgui.text_colored(f"[PREVIEW MODE] {edit_count} edit(s) - Undo/Accept/Reset", 1.0, 0.8, 0.2, 1.0)
                else:
                    imgui.text_colored("[PREVIEW MODE] Draw lines to edit, or Accept/Reset", 1.0, 0.8, 0.2, 1.0)

                # Show piece-to-source mapping
                current_pieces_3d_preview = obj._manual_cut_data.get('current_pieces_3d', [])
                source_contours_preview = obj._manual_cut_data.get('source_contours', [])

                if len(current_pieces_3d_preview) > 0 and len(source_contours_preview) > 0:
                    imgui.separator()
                    imgui.text("Piece -> Source mapping:")

                    # Compute centroids for matching
                    piece_centroids = [np.mean(p, axis=0) for p in current_pieces_3d_preview]
                    source_centroids = [np.mean(s, axis=0) for s in source_contours_preview]

                    # Piece colors
                    piece_colors = [
                        (1.0, 1.0, 1.0, 1.0),  # White
                        (0.2, 0.6, 1.0, 1.0),  # Blue
                        (1.0, 0.4, 0.2, 1.0),  # Orange
                        (0.2, 1.0, 0.4, 1.0),  # Green
                        (1.0, 0.8, 0.2, 1.0),  # Yellow
                        (0.8, 0.2, 1.0, 1.0),  # Purple
                    ]

                    # Match each piece to closest source
                    source_labels = obj._manual_cut_data.get('source_labels', list(range(len(source_contours_preview))))
                    used_sources = set()
                    for piece_idx, piece_centroid in enumerate(piece_centroids):
                        best_src = None
                        best_dist = float('inf')
                        for src_idx, src_centroid in enumerate(source_centroids):
                            if src_idx in used_sources:
                                continue
                            dist = np.linalg.norm(piece_centroid - src_centroid)
                            if dist < best_dist:
                                best_dist = dist
                                best_src = src_idx

                        if best_src is not None:
                            used_sources.add(best_src)
                            src_label = source_labels[best_src] if best_src < len(source_labels) else best_src
                            color = piece_colors[piece_idx % len(piece_colors)]
                            imgui.text_colored(f"  P{piece_idx} ({len(current_pieces_3d_preview[piece_idx])} verts) -> S{src_label}", *color)

                # Show optimization scales
                if hasattr(obj, '_bp_viz_data') and len(obj._bp_viz_data) > 0:
                    latest_viz = obj._bp_viz_data[-1]
                    opt_scales = latest_viz.get('scales', [])
                    if len(opt_scales) > 0:
                        imgui.separator()
                        imgui.text("Optimization scales:")
                        for si, scale in enumerate(opt_scales):
                            if isinstance(scale, (list, tuple)) and len(scale) == 2:
                                imgui.text(f"  Source {si}: ({scale[0]:.3f}, {scale[1]:.3f})")
                            else:
                                imgui.text(f"  Source {si}: {scale:.3f}")

                imgui.separator()

            # Show hint for 1-to-1 case
            if num_selected == 1:
                imgui.text_colored("(1:1 mapping - no cutting needed, click Skip)", 0.5, 1.0, 0.5, 1.0)

            # Buttons: Optimize, Optimize All, Skip, Next Cut, OK, Reset, Cancel
            imgui.separator()
            button_width = 80

            # Check if auto-optimize-all mode is active
            auto_optimize_all = getattr(obj, '_auto_optimize_all', False)
            should_auto_optimize = auto_optimize_all and num_selected >= 2

            # Optimize button - run automatic optimization on current pieces
            # Can be used after manual cuts to optimize remaining subdivisions
            # Hide in preview mode (Accept/Reset are shown instead) and in assignment mode
            in_preview_for_button = obj._manual_cut_data.get('optimization_preview', False) if obj._manual_cut_data else False
            in_assignment_mode_for_button = obj._manual_cut_data.get('assignment_mode', False) if obj._manual_cut_data else False
            if num_selected >= 2 and not in_preview_for_button and not in_assignment_mode_for_button:
                # Determine if we should run optimization (button click or auto-optimize mode)
                run_optimization = False
                skip_preview = False  # For Optimize All: skip preview and auto-accept

                if imgui.button("Optimize", button_width, 30):
                    run_optimization = True
                imgui.same_line()

                # Optimize All button - optimize and accept without preview
                if imgui.button("Opt All", button_width, 30):
                    run_optimization = True
                    skip_preview = True
                    obj._auto_optimize_all = True  # Set flag for subsequent windows
                    print(f"[OptAll] Starting Optimize All mode...")
                imgui.same_line()

                # Auto-trigger optimization if in auto-optimize-all mode
                if should_auto_optimize and not run_optimization:
                    run_optimization = True
                    skip_preview = True
                    print(f"[OptAll] Auto-triggering optimization...")

                if run_optimization:
                    # Get source data
                    source_contours = obj._manual_cut_data.get('source_contours', [])
                    source_bps = obj._manual_cut_data.get('source_bps', [])
                    target_bp = obj._manual_cut_data.get('target_bp')
                    stream_indices = obj._manual_cut_data.get('stream_indices', list(range(len(source_contours))))
                    target_level = obj._manual_cut_data.get('target_level')
                    source_level = obj._manual_cut_data.get('source_level')
                    current_pieces_3d = obj._manual_cut_data.get('current_pieces_3d', [obj._manual_cut_data.get('target_contour')])

                    print(f"Running optimization on {len(current_pieces_3d)} current pieces -> {required_pieces} needed...")

                    # Filter by selected sources
                    selected_source_contours = [source_contours[i] for i in selected_sources if i < len(source_contours)]
                    selected_source_bps = [source_bps[i] for i in selected_sources if i < len(source_bps)]
                    selected_stream_indices = [stream_indices[i] for i in selected_sources if i < len(stream_indices)]

                    if len(selected_source_contours) >= 2:
                        # Get matched pairs for combining with optimized pieces
                        matched_pairs = obj._manual_cut_data.get('matched_pairs', [])

                        # Optimize remaining (unmatched) pieces
                        final_pieces = obj._optimize_remaining_pieces(
                            current_pieces_3d,
                            selected_source_contours, selected_source_bps,
                            selected_stream_indices, target_bp,
                            target_level, source_level,
                            matched_pairs=matched_pairs
                        )

                        if final_pieces is not None and len(final_pieces) > 0:
                            # Convert 3D pieces to 2D for display
                            target_mean = target_bp['mean']
                            # Negate basis vectors to rotate 180° - match initial cutting window
                            target_x = -target_bp['basis_x']
                            target_y = -target_bp['basis_y']

                            print(f"[Optimize DEBUG] Converting {len(final_pieces)} pieces to 2D")
                            optimized_2d = []
                            for pi, piece_3d in enumerate(final_pieces):
                                piece_2d = np.array([
                                    [np.dot(v - target_mean, target_x), np.dot(v - target_mean, target_y)]
                                    for v in piece_3d
                                ])
                                optimized_2d.append(piece_2d)
                                print(f"  Piece {pi}: {len(piece_3d)} verts 3D -> {len(piece_2d)} verts 2D")
                                if len(piece_2d) > 0:
                                    centroid_2d = np.mean(piece_2d, axis=0)
                                    print(f"    2D centroid: ({centroid_2d[0]:.4f}, {centroid_2d[1]:.4f})")

                            # Update current pieces with optimized result
                            obj._manual_cut_data['current_pieces'] = optimized_2d
                            obj._manual_cut_data['current_pieces_3d'] = list(final_pieces)
                            obj._manual_cut_data['pre_optimization_pieces'] = current_pieces.copy()
                            obj._manual_cut_data['pre_optimization_pieces_3d'] = [p.copy() for p in current_pieces_3d]
                            obj._manual_cut_data['cut_lines'] = []
                            obj._manual_cut_data['edit_history'] = []
                            obj._manual_cut_line = None

                            # Store transformed source contours for display on cutting panel
                            # Also update target_2d, source_2d_list, and current_pieces to match the coordinate system
                            if hasattr(obj, '_bp_viz_data') and len(obj._bp_viz_data) > 0:
                                latest_viz = obj._bp_viz_data[-1]
                                print(f"[Optimize DEBUG] _bp_viz_data has {len(obj._bp_viz_data)} entries, latest keys: {list(latest_viz.keys())}")
                                transformed_srcs = latest_viz.get('final_transformed', [])
                                obj._manual_cut_data['transformed_sources_2d'] = transformed_srcs

                                # Update target_2d to match optimization's coordinate system
                                opt_target_2d = latest_viz.get('target_2d', None)
                                old_target_2d = obj._manual_cut_data.get('target_2d', None)
                                if opt_target_2d is not None:
                                    # Compare old vs new target_2d
                                    if old_target_2d is not None and hasattr(old_target_2d, '__len__') and len(old_target_2d) > 0:
                                        old_centroid = np.mean(old_target_2d, axis=0)
                                        new_centroid = np.mean(opt_target_2d, axis=0)
                                        print(f"[Optimize DEBUG] target_2d comparison:")
                                        print(f"  old: {len(old_target_2d)} verts, centroid=({old_centroid[0]:.4f}, {old_centroid[1]:.4f}), first=({old_target_2d[0][0]:.4f}, {old_target_2d[0][1]:.4f})")
                                        print(f"  new: {len(opt_target_2d)} verts, centroid=({new_centroid[0]:.4f}, {new_centroid[1]:.4f}), first=({opt_target_2d[0][0]:.4f}, {opt_target_2d[0][1]:.4f})")
                                    else:
                                        print(f"[Optimize DEBUG] old_target_2d is None or empty: {type(old_target_2d)}, hasattr len: {hasattr(old_target_2d, '__len__')}")
                                    obj._manual_cut_data['target_2d'] = opt_target_2d
                                    print(f"[Optimize DEBUG] Updated target_2d: {len(opt_target_2d)} verts")
                                else:
                                    print(f"[Optimize DEBUG] WARNING: opt_target_2d is None!")

                                # Update source_2d_list to match (use source_2d_shapes + translations)
                                source_shapes = latest_viz.get('source_2d_shapes', [])
                                init_trans = latest_viz.get('initial_translations', [])
                                print(f"[Optimize DEBUG] source_shapes: {len(source_shapes)}, init_trans: {len(init_trans)}")
                                if len(source_shapes) > 0 and len(init_trans) == len(source_shapes):
                                    new_source_2d_list = []
                                    for i, (shape, trans) in enumerate(zip(source_shapes, init_trans)):
                                        # Reconstruct original source position (before any transform)
                                        src_2d = np.array(shape) + np.array(trans)
                                        new_source_2d_list.append(src_2d)
                                    obj._manual_cut_data['source_2d_list'] = new_source_2d_list
                                    print(f"[Optimize DEBUG] Updated source_2d_list: {len(new_source_2d_list)} sources")

                                # Use pieces_2d directly from optimization (same coordinate system)
                                opt_pieces_2d = latest_viz.get('pieces_2d', [])
                                if len(opt_pieces_2d) > 0:
                                    obj._manual_cut_data['current_pieces'] = opt_pieces_2d
                                    print(f"[Optimize DEBUG] Using pieces_2d from optimization ({len(opt_pieces_2d)} pieces)")
                                    for pi, pc in enumerate(opt_pieces_2d):
                                        pc_centroid = np.mean(pc, axis=0)
                                        print(f"  Piece {pi}: {len(pc)} verts, centroid ({pc_centroid[0]:.4f}, {pc_centroid[1]:.4f})")
                                else:
                                    # Fallback to projecting from 3D (old behavior)
                                    obj._manual_cut_data['current_pieces'] = optimized_2d

                                print(f"[Optimize DEBUG] Transformed sources: {len(transformed_srcs)}")
                                for si, src in enumerate(transformed_srcs):
                                    if src is not None and len(src) > 0:
                                        src_centroid = np.mean(src, axis=0)
                                        print(f"  Source {si}: {len(src)} verts, centroid ({src_centroid[0]:.4f}, {src_centroid[1]:.4f})")

                            if skip_preview:
                                # Optimize All: trigger auto-accept flag
                                obj._manual_cut_data['optimization_preview'] = True
                                obj._manual_cut_data['auto_accept'] = True
                                print(f"[OptAll] Optimization produced {len(final_pieces)} pieces - auto-accepting...")
                            else:
                                # Normal: enter preview mode
                                obj._manual_cut_data['optimization_preview'] = True
                                print(f"[Preview] Optimization produced {len(final_pieces)} pieces - enter preview mode")
                                print(f"[Preview] You can now Accept, Reset, or edit with cutting lines")
                        else:
                            print(f"[Error] Optimization failed to produce pieces")
                    else:
                        print(f"Need at least 2 source contours for optimization")

            # ===== Preview Mode Buttons: Accept / Reset =====
            in_preview_mode = obj._manual_cut_data.get('optimization_preview', False) if obj._manual_cut_data else False
            if in_preview_mode:
                # Check for auto-accept (from Optimize All)
                auto_accept = obj._manual_cut_data.get('auto_accept', False)

                # Accept button - finalize the optimization result
                # Also triggered automatically in Optimize All mode
                # Don't show button if auto-accepting (cleaner UI)
                if auto_accept:
                    accept_clicked = True
                else:
                    accept_clicked = imgui.button("Accept", button_width, 30)
                if accept_clicked:
                    if auto_accept:
                        obj._manual_cut_data['auto_accept'] = False  # Clear flag
                    print(f"[Preview] Accepting optimization result...")

                    # Get the current optimized pieces
                    current_pieces_3d = obj._manual_cut_data.get('current_pieces_3d', [])
                    # final_pieces will be set later to exclude matched pieces

                    # Get context data
                    source_contours = obj._manual_cut_data.get('source_contours', [])
                    source_bps = obj._manual_cut_data.get('source_bps', [])
                    target_bp = obj._manual_cut_data.get('target_bp')
                    stream_indices = obj._manual_cut_data.get('stream_indices', list(range(len(source_contours))))
                    target_level = obj._manual_cut_data.get('target_level')
                    source_level = obj._manual_cut_data.get('source_level')
                    matched_pairs = obj._manual_cut_data.get('matched_pairs', [])
                    parent_finalized_pieces = obj._manual_cut_data.get('parent_finalized_pieces', {})
                    original_source_indices = obj._manual_cut_data.get('original_source_indices', None)
                    selected_sources = obj._manual_cut_data.get('selected_sources', list(range(len(source_contours))))

                    # Exit preview mode
                    obj._manual_cut_data['optimization_preview'] = False

                    # Now run the finalization code
                    # Use sub-window context if original_source_indices is set (indicating we're in a sub-cut)
                    # Note: parent_finalized_pieces may be empty if no 1:1 matches from parent
                    if original_source_indices:
                        # Sub-window context: need to combine with parent's 1:1 pieces
                        source_contours_full = obj._manual_cut_data.get('source_contours', [])
                        # Compute total_pieces from max stream index (NOT just count)
                        # Stream indices can be larger than count-1, e.g., streams [3,1,2] with count 3
                        # needs array size 4 to fit index 3
                        parent_context = obj._manual_cut_data.get('parent_context', None)
                        all_stream_indices_for_size = parent_context.get('stream_indices', []) if parent_context else []
                        max_parent_idx = max(parent_finalized_pieces.keys()) if parent_finalized_pieces else -1
                        max_current_idx = max(original_source_indices) if original_source_indices else -1
                        max_all_streams = max(all_stream_indices_for_size) if all_stream_indices_for_size else -1
                        total_pieces = max(max_parent_idx, max_current_idx, max_all_streams) + 1
                        all_pieces = [None] * total_pieces
                        print(f"[Accept] Using total_pieces={total_pieces} (from max stream index)")

                        # Fill in parent's finalized pieces (1:1 assignments from parent)
                        print(f"[Accept DEBUG] parent_finalized_pieces has {len(parent_finalized_pieces)} entries:")
                        for orig_src_idx, piece in parent_finalized_pieces.items():
                            if orig_src_idx < total_pieces:
                                all_pieces[orig_src_idx] = piece
                            piece_id = id(piece) if piece is not None else 0
                            piece_verts = len(piece) if piece is not None else 0
                            piece_centroid = np.mean(piece, axis=0) if piece is not None and len(piece) > 0 else [0,0,0]
                            print(f"[Accept DEBUG]   parent_finalized[{orig_src_idx}]: {piece_verts} verts, id={piece_id}, centroid={piece_centroid}")

                        # Fill in pre-matched pieces (mapped to original indices)
                        matched_piece_indices = set()
                        for piece_idx, local_src_idx in matched_pairs:
                            if piece_idx < len(current_pieces_3d) and local_src_idx < len(original_source_indices):
                                orig_src_idx = original_source_indices[local_src_idx]
                                if orig_src_idx < total_pieces:
                                    all_pieces[orig_src_idx] = current_pieces_3d[piece_idx]
                                    matched_piece_indices.add(piece_idx)

                        # Build list of unmatched pieces (excluding already-matched ones)
                        unmatched_pieces = [current_pieces_3d[i] for i in range(len(current_pieces_3d))
                                           if i not in matched_piece_indices]

                        # Fill in optimized pieces for remaining sources (mapped to original indices)
                        opt_idx = 0
                        for local_src_idx in selected_sources:
                            if local_src_idx < len(original_source_indices):
                                orig_src_idx = original_source_indices[local_src_idx]
                                if orig_src_idx < total_pieces and all_pieces[orig_src_idx] is None:
                                    if opt_idx < len(unmatched_pieces):
                                        all_pieces[orig_src_idx] = unmatched_pieces[opt_idx]
                                        opt_idx += 1

                        # Get original stream indices from parent context
                        # IMPORTANT: Use the actual stream indices, not just [0, 1, ..., M-1]
                        parent_context = obj._manual_cut_data.get('parent_context', None)
                        if parent_context and 'stream_indices' in parent_context:
                            all_stream_indices = parent_context['stream_indices']
                        else:
                            # Fallback to initial stream_indices if parent_context not available
                            all_stream_indices = obj._manual_cut_data.get('stream_indices', list(range(total_pieces)))
                    else:
                        # Normal context (not a sub-window)
                        source_contours_full = obj._manual_cut_data.get('source_contours', [])
                        all_pieces = [None] * len(source_contours_full)

                        # Fill in pre-matched pieces
                        matched_piece_indices = set()
                        for piece_idx, src_idx in matched_pairs:
                            if piece_idx < len(current_pieces_3d) and src_idx < len(all_pieces):
                                all_pieces[src_idx] = current_pieces_3d[piece_idx]
                                matched_piece_indices.add(piece_idx)

                        # Build list of unmatched pieces (excluding already-matched ones)
                        unmatched_pieces = [current_pieces_3d[i] for i in range(len(current_pieces_3d))
                                           if i not in matched_piece_indices]

                        # Fill in optimized pieces for remaining sources
                        opt_idx = 0
                        for src_idx in selected_sources:
                            if src_idx < len(all_pieces) and all_pieces[src_idx] is None and opt_idx < len(unmatched_pieces):
                                all_pieces[src_idx] = unmatched_pieces[opt_idx]
                                opt_idx += 1

                        all_stream_indices = obj._manual_cut_data.get('stream_indices', stream_indices)

                    # Check if we have all pieces
                    # Note: all_pieces may have extra None slots for unused indices
                    # (e.g., streams [3,1,2] need array size 4 but only 3 pieces)
                    filled_count = sum(1 for p in all_pieces if p is not None)
                    # Get actual required count from parent context's stream indices
                    required_pieces = len(all_stream_indices_for_size) if original_source_indices and all_stream_indices_for_size else len(all_pieces)
                    print(f"[Accept] Filled {filled_count}/{required_pieces} pieces (array size {len(all_pieces)}): matched={len(matched_pairs)}, unmatched={len(unmatched_pieces)}")
                    print(f"[Accept DEBUG] Final all_pieces state:")
                    for i, p in enumerate(all_pieces):
                        if p is not None:
                            p_id = id(p)
                            p_centroid = np.mean(p, axis=0) if len(p) > 0 else [0,0,0]
                            print(f"  all_pieces[{i}]: {len(p)} verts, id={p_id}, centroid={p_centroid}")
                        else:
                            print(f"  all_pieces[{i}]: None")

                    # Update parent_finalized_pieces with THIS sub-cut's results
                    # Use all_pieces directly since it already has correct mappings
                    updated_finalized = dict(parent_finalized_pieces) if parent_finalized_pieces else {}
                    if original_source_indices:
                        for local_src_idx in range(len(original_source_indices)):
                            orig_src_idx = original_source_indices[local_src_idx]
                            if orig_src_idx < len(all_pieces) and all_pieces[orig_src_idx] is not None:
                                updated_finalized[orig_src_idx] = all_pieces[orig_src_idx]

                    # Level-by-level processing: ALWAYS check pending sub-cuts first
                    # This must happen regardless of whether all pieces are filled
                    # IMPORTANT: Use subcut_level (where we ARE), not current_subcut_level
                    subcut_level = obj._manual_cut_data.get('subcut_level', 0)
                    pending_by_level = obj._manual_cut_data.get('pending_subcuts_by_level', {})

                    print(f"[Accept] Subcut level: {subcut_level}")
                    print(f"[Accept] Pending by level: {[(lvl, len(lst)) for lvl, lst in pending_by_level.items()]}")

                    # Find lowest level with pending sub-cuts (breadth-first)
                    next_pending = None
                    for level in sorted(pending_by_level.keys()):
                        if len(pending_by_level[level]) > 0:
                            next_pending = (level, pending_by_level[level])
                            break

                    if next_pending is not None:
                        # Open next pending sub-cut
                        level, pending_list = next_pending
                        print(f"[Accept] Opening pending sub-cut at level {level} ({len(pending_list)} remaining)")
                        next_subcut = pending_list[0]
                        obj._manual_cut_data['parent_finalized_pieces'] = updated_finalized
                        obj._open_subcut_for_piece(next_subcut)
                    else:
                        # No more pending sub-cuts - check if ready to finalize
                        # Compare against required_pieces (actual source count), not array size
                        if filled_count >= required_pieces:
                            print(f"[Accept] Finalizing: {required_pieces} pieces ({len(matched_pairs)} pre-matched, {len(unmatched_pieces)} optimized)")

                            # All levels done - store final result and call cut_streams
                            if not hasattr(obj, '_manual_cut_results') or obj._manual_cut_results is None:
                                obj._manual_cut_results = {}

                            target_i = obj._manual_cut_data.get('target_i', 0)
                            result_key = (target_level, target_i)
                            print(f"[Accept] Storing result: target_level={target_level}, target_i={target_i}")
                            print(f"[Accept] all_pieces has {len(all_pieces)} entries, all_stream_indices={all_stream_indices}")
                            obj._manual_cut_results[result_key] = {
                                'cut_contours': all_pieces,
                                'source_indices': all_stream_indices,
                                'is_1to1': False,
                            }
                            print(f"[Accept] All levels complete, stored result with key {result_key}")

                            # Clear stale data before continuing
                            obj._manual_cut_data = None
                            obj._manual_cut_original_state = None
                            obj._manual_cut_pending = False
                            obj.cut_streams(cut_method='bp', muscle_name=muscle_name)
                            # Smoothening and alignment are now handled inside cut_streams
                            if not obj._manual_cut_pending and obj._manual_cut_data is None:
                                # Clear auto-optimize-all flag when all cutting is done
                                if hasattr(obj, '_auto_optimize_all'):
                                    obj._auto_optimize_all = False
                                if name in self._manual_cut_mouse:
                                    del self._manual_cut_mouse[name]
                                imgui.end()
                                continue
                        else:
                            # No pending sub-cuts but pieces incomplete - this shouldn't happen
                            print(f"[Accept] WARNING: No pending sub-cuts but only {filled_count}/{required_pieces} pieces filled")

                # Skip remaining preview buttons if auto-accepting (cleaner UI)
                if not auto_accept:
                    imgui.same_line()

                    # Reset button - go back to pre-optimization state
                    if imgui.button("Reset", button_width, 30):
                        print(f"[Preview] Resetting to pre-optimization state...")
                        pre_pieces = obj._manual_cut_data.get('pre_optimization_pieces', [])
                        pre_pieces_3d = obj._manual_cut_data.get('pre_optimization_pieces_3d', [])
                        if pre_pieces and pre_pieces_3d:
                            obj._manual_cut_data['current_pieces'] = pre_pieces
                            obj._manual_cut_data['current_pieces_3d'] = pre_pieces_3d
                        obj._manual_cut_data['optimization_preview'] = False
                        obj._manual_cut_data['cut_lines'] = []
                        obj._manual_cut_data['edit_history'] = []
                        obj._manual_cut_line = None
                        # Clear optimization display data
                        obj._manual_cut_data['transformed_sources_2d'] = None
                        # Clear _bp_viz_data to remove optimization results
                        if hasattr(obj, '_bp_viz_data'):
                            obj._bp_viz_data = []
                        print(f"[Preview] Reset complete - you can draw new cuts or optimize again")

                    imgui.same_line()

                    # Next Cut button - apply current cutting line to edit optimized pieces
                    has_cut_line = obj._manual_cut_line is not None
                    if has_cut_line:
                        if imgui.button("Cut", button_width, 30):
                            print(f"[Preview] Applying cut to edit optimized pieces...")

                            # Save current state for undo before applying cut
                            edit_history = obj._manual_cut_data.get('edit_history', [])
                            edit_history.append({
                                'pieces': [p.copy() for p in obj._manual_cut_data.get('current_pieces', [])],
                                'pieces_3d': [p.copy() for p in obj._manual_cut_data.get('current_pieces_3d', [])],
                            })
                            obj._manual_cut_data['edit_history'] = edit_history

                            # Apply the iterative cut to further subdivide pieces
                            success = obj._apply_iterative_cut()
                            if success:
                                print(f"[Preview] Cut applied - now have {len(obj._manual_cut_data.get('current_pieces', []))} pieces (edit #{len(edit_history)})")
                                obj._manual_cut_line = None  # Clear line after applying
                            else:
                                # Remove the history entry since cut failed
                                edit_history.pop()
                                print(f"[Preview] Cut failed - try drawing a different line")
                        imgui.same_line()

                    # Undo button - revert to previous state
                    edit_history = obj._manual_cut_data.get('edit_history', [])
                    if len(edit_history) > 0:
                        if imgui.button("Undo", button_width, 30):
                            prev_state = edit_history.pop()
                            obj._manual_cut_data['current_pieces'] = prev_state['pieces']
                            obj._manual_cut_data['current_pieces_3d'] = prev_state['pieces_3d']
                            obj._manual_cut_data['edit_history'] = edit_history
                            obj._manual_cut_line = None
                            print(f"[Preview] Undid last edit - now have {len(prev_state['pieces'])} pieces")
                        imgui.same_line()

            # Skip button - for 1-to-1 case (only 1 source selected, no cutting needed)
            # Also auto-triggered in Optimize All mode
            if num_selected == 1:
                skip_clicked = imgui.button("Skip (1:1)", button_width, 30)
                # Auto-skip in Optimize All mode
                if auto_optimize_all and not skip_clicked:
                    skip_clicked = True
                    print(f"[OptAll] Auto-skipping 1:1 case...")
                if skip_clicked:
                    print(f"Skipping cut for target (1:1 mapping with source {selected_sources[0]})")
                    # Store the 1:1 result - the single selected source maps directly to target
                    selected_src_idx = selected_sources[0]
                    target_i = obj._manual_cut_data.get('target_i', 0)
                    target_level = obj._manual_cut_data.get('target_level', 0)
                    source_indices = obj._manual_cut_data.get('source_indices', [])

                    # Store result in _manual_cut_results (persistent storage)
                    # Use the actual source index from source_indices, not the local index
                    actual_source_idx = source_indices[selected_src_idx] if selected_src_idx < len(source_indices) else selected_src_idx

                    if not hasattr(obj, '_manual_cut_results') or obj._manual_cut_results is None:
                        obj._manual_cut_results = {}

                    result_key = (target_level, target_i)
                    obj._manual_cut_results[result_key] = {
                        'cut_contours': [obj._manual_cut_data['target_contour']],  # Single piece = target
                        'source_indices': [actual_source_idx],  # Maps to this source
                        'is_1to1': True,  # Flag for 1:1 mapping
                    }
                    print(f"Stored 1:1 result with key {result_key}, source {actual_source_idx}")

                    # Continue with cut_streams - it will find the next cut point or finish
                    obj._manual_cut_data = None
                    obj._manual_cut_line = None
                    obj._manual_cut_pending = False
                    obj.cut_streams(cut_method='bp', muscle_name=muscle_name)

                    # Smoothening and alignment are now handled inside cut_streams
                    if not obj._manual_cut_pending and obj._manual_cut_data is None:
                        # Clear auto-optimize-all flag when all cutting is done
                        if hasattr(obj, '_auto_optimize_all'):
                            obj._auto_optimize_all = False
                        if name in self._manual_cut_mouse:
                            del self._manual_cut_mouse[name]
                        imgui.end()
                        continue
                    # cut_streams returned early for another manual cut - don't close window
                imgui.same_line()

            # Check if we're in assignment mode (after cutting)
            has_valid_line = obj._manual_cut_line is not None
            in_assignment_mode = obj._manual_cut_data.get('assignment_mode', False)
            piece_assignments = obj._manual_cut_data.get('piece_assignments', {})
            in_preview_mode_check = obj._manual_cut_data.get('optimization_preview', False)

            # Skip normal cutting/assignment UI if in preview mode (Accept/Reset/Cut shown above)
            if in_preview_mode_check:
                # In preview mode, show Save&Close and Cancel buttons
                if imgui.button("Save && Close", button_width, 30):
                    obj._save_and_close_manual_cut(muscle_name=muscle_name)
                    if hasattr(obj, '_auto_optimize_all'):
                        obj._auto_optimize_all = False
                    if name in self._manual_cut_mouse:
                        del self._manual_cut_mouse[name]
                    imgui.end()
                    continue
                imgui.same_line()
                if imgui.button("Cancel", button_width, 30):
                    obj._cancel_manual_cut()
                    if name in self._manual_cut_mouse:
                        del self._manual_cut_mouse[name]
                imgui.end()
                continue

            if in_assignment_mode:
                # === ASSIGNMENT MODE UI ===
                imgui.text("Assign sources to pieces:")
                imgui.separator()

                # Piece colors (same as visualization)
                piece_colors = [
                    (1.0, 1.0, 1.0),  # White
                    (0.2, 0.6, 1.0),  # Blue
                    (1.0, 0.4, 0.2),  # Orange
                    (0.2, 1.0, 0.4),  # Green
                    (1.0, 0.8, 0.2),  # Yellow
                    (0.8, 0.2, 1.0),  # Purple
                ]

                # Get source labels (original indices for display)
                source_labels = obj._manual_cut_data.get('source_labels', list(range(len(source_contours_3d))))

                # Show each piece with source checkboxes
                for piece_idx in range(len(current_pieces)):
                    assigned = piece_assignments.get(piece_idx, [])
                    # Draw color indicator
                    color = piece_colors[piece_idx % len(piece_colors)]
                    imgui.push_style_color(imgui.COLOR_BUTTON, *color, 1.0)
                    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *color, 1.0)
                    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *color, 1.0)
                    imgui.button(f"P{piece_idx}##color", 30, 20)
                    imgui.pop_style_color(3)
                    imgui.same_line()
                    imgui.text("->")
                    imgui.same_line()

                    # Checkboxes for each source (use original labels for display)
                    for src_idx in range(len(source_contours_3d)):
                        is_assigned = src_idx in assigned
                        # Display original source index from source_labels
                        src_label = source_labels[src_idx] if src_idx < len(source_labels) else src_idx
                        changed, new_val = imgui.checkbox(f"S{src_label}##p{piece_idx}", is_assigned)
                        if changed:
                            if new_val:
                                # Add source to this piece, remove from others
                                for p in piece_assignments:
                                    if src_idx in piece_assignments[p]:
                                        piece_assignments[p].remove(src_idx)
                                if piece_idx not in piece_assignments:
                                    piece_assignments[piece_idx] = []
                                piece_assignments[piece_idx].append(src_idx)
                            else:
                                if src_idx in piece_assignments.get(piece_idx, []):
                                    piece_assignments[piece_idx].remove(src_idx)
                            obj._manual_cut_data['piece_assignments'] = piece_assignments
                        imgui.same_line()
                    imgui.new_line()

                imgui.separator()

                # Confirm button - processes assignments
                if imgui.button("Confirm", button_width, 30):
                    # Process assignments: 1:1 finalized, 2:1 opens sub-window
                    obj._process_piece_assignments()

                    # Check what to do next
                    # IMPORTANT: Use subcut_level (where we ARE) not current_subcut_level (batch tracking)
                    # _process_piece_assignments queues to subcut_level + 1
                    subcut_level = obj._manual_cut_data.get('subcut_level', 0)
                    pending_by_level = obj._manual_cut_data.get('pending_subcuts_by_level', {})
                    newly_queued = pending_by_level.get(subcut_level + 1, [])

                    print(f"[Confirm] Subcut level: {subcut_level}")
                    print(f"[Confirm] Pending by level: {[(lvl, len(lst)) for lvl, lst in pending_by_level.items()]}")

                    if len(newly_queued) > 0:
                        # Open sub-window for first pending N:1 case from THIS window
                        print(f"[Confirm] {len(newly_queued)} sub-cuts at level {subcut_level + 1}, opening next...")
                        obj._open_subcut_for_piece(newly_queued[0])
                    else:
                        # Check if there are other pending sub-cuts at any level
                        # Find lowest level with pending sub-cuts (breadth-first)
                        next_pending = None
                        for level in sorted(pending_by_level.keys()):
                            if len(pending_by_level[level]) > 0:
                                next_pending = (level, pending_by_level[level])
                                break

                        if next_pending is not None:
                            # Open next pending sub-cut
                            level, pending_list = next_pending
                            print(f"[Confirm] Opening pending sub-cut at level {level} ({len(pending_list)} remaining)")
                            obj._open_subcut_for_piece(pending_list[0])
                        else:
                            # All done - finalize
                            # Extract values BEFORE finalize in case it modifies _manual_cut_data
                            target_level = obj._manual_cut_data.get('target_level', 0)
                            target_i = obj._manual_cut_data.get('target_i', 0)

                            # _finalize_manual_cuts() stores result with correct stream_indices from parent_context
                            cut_result, _ = obj._finalize_manual_cuts()
                            if cut_result is not None:
                                # NOTE: _finalize_manual_cuts() already stored the result with correct stream_indices
                                # No need to overwrite here - just verify it exists
                                result_key = (target_level, target_i)
                                if hasattr(obj, '_manual_cut_results') and result_key in obj._manual_cut_results:
                                    print(f"[Confirm] Result already stored with key {result_key}")
                                else:
                                    print(f"[Confirm] WARNING: Result not found for key {result_key}")

                                # Clear _manual_cut_data to prevent stale data issues
                                obj._manual_cut_data = None
                                obj._manual_cut_original_state = None
                                obj._manual_cut_pending = False
                                obj.cut_streams(cut_method='bp', muscle_name=muscle_name)
                                # Smoothening and alignment are now handled inside cut_streams
                                if not obj._manual_cut_pending and obj._manual_cut_data is None:
                                    # Clear auto-optimize-all flag when all cutting is done
                                    if hasattr(obj, '_auto_optimize_all'):
                                        obj._auto_optimize_all = False
                                    if name in self._manual_cut_mouse:
                                        del self._manual_cut_mouse[name]
                                    imgui.end()
                                    continue
                                # cut_streams returned early for another manual cut - don't close window

                # NOTE: No "Back" button - only one cut allowed per window
                # User must use "Reset" to start over or "Done" to proceed
            else:
                # === CUTTING MODE UI ===
                # Cut button - applies cut and enters assignment mode
                cut_enabled = has_valid_line
                if not cut_enabled:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

                if imgui.button("Cut", button_width, 30):
                    if cut_enabled:
                        # Apply the cut
                        success = obj._apply_iterative_cut()
                        if success:
                            obj._manual_cut_line = None

                            # Enter assignment mode
                            obj._manual_cut_data['assignment_mode'] = True

                            # Initialize assignments by distance
                            obj._init_piece_assignments_by_distance()

                            print(f"Cut applied. Now assign sources to pieces.")
                        else:
                            print("Failed to apply cut - line must cross a piece twice")
                    else:
                        print("Draw a cutting line first")

                if not cut_enabled:
                    imgui.pop_style_var()

            # Hide Reset/Cancel when auto-optimizing (cleaner UI)
            if not should_auto_optimize:
                imgui.same_line()
                if imgui.button("Reset", button_width, 30):
                    # Reset to original state (before any cuts or sub-windows)
                    obj._reset_to_original_state()
                    # Also clear auto-optimize-all flag
                    if hasattr(obj, '_auto_optimize_all'):
                        obj._auto_optimize_all = False
                    if name in self._manual_cut_mouse:
                        self._manual_cut_mouse[name]['dragging'] = False
                        self._manual_cut_mouse[name]['zoom'] = 1.0
                        self._manual_cut_mouse[name]['pan'] = [0.0, 0.0]

                imgui.same_line()
                if imgui.button("Save && Close", button_width, 30):
                    obj._save_and_close_manual_cut(muscle_name=muscle_name)
                    if hasattr(obj, '_auto_optimize_all'):
                        obj._auto_optimize_all = False
                    if name in self._manual_cut_mouse:
                        del self._manual_cut_mouse[name]
                    imgui.end()
                    continue

                imgui.same_line()
                if imgui.button("Cancel", button_width, 30):
                    obj._cancel_manual_cut()
                    # Also clear auto-optimize-all flag
                    if hasattr(obj, '_auto_optimize_all'):
                        obj._auto_optimize_all = False
                    if name in self._manual_cut_mouse:
                        del self._manual_cut_mouse[name]

            imgui.end()

    def _render_level_select_windows(self):
        """Render level selection windows for manual level selection after Contour Select."""
        for name, obj in self.zygote_muscle_meshes.items():
            if not hasattr(obj, '_level_select_window_open') or not obj._level_select_window_open:
                continue

            if not hasattr(obj, '_level_select_checkboxes') or obj._level_select_checkboxes is None:
                continue

            if not hasattr(obj, '_level_select_original') or obj._level_select_original is None:
                continue

            # Get data
            orig = obj._level_select_original
            checkboxes = obj._level_select_checkboxes
            stream_groups = orig['stream_groups']
            max_stream_count = len(checkboxes)

            # Window setup
            imgui.set_next_window_size(500, 600, imgui.FIRST_USE_EVER)
            expanded, opened = imgui.begin(f"Level Select: {name}", True)

            if not opened:
                # User closed window - cancel selection
                obj._level_select_window_open = False
                obj._level_select_checkboxes = None
                obj._level_select_original = None
                imgui.end()
                continue

            imgui.text(f"Streams: {max_stream_count}")
            imgui.same_line()
            if imgui.button("Rebuild Groups"):
                obj.rebuild_stream_groups()
                # Update the original reference
                orig['stream_groups'] = obj.stream_groups
            imgui.text("Check/uncheck levels. Linked levels toggle together.")
            imgui.separator()

            # Create scrollable region for checkboxes
            imgui.begin_child("LevelCheckboxes", 0, -50, border=True)

            # Track if visualization needs update
            vis_changed = False

            # Find max levels across all streams
            max_levels = max(len(checkboxes[s]) for s in range(max_stream_count))

            # Column width for each stream
            col_width = 80

            # Compute display order so that streams from same merged contour are adjacent
            # Find levels with linked groups and use that structure for ordering
            stream_display_order = list(range(max_stream_count))

            # Debug: print stream_groups to see linking (only once per window open)
            if not hasattr(obj, '_level_select_debug_printed') or not obj._level_select_debug_printed:
                print(f"[DEBUG] Level Select - stream_groups has {len(stream_groups)} levels")
                for lvl_i, groups in enumerate(stream_groups):
                    linked = [g for g in groups if len(g) > 1]
                    if linked:
                        print(f"  Level {lvl_i}: linked groups = {linked}")
                obj._level_select_debug_printed = True

            # Find the most merged level (fewest unique contours = most linking)
            best_level = 0
            max_linked = 0
            for lvl_i, groups in enumerate(stream_groups):
                linked_streams = sum(len(g) for g in groups if len(g) > 1)
                if linked_streams > max_linked:
                    max_linked = linked_streams
                    best_level = lvl_i

            if max_linked > 0 and best_level < len(stream_groups):
                # Use the linked structure at this level to order streams
                # Streams in same group should be adjacent
                groups_at_best = stream_groups[best_level]
                stream_display_order = []
                for group in groups_at_best:
                    stream_display_order.extend(sorted(group))
                # Add any missing streams
                for s in range(max_stream_count):
                    if s not in stream_display_order:
                        stream_display_order.append(s)
                print(f"[DEBUG] Using level {best_level} for ordering, display_order = {stream_display_order}")
            elif hasattr(obj, 'stream_contours') and obj.stream_contours is not None and len(obj.stream_contours) > 0:
                # Fall back to spatial ordering at first level
                try:
                    centroids = []
                    for stream_i in range(max_stream_count):
                        if stream_i < len(obj.stream_contours) and len(obj.stream_contours[stream_i]) > 0:
                            first_contour = obj.stream_contours[stream_i][0]
                            if len(first_contour) > 0:
                                centroid = np.mean(first_contour, axis=0)
                                centroids.append((stream_i, centroid))
                            else:
                                centroids.append((stream_i, np.zeros(3)))
                        else:
                            centroids.append((stream_i, np.zeros(3)))

                    if len(centroids) > 1:
                        positions = np.array([c[1] for c in centroids])
                        mean_pos = np.mean(positions, axis=0)
                        centered = positions - mean_pos

                        if centered.shape[0] >= 2:
                            cov = np.cov(centered.T)
                            if cov.shape == (3, 3):
                                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                                principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
                                projections = [(s, np.dot(c - mean_pos, principal_axis)) for s, c in centroids]
                                projections.sort(key=lambda x: x[1])
                                stream_display_order = [p[0] for p in projections]
                    print(f"[DEBUG] Using spatial ordering at first level, display_order = {stream_display_order}")
                except Exception as e:
                    print(f"[DEBUG] Exception in ordering: {e}")

            # Header row with stream labels (in spatial order)
            imgui.text("Level")
            for col_i, stream_i in enumerate(stream_display_order):
                imgui.same_line(60 + col_i * col_width)
                imgui.text(f"S{stream_i}")
            imgui.separator()

            # Base colors - many distinct, bright colors
            base_colors = [
                (1.0, 0.2, 0.2),   # red
                (0.2, 0.8, 0.2),   # green
                (0.2, 0.4, 1.0),   # blue
                (1.0, 0.6, 0.1),   # orange
                (0.8, 0.2, 0.9),   # purple
                (0.1, 0.9, 0.9),   # cyan
                (0.9, 0.9, 0.2),   # yellow
                (1.0, 0.4, 0.7),   # pink
                (0.6, 0.4, 0.2),   # brown
                (0.4, 1.0, 0.6),   # mint
                (0.6, 0.6, 1.0),   # lavender
                (1.0, 0.8, 0.4),   # peach
            ]

            # Pre-compute colors for each stream at each level
            # - Same group: keep color
            # - Merge: blend colors
            # - Split: assign NEW distinct colors
            stream_colors_rgb = [[(0.5, 0.5, 0.5)] * max_stream_count for _ in range(max_levels)]
            group_to_color = {}  # group_key -> color
            next_base_color = 0

            for level_i in range(max_levels):
                if level_i < len(stream_groups):
                    for group in stream_groups[level_i]:
                        group_key = tuple(sorted(group))

                        if group_key in group_to_color:
                            # Already assigned (same group seen before)
                            color = group_to_color[group_key]
                        elif level_i == 0:
                            # First level: assign base color
                            color = base_colors[next_base_color % len(base_colors)]
                            next_base_color += 1
                            group_to_color[group_key] = color
                        else:
                            # Check previous level colors of streams in this group
                            prev_colors = []
                            for s in group:
                                if s < max_stream_count:
                                    prev_colors.append(stream_colors_rgb[level_i - 1][s])

                            unique_colors = list(set(prev_colors))

                            if len(unique_colors) > 1:
                                # Merge: blend all previous colors
                                r = sum(c[0] for c in unique_colors) / len(unique_colors)
                                g = sum(c[1] for c in unique_colors) / len(unique_colors)
                                b = sum(c[2] for c in unique_colors) / len(unique_colors)
                                color = (r, g, b)
                            else:
                                # New group (split or first appearance): new distinct color
                                color = base_colors[next_base_color % len(base_colors)]
                                next_base_color += 1

                            group_to_color[group_key] = color

                        # Assign color to all streams in group
                        for s in group:
                            if s < max_stream_count:
                                stream_colors_rgb[level_i][s] = color

            # Render rows (one per level)
            for level_i in range(max_levels):
                # Build stream-to-group mapping for this level
                stream_to_group = {}  # stream_i -> group list
                has_any_linked = False
                if level_i < len(stream_groups):
                    for group in stream_groups[level_i]:
                        is_multi = len(group) > 1
                        if is_multi:
                            has_any_linked = True
                        for s in group:
                            stream_to_group[s] = group if is_multi else None

                # Level label with link indicator
                if has_any_linked:
                    imgui.text_colored(f"{level_i:3d} *", 0.0, 1.0, 1.0, 1.0)
                else:
                    imgui.text(f"{level_i:3d}  ")

                # Checkbox for each stream at this level (in spatial display order)
                for col_i, stream_i in enumerate(stream_display_order):
                    imgui.same_line(60 + col_i * col_width)

                    if level_i < len(checkboxes[stream_i]):
                        checkbox_id = f"##lvl_{stream_i}_{level_i}"
                        is_checked = checkboxes[stream_i][level_i]

                        # Find THIS stream's group
                        this_group = stream_to_group.get(stream_i, None)
                        is_this_linked = this_group is not None and len(this_group) > 1

                        # Always show color based on group identity
                        color = stream_colors_rgb[level_i][stream_i]
                        # Tint checkbox frame - use bright colors
                        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, color[0] * 0.8, color[1] * 0.8, color[2] * 0.8, 1.0)
                        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, color[0], color[1], color[2], 1.0)
                        # Checkmark color - use black or white based on background brightness
                        brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
                        if brightness > 0.5:
                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0, 1.0)  # black
                        else:
                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, 1.0, 1.0, 1.0, 1.0)  # white

                        changed, new_value = imgui.checkbox(checkbox_id, is_checked)

                        imgui.pop_style_color(3)

                        if changed:
                            vis_changed = True
                            checkboxes[stream_i][level_i] = new_value

                            # If linked, update all streams in THIS stream's group
                            if is_this_linked:
                                for other_stream in this_group:
                                    if other_stream != stream_i and other_stream < max_stream_count:
                                        if level_i < len(checkboxes[other_stream]):
                                            checkboxes[other_stream][level_i] = new_value
                    else:
                        # This stream doesn't have this level - show placeholder
                        imgui.text("-")

            imgui.end_child()

            # Update visualization if checkboxes changed
            if vis_changed:
                obj._update_level_select_visualization()

            # Buttons
            button_width = 120
            if imgui.button("Undo Selection", button_width, 30):
                obj._undo_level_selection()

            imgui.same_line()
            if imgui.button("Finish Select", button_width, 30):
                obj._apply_level_selection()

            imgui.end()

    def reset(self, reset_time=None):
        self.env.reset(reset_time)
        self.reward_buffer = [self.env.get_reward()]
        # Reset soft body and VIPER simulations
        for name, obj in self.zygote_muscle_meshes.items():
            if obj.soft_body is not None:
                obj.reset_soft_body()
            if hasattr(obj, 'viper_sim') and obj.viper_sim is not None:
                obj.reset_viper()

    def zero_reset(self):
        self.env.zero_reset()
        self.reward_buffer = [self.env.get_reward()]
        # Reset soft body and VIPER simulations
        for name, obj in self.zygote_muscle_meshes.items():
            if obj.soft_body is not None:
                obj.reset_soft_body()
            if hasattr(obj, 'viper_sim') and obj.viper_sim is not None:
                obj.reset_viper()

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

    def startLoop(self):        
        while not glfw.window_should_close(self.window):
            start_time = time.time()
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

            while time.time() - start_time < 1.0 / 30:
                time.sleep(1E-6)

        self.impl.shutdown()
        glfw.terminate()
        return

