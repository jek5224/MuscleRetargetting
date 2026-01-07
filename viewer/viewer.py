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
        self.inter_muscle_constraint_threshold = 0.01  # 1cm default threshold
        self.coupled_as_unified_volume = True  # Treat all muscles as one unified system

        # 2D Inspect window state
        self.inspect_2d_open = {}  # Dict: muscle_name -> bool (window open state)
        self.inspect_2d_stream_idx = {}  # Dict: muscle_name -> selected stream index
        self.inspect_2d_contour_idx = {}  # Dict: muscle_name -> selected contour index

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
                if obj.is_draw:
                    obj.draw([obj.color[0], obj.color[1], obj.color[2], obj.transparency])
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

                imgui.same_line()
                imgui.text(f"{len(self.inter_muscle_constraints)} constraints")

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

                        # Left column: "Process All" button (height matches 8 buttons on right)
                        num_process_buttons = 8
                        process_all_height = num_process_buttons * imgui.get_frame_height() + (num_process_buttons - 1) * imgui.get_style().item_spacing[1]
                        if imgui.button(f"Process\nAll##{name}", width=100, height=process_all_height):
                            try:
                                print(f"[{name}] Running full pipeline...")
                                # Step 1: Scalar Field
                                if len(obj.edge_groups) > 0 and len(obj.edge_classes) > 0:
                                    print(f"  [1/7] Computing Scalar Field...")
                                    obj.compute_scalar_field()
                                else:
                                    print(f"  [1/7] Skipping Scalar Field (no edge groups)")

                                # Step 2: Find Contours
                                if obj.scalar_field is not None:
                                    print(f"  [2/7] Finding Contours...")
                                    obj.find_contours(skeleton_meshes=self.zygote_skeleton_meshes)

                                # Step 3: Refine Contours
                                if obj.contours is not None and len(obj.contours) > 0:
                                    print(f"  [3/7] Refining Contours...")
                                    obj.refine_contours(max_spacing_threshold=0.01)

                                # Step 4: Smoothen Contours
                                if obj.contours is not None and len(obj.contours) > 0:
                                    print(f"  [4/7] Smoothening Contours...")
                                    obj.smoothen_contours()

                                # Step 5: Find Streams
                                if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                                    print(f"  [5/7] Finding Streams...")
                                    obj.find_contour_stream(skeleton_meshes=self.zygote_skeleton_meshes)

                                # Step 6: Resample Contours
                                if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                                    print(f"  [6/7] Resampling Contours...")
                                    obj.resample_contours(num_samples=32)


                                # Step 7: Build Contour Mesh
                                if obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
                                    print(f"  [7/7] Building Contour Mesh...")
                                    obj.build_contour_mesh()

                                # Step 8: Tetrahedralize
                                if obj.contour_mesh_vertices is not None:
                                    print(f"  [8/8] Tetrahedralizing...")
                                    obj.soft_body = None
                                    obj.tetrahedralize_contour_mesh()
                                    if obj.tet_vertices is not None:
                                        obj.is_draw_contours = False
                                        obj.is_draw_tet_mesh = True

                                print(f"[{name}] Pipeline complete!")
                            except Exception as e:
                                print(f"[{name}] Pipeline error: {e}")
                                import traceback
                                traceback.print_exc()

                        # Right column: Individual buttons (use -1 to auto-fill column width)
                        imgui.next_column()
                        col_button_width = 180  # Fits in the right column

                        if imgui.button(f"Scalar Field##{name}", width=col_button_width):
                            if len(obj.edge_groups) > 0 and len(obj.edge_classes) > 0:
                                try:
                                    obj.compute_scalar_field()
                                except Exception as e:
                                    print(f"[{name}] Scalar Field error: {e}")
                            else:
                                print(f"[{name}] Need edge_groups and edge_classes")
                        if imgui.button(f"Find Contours##{name}", width=col_button_width):
                            if obj.scalar_field is not None:
                                try:
                                    obj.find_contours(skeleton_meshes=self.zygote_skeleton_meshes)
                                except Exception as e:
                                    print(f"[{name}] Find Contours error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Scalar Field' first")
                        if imgui.button(f"Refine Contours##{name}", width=col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0:
                                try:
                                    obj.refine_contours(max_spacing_threshold=0.01)
                                except Exception as e:
                                    print(f"[{name}] Refine Contours error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Contours' first")
                        if imgui.button(f"Smoothen Contours##{name}", width=col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0:
                                try:
                                    obj.smoothen_contours()
                                except Exception as e:
                                    print(f"[{name}] Smoothen Contours error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Contours' first")
                        if imgui.button(f"Find Streams##{name}", width=col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None and len(obj.bounding_planes) > 0:
                                try:
                                    obj.find_contour_stream(skeleton_meshes=self.zygote_skeleton_meshes)
                                except Exception as e:
                                    print(f"[{name}] Find Streams error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Contours' first")
                        if imgui.button(f"Optimize Streams##{name}", width=col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None and len(obj.bounding_planes) > 0:
                                try:
                                    obj.optimize_contour_stream(shape_threshold=0.1, min_contours=3)
                                except Exception as e:
                                    print(f"[{name}] Optimize Streams error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Streams' first")
                        if imgui.button(f"Resample Contours##{name}", width=col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0 and obj.bounding_planes is not None:
                                try:
                                    obj.resample_contours(num_samples=32)
                                except Exception as e:
                                    print(f"[{name}] Resample Contours error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Smoothen Contours' first")
                        if imgui.button(f"Build Contour Mesh##{name}", width=col_button_width):
                            if obj.contours is not None and len(obj.contours) > 0 and obj.draw_contour_stream is not None:
                                try:
                                    obj.build_contour_mesh()
                                except Exception as e:
                                    print(f"[{name}] Build Contour Mesh error: {e}")
                            else:
                                print(f"[{name}] Prerequisites: Run 'Find Streams' first")

                        # Tetrahedralization for soft body simulation
                        if imgui.button(f"Tetrahedralize##{name}", width=col_button_width):
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

                        # Inspect 2D button - opens visualization window for fiber samples and contours
                        inspect_width = button_width * 2 + imgui.get_style().item_spacing[0]
                        has_fiber_data = (hasattr(obj, 'fiber_architecture') and obj.fiber_architecture is not None and
                                         len(obj.fiber_architecture) > 0 and hasattr(obj, 'contours') and
                                         obj.contours is not None and len(obj.contours) > 0)
                        if not has_fiber_data:
                            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                        if imgui.button(f"Inspect 2D##{name}", width=inspect_width):
                            if has_fiber_data:
                                self.inspect_2d_open[name] = True
                                if name not in self.inspect_2d_stream_idx:
                                    self.inspect_2d_stream_idx[name] = 0
                                if name not in self.inspect_2d_contour_idx:
                                    self.inspect_2d_contour_idx[name] = 0
                            else:
                                print(f"[{name}] No fiber/contour data. Run 'Find Streams' first.")
                        if not has_fiber_data:
                            imgui.pop_style_var()

                        _, obj.is_one_fiber = imgui.checkbox(f"One Fiber##{name}", obj.is_one_fiber)

                        # Sampling method selector
                        sampling_methods = ['sobol_unit_square', 'sobol_min_contour']
                        current_idx = sampling_methods.index(obj.sampling_method) if obj.sampling_method in sampling_methods else 0
                        changed, new_idx = imgui.combo(f"Sampling##{name}", current_idx, sampling_methods)
                        if changed:
                            obj.sampling_method = sampling_methods[new_idx]

                        # Cutting method selector
                        cutting_methods = ['area_based', 'voronoi', 'angular', 'gradient', 'ratio', 'cumulative_area', 'projected_area']
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
                                imgui.text(f"{obj.edge_classes[i]}")
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
                        _, obj.is_draw_edges = imgui.checkbox("Draw Edges", obj.is_draw_edges)
                        _, obj.is_draw_centroid = imgui.checkbox("Draw Centroid", obj.is_draw_centroid)
                        _, obj.is_draw_bounding_box = imgui.checkbox("Draw Bounding Box", obj.is_draw_bounding_box)
                        _, obj.is_draw_discarded = imgui.checkbox("Draw Discarded", obj.is_draw_discarded)
                        _, obj.is_draw_fiber_architecture = imgui.checkbox("Draw Fiber Architecture", obj.is_draw_fiber_architecture)
                        _, obj.is_draw_contour_mesh = imgui.checkbox("Draw Contour Mesh", obj.is_draw_contour_mesh)
                        _, obj.is_draw_tet_mesh = imgui.checkbox("Draw Tet Mesh", obj.is_draw_tet_mesh)
                        imgui.same_line()
                        _, obj.is_draw_tet_edges = imgui.checkbox("Tet Edges", obj.is_draw_tet_edges)

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

            # Check if data is available
            if (not hasattr(obj, 'fiber_architecture') or obj.fiber_architecture is None or
                len(obj.fiber_architecture) == 0 or not hasattr(obj, 'contours') or
                obj.contours is None or len(obj.contours) == 0):
                muscles_to_close.append(name)
                continue

            # Window setup
            imgui.set_next_window_size(700, 400, imgui.FIRST_USE_EVER)
            expanded, opened = imgui.begin(f"Inspect 2D: {name}", True)

            if not opened:
                muscles_to_close.append(name)
                imgui.end()
                continue

            # Stream and contour selection
            num_streams = len(obj.fiber_architecture)
            stream_idx = self.inspect_2d_stream_idx.get(name, 0)
            stream_idx = min(stream_idx, num_streams - 1)

            changed, new_stream_idx = imgui.slider_int(f"Stream##{name}_inspect", stream_idx, 0, max(0, num_streams - 1))
            if changed:
                self.inspect_2d_stream_idx[name] = new_stream_idx
                stream_idx = new_stream_idx

            num_contours = len(obj.contours[stream_idx]) if stream_idx < len(obj.contours) else 0
            contour_idx = self.inspect_2d_contour_idx.get(name, 0)
            contour_idx = min(contour_idx, max(0, num_contours - 1))

            changed, new_contour_idx = imgui.slider_int(f"Contour##{name}_inspect", contour_idx, 0, max(0, num_contours - 1))
            if changed:
                self.inspect_2d_contour_idx[name] = new_contour_idx
                contour_idx = new_contour_idx

            imgui.separator()

            # Get canvas dimensions
            canvas_size = 280
            padding = 20

            # Two columns: fiber samples (left) and contour (right)
            imgui.columns(2, f"inspect_cols##{name}", border=True)

            # Left column: Fiber samples on unit square
            imgui.text("Fiber Samples (Unit Square)")
            cursor_pos = imgui.get_cursor_screen_pos()
            draw_list = imgui.get_window_draw_list()

            # Draw unit square background
            x0, y0 = cursor_pos[0] + padding, cursor_pos[1] + padding
            x1, y1 = x0 + canvas_size, y0 + canvas_size

            # Background
            draw_list.add_rect_filled(x0, y0, x1, y1, imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))
            # Border
            draw_list.add_rect(x0, y0, x1, y1, imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 1.0), thickness=2.0)

            # Draw fiber samples
            if stream_idx < len(obj.fiber_architecture):
                fiber_samples = obj.fiber_architecture[stream_idx]
                for sample in fiber_samples:
                    if len(sample) >= 2:
                        # Sample is in [0,1] x [0,1]
                        sx = x0 + sample[0] * canvas_size
                        sy = y0 + (1 - sample[1]) * canvas_size  # Flip Y for screen coords
                        draw_list.add_circle_filled(sx, sy, 4, imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 1.0))

            # Reserve space for the drawing
            imgui.dummy(canvas_size + 2 * padding, canvas_size + 2 * padding)

            # Right column: Contour with waypoints
            imgui.next_column()
            imgui.text(f"Contour {contour_idx} with Waypoints")
            cursor_pos = imgui.get_cursor_screen_pos()

            # Draw contour background
            x0, y0 = cursor_pos[0] + padding, cursor_pos[1] + padding
            x1, y1 = x0 + canvas_size, y0 + canvas_size

            # Background
            draw_list.add_rect_filled(x0, y0, x1, y1, imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))

            # Get contour and project to 2D
            if (stream_idx < len(obj.contours) and contour_idx < len(obj.contours[stream_idx]) and
                stream_idx < len(obj.bounding_planes) and contour_idx < len(obj.bounding_planes[stream_idx])):

                contour = np.array(obj.contours[stream_idx][contour_idx])
                plane_info = obj.bounding_planes[stream_idx][contour_idx]

                if len(contour) > 0 and 'basis_x' in plane_info and 'basis_y' in plane_info:
                    mean = plane_info['mean']
                    basis_x = plane_info['basis_x']
                    basis_y = plane_info['basis_y']

                    # Project contour to 2D
                    contour_2d = np.array([[np.dot(v - mean, basis_x), np.dot(v - mean, basis_y)] for v in contour])

                    # Find bounding box for normalization
                    min_xy = contour_2d.min(axis=0)
                    max_xy = contour_2d.max(axis=0)
                    range_xy = max_xy - min_xy
                    range_xy[range_xy < 1e-10] = 1.0  # Avoid division by zero

                    # Normalize to [0,1] with some margin
                    margin = 0.1
                    contour_2d_norm = (contour_2d - min_xy) / range_xy * (1 - 2 * margin) + margin

                    # Draw contour polygon
                    points = []
                    for pt in contour_2d_norm:
                        px = x0 + pt[0] * canvas_size
                        py = y0 + (1 - pt[1]) * canvas_size
                        points.append((px, py))

                    # Draw contour lines
                    for i in range(len(points)):
                        p1 = points[i]
                        p2 = points[(i + 1) % len(points)]
                        draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                          imgui.get_color_u32_rgba(0.8, 0.8, 0.2, 1.0), thickness=2.0)

                    # Draw waypoints if available
                    if (hasattr(obj, 'waypoints') and obj.waypoints is not None and
                        stream_idx < len(obj.waypoints) and contour_idx < len(obj.waypoints[stream_idx])):

                        waypoints_3d = obj.waypoints[stream_idx][contour_idx]
                        if waypoints_3d is not None and len(waypoints_3d) > 0:
                            for wp in waypoints_3d:
                                wp = np.array(wp)
                                wp_2d = np.array([np.dot(wp - mean, basis_x), np.dot(wp - mean, basis_y)])
                                wp_norm = (wp_2d - min_xy) / range_xy * (1 - 2 * margin) + margin
                                wpx = x0 + wp_norm[0] * canvas_size
                                wpy = y0 + (1 - wp_norm[1]) * canvas_size
                                draw_list.add_circle_filled(wpx, wpy, 5, imgui.get_color_u32_rgba(0.9, 0.3, 0.3, 1.0))

                    # Draw bounding box
                    bp = plane_info.get('bounding_plane', None)
                    if bp is not None and len(bp) >= 4:
                        bp_2d = np.array([[np.dot(v - mean, basis_x), np.dot(v - mean, basis_y)] for v in bp])
                        bp_norm = (bp_2d - min_xy) / range_xy * (1 - 2 * margin) + margin
                        bp_points = []
                        for pt in bp_norm:
                            bpx = x0 + pt[0] * canvas_size
                            bpy = y0 + (1 - pt[1]) * canvas_size
                            bp_points.append((bpx, bpy))
                        for i in range(4):
                            p1 = bp_points[i]
                            p2 = bp_points[(i + 1) % 4]
                            draw_list.add_line(p1[0], p1[1], p2[0], p2[1],
                                              imgui.get_color_u32_rgba(0.3, 0.5, 0.9, 1.0), thickness=1.5)

            # Reserve space
            imgui.dummy(canvas_size + 2 * padding, canvas_size + 2 * padding)

            imgui.columns(1)
            imgui.end()

        # Close windows that were marked for closing
        for name in muscles_to_close:
            self.inspect_2d_open[name] = False

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

