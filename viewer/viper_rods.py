"""
VIPER-style Rod Simulation for Muscle Deformation

Based on: "VIPER: Volume Invariant Position-based Elastic Rods" (SCA 2019)
https://github.com/vcg-uvic/viper

Key innovations:
- Muscles represented as rod bundles (from waypoints/fiber architecture)
- Each vertex has: position (3), quaternion (4), scale (1) = 8 DOFs
- Volume preservation through isotropic scale adjustment
- Position-Based Dynamics (PBD) for real-time performance

Integration with existing system:
- Waypoints become rod vertices
- Fiber streams become individual rods
- Contour mesh is skinned from rod bundle
"""

import numpy as np

# Try to import taichi, fall back gracefully
try:
    import taichi as ti
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False
    print("Warning: Taichi not available, VIPER simulation will use NumPy fallback")

# Global Taichi initialization (only once per process)
_TAICHI_INITIALIZED = False
_TAICHI_ARCH = None

def _init_taichi_global():
    """Initialize Taichi globally (only once)."""
    global _TAICHI_INITIALIZED, _TAICHI_ARCH
    if _TAICHI_INITIALIZED or not TAICHI_AVAILABLE:
        return _TAICHI_ARCH

    try:
        ti.init(arch=ti.cuda, offline_cache=True)
        _TAICHI_ARCH = "cuda"
        print("VIPER: Taichi initialized with CUDA (GPU)")
    except Exception as e:
        print(f"VIPER: CUDA init failed ({e}), falling back to CPU")
        ti.init(arch=ti.cpu, offline_cache=True)
        _TAICHI_ARCH = "cpu"
        print("VIPER: Taichi initialized with CPU")

    _TAICHI_INITIALIZED = True
    return _TAICHI_ARCH


def check_viper_available():
    """Check if VIPER simulation is available (requires Taichi)."""
    return TAICHI_AVAILABLE


def get_viper_backend():
    """Get the Taichi backend being used (cuda/cpu/None)."""
    return _TAICHI_ARCH


@ti.data_oriented
class ViperRodSimulation:
    """
    VIPER-style rod bundle simulation for muscles.

    Converts waypoints/fiber architecture to rod representation and
    simulates using Position-Based Dynamics with volume preservation.
    """

    def __init__(self, use_gpu=True):
        """
        Initialize VIPER rod simulation.

        Args:
            use_gpu: Use GPU acceleration via Taichi
        """
        self.use_gpu = use_gpu and TAICHI_AVAILABLE
        self.initialized = False

        # Rod data
        self.num_rods = 0
        self.num_vertices_per_rod = 0
        self.total_vertices = 0

        # Simulation parameters
        self.dt = 1.0 / 60.0  # 60 Hz default
        self.iterations = 4  # PBD iterations
        self.damping = 0.99
        self.gravity = np.array([0.0, -9.81, 0.0])

        # Constraint stiffness (0-1, will be converted to compliance)
        self.stretch_stiffness = 0.9
        self.bend_stiffness = 0.5
        self.volume_stiffness = 0.8
        self.shape_stiffness = 0.3

        # Constraint toggles
        self.enable_stretch_constraint = True
        self.enable_bend_constraint = True
        self.enable_volume_constraint = True  # VIPER key feature
        self.enable_shape_constraint = True
        self.enable_collision = False  # Collision with skeleton meshes

        # Collision settings
        self.collision_margin = 0.002  # 2mm margin
        self.collision_stiffness = 0.9
        self.collision_meshes = []  # List of (vertices, faces) for collision

        # Rest state
        self.rest_positions = None
        self.rest_lengths = None
        self.rest_volumes = None
        self.rest_quaternions = None

        # Current state
        self.positions = None
        self.velocities = None
        self.quaternions = None  # Rotation per vertex
        self.scales = None  # Isotropic scale per vertex (VIPER key feature)
        self.scale_velocities = None

        # Fixed vertices (endpoints attached to skeleton)
        self.fixed_mask = None
        self.fixed_targets = None

        # Skinning data (rod vertices -> mesh vertices)
        self.skinning_weights = None
        self.skinning_indices = None

        # Initialize Taichi if available
        if self.use_gpu:
            self._init_taichi()

    def _init_taichi(self):
        """Initialize Taichi runtime (uses global init)."""
        if not TAICHI_AVAILABLE:
            return

        # Use global initialization (only happens once)
        arch = _init_taichi_global()
        self._taichi_initialized = (arch is not None)
        if self._taichi_initialized:
            print(f"VIPER: Using {arch.upper()} backend")

    def build_from_waypoints(self, waypoints, fixed_indices=None):
        """
        Build rod bundle from waypoints (fiber architecture).

        Args:
            waypoints: List of waypoint arrays, each shape (num_points, 3)
                      Each array is one fiber/rod
            fixed_indices: Dict mapping rod_idx -> list of vertex indices that are fixed

        Returns:
            True if successful
        """
        if not waypoints or len(waypoints) == 0:
            print("VIPER: No waypoints provided")
            return False

        # Count rods and vertices
        self.num_rods = len(waypoints)

        # Find max vertices per rod (pad shorter ones)
        max_verts = max(len(wp) for wp in waypoints)
        self.num_vertices_per_rod = max_verts
        self.total_vertices = self.num_rods * max_verts

        print(f"VIPER: Building {self.num_rods} rods, {max_verts} vertices each, {self.total_vertices} total")

        # Allocate arrays
        self.positions = np.zeros((self.num_rods, max_verts, 3), dtype=np.float64)
        self.velocities = np.zeros((self.num_rods, max_verts, 3), dtype=np.float64)
        self.quaternions = np.zeros((self.num_rods, max_verts, 4), dtype=np.float64)
        self.quaternions[:, :, 3] = 1.0  # Identity quaternion (w=1)
        self.scales = np.ones((self.num_rods, max_verts), dtype=np.float64)
        self.scale_velocities = np.zeros((self.num_rods, max_verts), dtype=np.float64)

        # Valid vertex mask (for padded rods)
        self.valid_mask = np.zeros((self.num_rods, max_verts), dtype=bool)

        # Fill in waypoint positions
        for rod_idx, wp in enumerate(waypoints):
            n = len(wp)
            self.positions[rod_idx, :n] = wp
            self.valid_mask[rod_idx, :n] = True

            # Pad with last vertex if needed
            if n < max_verts:
                self.positions[rod_idx, n:] = wp[-1]

        # Store rest positions
        self.rest_positions = self.positions.copy()

        # Compute rest lengths (distance between consecutive vertices)
        self.rest_lengths = np.zeros((self.num_rods, max_verts - 1), dtype=np.float64)
        for rod_idx in range(self.num_rods):
            for i in range(max_verts - 1):
                if self.valid_mask[rod_idx, i] and self.valid_mask[rod_idx, i + 1]:
                    diff = self.positions[rod_idx, i + 1] - self.positions[rod_idx, i]
                    self.rest_lengths[rod_idx, i] = np.linalg.norm(diff)

        # Compute rest quaternions (frame along rod)
        self._compute_rest_frames()

        # Compute rest volumes (per segment, using cross-sectional area)
        self._compute_rest_volumes()

        # Fixed vertices mask
        self.fixed_mask = np.zeros((self.num_rods, max_verts), dtype=bool)
        self.fixed_targets = np.zeros((self.num_rods, max_verts, 3), dtype=np.float64)

        if fixed_indices:
            for rod_idx, vert_indices in fixed_indices.items():
                if rod_idx < self.num_rods:
                    for v_idx in vert_indices:
                        if v_idx < max_verts:
                            self.fixed_mask[rod_idx, v_idx] = True
                            self.fixed_targets[rod_idx, v_idx] = self.positions[rod_idx, v_idx]
        else:
            # Default: fix first and last vertex of each rod
            for rod_idx in range(self.num_rods):
                self.fixed_mask[rod_idx, 0] = True
                self.fixed_targets[rod_idx, 0] = self.positions[rod_idx, 0]

                # Find last valid vertex
                last_valid = np.where(self.valid_mask[rod_idx])[0]
                if len(last_valid) > 0:
                    last_idx = last_valid[-1]
                    self.fixed_mask[rod_idx, last_idx] = True
                    self.fixed_targets[rod_idx, last_idx] = self.positions[rod_idx, last_idx]

        # Build Taichi fields if available
        if self.use_gpu and TAICHI_AVAILABLE:
            self._build_taichi_fields()

        self.initialized = True
        print(f"VIPER: Initialization complete")
        return True

    def _compute_rest_frames(self):
        """Compute rest frame (quaternion) for each vertex based on rod tangent."""
        self.rest_quaternions = np.zeros((self.num_rods, self.num_vertices_per_rod, 4), dtype=np.float64)
        self.rest_quaternions[:, :, 3] = 1.0  # Default identity

        for rod_idx in range(self.num_rods):
            for i in range(self.num_vertices_per_rod):
                if not self.valid_mask[rod_idx, i]:
                    continue

                # Compute tangent direction
                if i == 0:
                    if self.valid_mask[rod_idx, i + 1]:
                        tangent = self.positions[rod_idx, i + 1] - self.positions[rod_idx, i]
                    else:
                        tangent = np.array([0, 1, 0])
                elif i == self.num_vertices_per_rod - 1 or not self.valid_mask[rod_idx, i + 1]:
                    tangent = self.positions[rod_idx, i] - self.positions[rod_idx, i - 1]
                else:
                    tangent = self.positions[rod_idx, i + 1] - self.positions[rod_idx, i - 1]

                # Normalize
                length = np.linalg.norm(tangent)
                if length > 1e-10:
                    tangent = tangent / length
                else:
                    tangent = np.array([0, 1, 0])

                # Create quaternion from tangent (align Y-axis to tangent)
                self.rest_quaternions[rod_idx, i] = self._quat_from_direction(tangent)

        self.quaternions = self.rest_quaternions.copy()

    def _quat_from_direction(self, direction):
        """Create quaternion that aligns Y-axis to given direction."""
        y_axis = np.array([0, 1, 0])

        # Handle parallel/anti-parallel cases
        dot = np.dot(y_axis, direction)
        if dot > 0.9999:
            return np.array([0, 0, 0, 1])  # Identity
        elif dot < -0.9999:
            return np.array([1, 0, 0, 0])  # 180 degree rotation around X

        # Rotation axis is cross product
        axis = np.cross(y_axis, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-10:
            axis = axis / axis_len
        else:
            return np.array([0, 0, 0, 1])

        # Rotation angle
        angle = np.arccos(np.clip(dot, -1, 1))

        # Quaternion from axis-angle
        s = np.sin(angle / 2)
        c = np.cos(angle / 2)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])

    def _compute_rest_volumes(self):
        """
        Compute rest volumes for each rod segment.
        Volume = length * cross_sectional_area
        We use a default radius that can be adjusted.
        """
        default_radius = 0.01  # 1cm default cross-section radius

        self.rest_volumes = np.zeros((self.num_rods, self.num_vertices_per_rod - 1), dtype=np.float64)
        self.rest_radii = np.full((self.num_rods, self.num_vertices_per_rod), default_radius, dtype=np.float64)

        for rod_idx in range(self.num_rods):
            for i in range(self.num_vertices_per_rod - 1):
                if self.valid_mask[rod_idx, i] and self.valid_mask[rod_idx, i + 1]:
                    length = self.rest_lengths[rod_idx, i]
                    # Cylinder volume: pi * r^2 * length
                    self.rest_volumes[rod_idx, i] = np.pi * default_radius**2 * length

    def _build_taichi_fields(self):
        """Build Taichi fields for GPU computation."""
        if not TAICHI_AVAILABLE:
            return

        n_rods = self.num_rods
        n_verts = self.num_vertices_per_rod

        # Position and velocity fields
        self.ti_positions = ti.Vector.field(3, dtype=ti.f64, shape=(n_rods, n_verts))
        self.ti_velocities = ti.Vector.field(3, dtype=ti.f64, shape=(n_rods, n_verts))
        self.ti_rest_positions = ti.Vector.field(3, dtype=ti.f64, shape=(n_rods, n_verts))

        # Quaternion fields
        self.ti_quaternions = ti.Vector.field(4, dtype=ti.f64, shape=(n_rods, n_verts))
        self.ti_rest_quaternions = ti.Vector.field(4, dtype=ti.f64, shape=(n_rods, n_verts))

        # Scale fields (VIPER key feature)
        self.ti_scales = ti.field(dtype=ti.f64, shape=(n_rods, n_verts))
        self.ti_scale_velocities = ti.field(dtype=ti.f64, shape=(n_rods, n_verts))

        # Rest lengths and volumes
        self.ti_rest_lengths = ti.field(dtype=ti.f64, shape=(n_rods, n_verts - 1))
        self.ti_rest_volumes = ti.field(dtype=ti.f64, shape=(n_rods, n_verts - 1))

        # Masks
        self.ti_valid_mask = ti.field(dtype=ti.i32, shape=(n_rods, n_verts))
        self.ti_fixed_mask = ti.field(dtype=ti.i32, shape=(n_rods, n_verts))
        self.ti_fixed_targets = ti.Vector.field(3, dtype=ti.f64, shape=(n_rods, n_verts))

        # Copy data to Taichi fields
        self._sync_to_taichi()

        # Build Taichi kernels
        self._build_taichi_kernels()

    def _sync_to_taichi(self):
        """Sync NumPy arrays to Taichi fields."""
        if not self.use_gpu or not TAICHI_AVAILABLE:
            return

        self.ti_positions.from_numpy(self.positions)
        self.ti_velocities.from_numpy(self.velocities)
        self.ti_rest_positions.from_numpy(self.rest_positions)
        self.ti_quaternions.from_numpy(self.quaternions)
        self.ti_rest_quaternions.from_numpy(self.rest_quaternions)
        self.ti_scales.from_numpy(self.scales)
        self.ti_scale_velocities.from_numpy(self.scale_velocities)
        self.ti_rest_lengths.from_numpy(self.rest_lengths)
        self.ti_rest_volumes.from_numpy(self.rest_volumes)
        self.ti_valid_mask.from_numpy(self.valid_mask.astype(np.int32))
        self.ti_fixed_mask.from_numpy(self.fixed_mask.astype(np.int32))
        self.ti_fixed_targets.from_numpy(self.fixed_targets)

    def _sync_from_taichi(self):
        """Sync Taichi fields back to NumPy arrays."""
        if not self.use_gpu or not TAICHI_AVAILABLE:
            return

        self.positions = self.ti_positions.to_numpy()
        self.velocities = self.ti_velocities.to_numpy()
        self.quaternions = self.ti_quaternions.to_numpy()
        self.scales = self.ti_scales.to_numpy()
        self.scale_velocities = self.ti_scale_velocities.to_numpy()

    def _sync_positions_from_taichi(self):
        """Sync only positions from Taichi (for collision)."""
        if not self.use_gpu or not TAICHI_AVAILABLE:
            return
        self.positions = self.ti_positions.to_numpy()

    def _sync_positions_to_taichi(self):
        """Sync only positions to Taichi (after collision)."""
        if not self.use_gpu or not TAICHI_AVAILABLE:
            return
        self.ti_positions.from_numpy(self.positions)

    def _build_taichi_kernels(self):
        """Build Taichi kernels for PBD simulation."""
        if not TAICHI_AVAILABLE:
            return

        # References to fields for kernel access
        positions = self.ti_positions
        velocities = self.ti_velocities
        rest_positions = self.ti_rest_positions
        quaternions = self.ti_quaternions
        scales = self.ti_scales
        scale_velocities = self.ti_scale_velocities
        rest_lengths = self.ti_rest_lengths
        rest_volumes = self.ti_rest_volumes
        valid_mask = self.ti_valid_mask
        fixed_mask = self.ti_fixed_mask
        fixed_targets = self.ti_fixed_targets

        n_rods = self.num_rods
        n_verts = self.num_vertices_per_rod

        @ti.kernel
        def predict_positions(dt: ti.f64, damping: ti.f64, gx: ti.f64, gy: ti.f64, gz: ti.f64):
            """Predict positions using semi-implicit Euler."""
            gravity = ti.Vector([gx, gy, gz], dt=ti.f64)
            for rod_idx, vert_idx in ti.ndrange(n_rods, n_verts):
                if valid_mask[rod_idx, vert_idx] == 0:
                    continue
                if fixed_mask[rod_idx, vert_idx] == 1:
                    positions[rod_idx, vert_idx] = fixed_targets[rod_idx, vert_idx]
                    velocities[rod_idx, vert_idx] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
                else:
                    # Apply gravity and damping
                    v = velocities[rod_idx, vert_idx]
                    v = v * damping + gravity * dt
                    velocities[rod_idx, vert_idx] = v
                    positions[rod_idx, vert_idx] = positions[rod_idx, vert_idx] + v * dt

        @ti.kernel
        def solve_stretch_constraints(compliance: ti.f64):
            """Solve stretch constraints (maintain rod segment lengths)."""
            for rod_idx, seg_idx in ti.ndrange(n_rods, n_verts - 1):
                if valid_mask[rod_idx, seg_idx] == 0 or valid_mask[rod_idx, seg_idx + 1] == 0:
                    continue

                p0 = positions[rod_idx, seg_idx]
                p1 = positions[rod_idx, seg_idx + 1]
                s0 = scales[rod_idx, seg_idx]
                s1 = scales[rod_idx, seg_idx + 1]

                # Current length (scaled by average scale)
                diff = p1 - p0
                current_length = diff.norm()

                if current_length < 1e-10:
                    continue

                # Target length (rest length scaled by average scale)
                avg_scale = (s0 + s1) * 0.5
                target_length = rest_lengths[rod_idx, seg_idx] * avg_scale

                # Constraint: C = current_length - target_length
                C = current_length - target_length

                # Gradient magnitude
                grad_mag = ti.cast(1.0, ti.f64)

                # Compute correction
                w0 = ti.cast(1.0, ti.f64) if fixed_mask[rod_idx, seg_idx] == 0 else ti.cast(0.0, ti.f64)
                w1 = ti.cast(1.0, ti.f64) if fixed_mask[rod_idx, seg_idx + 1] == 0 else ti.cast(0.0, ti.f64)

                w_sum = w0 + w1 + compliance
                if w_sum > 1e-10:
                    delta_lambda = -C / w_sum
                    correction = diff / current_length * delta_lambda

                    if w0 > 0:
                        positions[rod_idx, seg_idx] = p0 - correction * w0
                    if w1 > 0:
                        positions[rod_idx, seg_idx + 1] = p1 + correction * w1

        @ti.kernel
        def solve_volume_constraints(compliance: ti.f64):
            """
            Solve volume constraints - VIPER key feature!
            Adjusts scale to preserve volume when rod stretches.
            Volume = pi * r^2 * L = const
            When L increases, r (and scale) must decrease.
            """
            for rod_idx, seg_idx in ti.ndrange(n_rods, n_verts - 1):
                if valid_mask[rod_idx, seg_idx] == 0 or valid_mask[rod_idx, seg_idx + 1] == 0:
                    continue

                p0 = positions[rod_idx, seg_idx]
                p1 = positions[rod_idx, seg_idx + 1]
                s0 = scales[rod_idx, seg_idx]
                s1 = scales[rod_idx, seg_idx + 1]

                # Current length
                diff = p1 - p0
                current_length = diff.norm()

                if current_length < 1e-10:
                    continue

                rest_length = rest_lengths[rod_idx, seg_idx]
                rest_volume = rest_volumes[rod_idx, seg_idx]

                if rest_volume < 1e-15:
                    continue

                # Current volume: pi * (avg_scale * r0)^2 * current_length
                # Simplified: volume proportional to scale^2 * length
                avg_scale = (s0 + s1) * 0.5
                current_volume = avg_scale * avg_scale * current_length

                # Rest volume (normalized): 1.0 * rest_length (scale=1 at rest)
                rest_vol_normalized = rest_length

                # Constraint: current_volume = rest_volume
                # C = current_volume - rest_volume
                C = current_volume - rest_vol_normalized

                # To preserve volume when length increases, scale must decrease
                # d(volume)/d(scale) = 2 * scale * length
                grad_scale = ti.cast(2.0, ti.f64) * avg_scale * current_length

                if ti.abs(grad_scale) < 1e-10:
                    continue

                # Compute scale correction
                w0 = ti.cast(1.0, ti.f64) if fixed_mask[rod_idx, seg_idx] == 0 else ti.cast(0.0, ti.f64)
                w1 = ti.cast(1.0, ti.f64) if fixed_mask[rod_idx, seg_idx + 1] == 0 else ti.cast(0.0, ti.f64)

                denom = grad_scale * grad_scale * (w0 + w1) + compliance
                if denom > 1e-10:
                    delta_lambda = -C / denom
                    delta_scale = grad_scale * delta_lambda * 0.5  # Distribute to both

                    # Clamp scale changes to prevent instability
                    delta_scale = ti.max(ti.min(delta_scale, ti.cast(0.1, ti.f64)), ti.cast(-0.1, ti.f64))

                    if w0 > 0:
                        new_s0 = s0 + delta_scale
                        scales[rod_idx, seg_idx] = ti.max(new_s0, ti.cast(0.3, ti.f64))  # Min scale 0.3
                    if w1 > 0:
                        new_s1 = s1 + delta_scale
                        scales[rod_idx, seg_idx + 1] = ti.max(new_s1, ti.cast(0.3, ti.f64))

        @ti.kernel
        def solve_bend_constraints(compliance: ti.f64):
            """Solve bending constraints (smooth curvature along rod)."""
            for rod_idx, vert_idx in ti.ndrange(n_rods, n_verts - 2):
                if (valid_mask[rod_idx, vert_idx] == 0 or
                    valid_mask[rod_idx, vert_idx + 1] == 0 or
                    valid_mask[rod_idx, vert_idx + 2] == 0):
                    continue

                p0 = positions[rod_idx, vert_idx]
                p1 = positions[rod_idx, vert_idx + 1]
                p2 = positions[rod_idx, vert_idx + 2]

                # Compute current angle at p1
                e0 = p1 - p0
                e1 = p2 - p1

                len0 = e0.norm()
                len1 = e1.norm()

                if len0 < 1e-10 or len1 < 1e-10:
                    continue

                e0_n = e0 / len0
                e1_n = e1 / len1

                # Dot product gives cos(angle)
                dot = e0_n.dot(e1_n)
                dot = ti.max(ti.min(dot, ti.cast(0.999, ti.f64)), ti.cast(-0.999, ti.f64))

                # Target: straight rod (dot = 1, angle = 0)
                # For rest curvature, we'd use rest quaternions
                rest_dot = ti.cast(1.0, ti.f64)  # Straight rod default

                # Constraint: C = dot - rest_dot (want to minimize bending)
                C = rest_dot - dot  # Positive when bent

                if ti.abs(C) < 1e-6:
                    continue

                # Simple approach: push middle vertex toward line between p0 and p2
                midpoint = (p0 + p2) * 0.5
                correction = midpoint - p1

                w1 = ti.cast(1.0, ti.f64) if fixed_mask[rod_idx, vert_idx + 1] == 0 else ti.cast(0.0, ti.f64)

                if w1 > 0:
                    # Apply small correction toward straight
                    alpha = ti.cast(0.1, ti.f64) * (ti.cast(1.0, ti.f64) - compliance)
                    positions[rod_idx, vert_idx + 1] = p1 + correction * alpha * w1

        @ti.kernel
        def update_velocities(dt: ti.f64, old_positions: ti.template()):
            """Update velocities from position changes."""
            for rod_idx, vert_idx in ti.ndrange(n_rods, n_verts):
                if valid_mask[rod_idx, vert_idx] == 0:
                    continue
                if fixed_mask[rod_idx, vert_idx] == 1:
                    velocities[rod_idx, vert_idx] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
                else:
                    old_p = old_positions[rod_idx, vert_idx]
                    new_p = positions[rod_idx, vert_idx]
                    velocities[rod_idx, vert_idx] = (new_p - old_p) / dt

        # Store kernels
        self._kernel_predict = predict_positions
        self._kernel_stretch = solve_stretch_constraints
        self._kernel_volume = solve_volume_constraints
        self._kernel_bend = solve_bend_constraints
        self._kernel_update_vel = update_velocities

        # Temporary storage for old positions
        self.ti_old_positions = ti.Vector.field(3, dtype=ti.f64, shape=(n_rods, n_verts))

    def set_fixed_targets(self, rod_idx, vert_idx, target):
        """Set target position for a fixed vertex."""
        if rod_idx < self.num_rods and vert_idx < self.num_vertices_per_rod:
            self.fixed_targets[rod_idx, vert_idx] = target
            if self.use_gpu and TAICHI_AVAILABLE:
                # Update single value in Taichi field
                self.ti_fixed_targets[rod_idx, vert_idx] = target

    def set_endpoint_targets(self, origin_targets, insertion_targets):
        """
        Set target positions for rod endpoints from skeleton.

        Args:
            origin_targets: List of (rod_idx, position) for origin endpoints
            insertion_targets: List of (rod_idx, position) for insertion endpoints
        """
        for rod_idx, pos in origin_targets:
            if rod_idx < self.num_rods:
                self.set_fixed_targets(rod_idx, 0, pos)

        for rod_idx, pos in insertion_targets:
            if rod_idx < self.num_rods:
                # Find last valid vertex
                last_valid = np.where(self.valid_mask[rod_idx])[0]
                if len(last_valid) > 0:
                    last_idx = last_valid[-1]
                    self.set_fixed_targets(rod_idx, last_idx, pos)

    def set_collision_meshes(self, trimeshes):
        """
        Set collision meshes (skeleton bones).

        Args:
            trimeshes: List of trimesh objects or (vertices, faces) tuples
        """
        self.collision_meshes = []
        for mesh in trimeshes:
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                self.collision_meshes.append((
                    np.array(mesh.vertices, dtype=np.float64),
                    np.array(mesh.faces, dtype=np.int32)
                ))
            elif isinstance(mesh, tuple) and len(mesh) == 2:
                self.collision_meshes.append((
                    np.array(mesh[0], dtype=np.float64),
                    np.array(mesh[1], dtype=np.int32)
                ))

    def _solve_collision_numpy(self):
        """Solve collision constraints (CPU version)."""
        if not self.collision_meshes or not self.enable_collision:
            return

        margin = self.collision_margin

        for rod_idx in range(self.num_rods):
            for vert_idx in range(self.num_vertices_per_rod):
                if not self.valid_mask[rod_idx, vert_idx]:
                    continue
                if self.fixed_mask[rod_idx, vert_idx]:
                    continue

                pos = self.positions[rod_idx, vert_idx]
                scale = self.scales[rod_idx, vert_idx] if self.scales is not None else 1.0
                radius = margin * scale

                # Check collision with each mesh
                for verts, faces in self.collision_meshes:
                    # Simple point-in-mesh test and push out
                    # For now, just use closest point on mesh surface
                    min_dist = float('inf')
                    closest_point = None
                    closest_normal = None

                    for face in faces:
                        # Triangle vertices
                        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]

                        # Closest point on triangle
                        cp, dist, normal = self._closest_point_on_triangle(pos, v0, v1, v2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = cp
                            closest_normal = normal

                    # If penetrating, push out
                    if closest_point is not None and min_dist < radius:
                        push_dir = pos - closest_point
                        push_len = np.linalg.norm(push_dir)
                        if push_len > 1e-10:
                            push_dir = push_dir / push_len
                        else:
                            push_dir = closest_normal if closest_normal is not None else np.array([0, 1, 0])

                        correction = push_dir * (radius - min_dist) * self.collision_stiffness
                        self.positions[rod_idx, vert_idx] = pos + correction

    def _closest_point_on_triangle(self, p, v0, v1, v2):
        """Find closest point on triangle to point p. Returns (point, distance, normal)."""
        # Edge vectors
        e0 = v1 - v0
        e1 = v2 - v0
        v0p = p - v0

        # Compute barycentric coordinates
        d00 = np.dot(e0, e0)
        d01 = np.dot(e0, e1)
        d11 = np.dot(e1, e1)
        d20 = np.dot(v0p, e0)
        d21 = np.dot(v0p, e1)

        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            # Degenerate triangle
            return v0, np.linalg.norm(p - v0), np.array([0, 1, 0])

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        # Clamp to triangle
        if u < 0:
            # Closest to edge v1-v2
            t = np.clip(np.dot(p - v1, v2 - v1) / max(np.dot(v2 - v1, v2 - v1), 1e-10), 0, 1)
            closest = v1 + t * (v2 - v1)
        elif v < 0:
            # Closest to edge v0-v2
            t = np.clip(np.dot(p - v0, v2 - v0) / max(np.dot(v2 - v0, v2 - v0), 1e-10), 0, 1)
            closest = v0 + t * (v2 - v0)
        elif w < 0:
            # Closest to edge v0-v1
            t = np.clip(np.dot(p - v0, v1 - v0) / max(np.dot(v1 - v0, v1 - v0), 1e-10), 0, 1)
            closest = v0 + t * (v1 - v0)
        else:
            # Inside triangle
            closest = u * v0 + v * v1 + w * v2

        # Normal
        normal = np.cross(e0, e1)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-10:
            normal = normal / norm_len
        else:
            normal = np.array([0, 1, 0])

        return closest, np.linalg.norm(p - closest), normal

    def step(self, dt=None):
        """
        Advance simulation by one timestep using PBD.

        Args:
            dt: Timestep (uses self.dt if None)

        Returns:
            Maximum displacement (for convergence check)
        """
        if not self.initialized:
            return 0.0

        if dt is None:
            dt = self.dt

        if self.use_gpu and TAICHI_AVAILABLE:
            return self._step_taichi(dt)
        else:
            return self._step_numpy(dt)

    def _step_taichi(self, dt):
        """PBD step using Taichi kernels."""
        # Store old positions for velocity update and max displacement calc
        old_positions = self.positions.copy()
        self._copy_positions_kernel()

        # 1. Predict positions (apply gravity, damping)
        self._kernel_predict(dt, self.damping,
                            self.gravity[0], self.gravity[1], self.gravity[2])

        # 2. Solve constraints iteratively
        stretch_compliance = 1.0 - self.stretch_stiffness
        volume_compliance = 1.0 - self.volume_stiffness
        bend_compliance = 1.0 - self.bend_stiffness

        for _ in range(self.iterations):
            if self.enable_stretch_constraint:
                self._kernel_stretch(stretch_compliance)
            if self.enable_volume_constraint:
                self._kernel_volume(volume_compliance)
            if self.enable_bend_constraint:
                self._kernel_bend(bend_compliance)
            # Collision (uses numpy for now, TODO: Taichi kernel)
            if self.enable_collision:
                self._sync_positions_from_taichi()
                self._solve_collision_numpy()
                self._sync_positions_to_taichi()

        # 3. Update velocities
        self._kernel_update_vel(dt, self.ti_old_positions)

        # Sync back to numpy
        self._sync_from_taichi()

        # Calculate max displacement
        displacement = np.linalg.norm(self.positions - old_positions, axis=2)
        max_disp = np.max(displacement[self.valid_mask])

        return max_disp

    @ti.kernel
    def _copy_positions_kernel(self):
        """Copy current positions to old positions."""
        for rod_idx, vert_idx in ti.ndrange(self.num_rods, self.num_vertices_per_rod):
            self.ti_old_positions[rod_idx, vert_idx] = self.ti_positions[rod_idx, vert_idx]

    def _step_numpy(self, dt):
        """PBD step using NumPy (CPU fallback)."""
        # Store old positions
        old_positions = self.positions.copy()

        # 1. Predict positions
        for rod_idx in range(self.num_rods):
            for vert_idx in range(self.num_vertices_per_rod):
                if not self.valid_mask[rod_idx, vert_idx]:
                    continue
                if self.fixed_mask[rod_idx, vert_idx]:
                    self.positions[rod_idx, vert_idx] = self.fixed_targets[rod_idx, vert_idx]
                    self.velocities[rod_idx, vert_idx] = 0
                else:
                    # Apply gravity and damping
                    self.velocities[rod_idx, vert_idx] *= self.damping
                    self.velocities[rod_idx, vert_idx] += self.gravity * dt
                    self.positions[rod_idx, vert_idx] += self.velocities[rod_idx, vert_idx] * dt

        # 2. Solve constraints
        for _ in range(self.iterations):
            self._solve_stretch_numpy()
            self._solve_volume_numpy()
            self._solve_bend_numpy()
            if self.enable_collision:
                self._solve_collision_numpy()

        # 3. Update velocities
        for rod_idx in range(self.num_rods):
            for vert_idx in range(self.num_vertices_per_rod):
                if not self.valid_mask[rod_idx, vert_idx]:
                    continue
                if not self.fixed_mask[rod_idx, vert_idx]:
                    self.velocities[rod_idx, vert_idx] = (
                        self.positions[rod_idx, vert_idx] - old_positions[rod_idx, vert_idx]
                    ) / dt

        # Calculate max displacement
        displacement = np.linalg.norm(self.positions - old_positions, axis=2)
        max_disp = np.max(displacement[self.valid_mask])

        return max_disp

    def _solve_stretch_numpy(self):
        """Solve stretch constraints (NumPy version)."""
        compliance = 1.0 - self.stretch_stiffness

        for rod_idx in range(self.num_rods):
            for seg_idx in range(self.num_vertices_per_rod - 1):
                if not (self.valid_mask[rod_idx, seg_idx] and
                        self.valid_mask[rod_idx, seg_idx + 1]):
                    continue

                p0 = self.positions[rod_idx, seg_idx]
                p1 = self.positions[rod_idx, seg_idx + 1]
                s0 = self.scales[rod_idx, seg_idx]
                s1 = self.scales[rod_idx, seg_idx + 1]

                diff = p1 - p0
                current_length = np.linalg.norm(diff)

                if current_length < 1e-10:
                    continue

                avg_scale = (s0 + s1) * 0.5
                target_length = self.rest_lengths[rod_idx, seg_idx] * avg_scale

                C = current_length - target_length

                w0 = 1.0 if not self.fixed_mask[rod_idx, seg_idx] else 0.0
                w1 = 1.0 if not self.fixed_mask[rod_idx, seg_idx + 1] else 0.0

                w_sum = w0 + w1 + compliance
                if w_sum > 1e-10:
                    delta_lambda = -C / w_sum
                    correction = diff / current_length * delta_lambda

                    if w0 > 0:
                        self.positions[rod_idx, seg_idx] -= correction * w0
                    if w1 > 0:
                        self.positions[rod_idx, seg_idx + 1] += correction * w1

    def _solve_volume_numpy(self):
        """Solve volume constraints (NumPy version) - VIPER key feature."""
        compliance = 1.0 - self.volume_stiffness

        for rod_idx in range(self.num_rods):
            for seg_idx in range(self.num_vertices_per_rod - 1):
                if not (self.valid_mask[rod_idx, seg_idx] and
                        self.valid_mask[rod_idx, seg_idx + 1]):
                    continue

                p0 = self.positions[rod_idx, seg_idx]
                p1 = self.positions[rod_idx, seg_idx + 1]
                s0 = self.scales[rod_idx, seg_idx]
                s1 = self.scales[rod_idx, seg_idx + 1]

                diff = p1 - p0
                current_length = np.linalg.norm(diff)

                if current_length < 1e-10:
                    continue

                rest_length = self.rest_lengths[rod_idx, seg_idx]

                # Volume proportional to scale^2 * length
                avg_scale = (s0 + s1) * 0.5
                current_volume = avg_scale * avg_scale * current_length
                rest_vol_normalized = rest_length

                C = current_volume - rest_vol_normalized

                grad_scale = 2.0 * avg_scale * current_length

                if abs(grad_scale) < 1e-10:
                    continue

                w0 = 1.0 if not self.fixed_mask[rod_idx, seg_idx] else 0.0
                w1 = 1.0 if not self.fixed_mask[rod_idx, seg_idx + 1] else 0.0

                denom = grad_scale * grad_scale * (w0 + w1) + compliance
                if denom > 1e-10:
                    delta_lambda = -C / denom
                    delta_scale = grad_scale * delta_lambda * 0.5
                    delta_scale = np.clip(delta_scale, -0.1, 0.1)

                    if w0 > 0:
                        self.scales[rod_idx, seg_idx] = max(s0 + delta_scale, 0.3)
                    if w1 > 0:
                        self.scales[rod_idx, seg_idx + 1] = max(s1 + delta_scale, 0.3)

    def _solve_bend_numpy(self):
        """Solve bending constraints (NumPy version)."""
        compliance = 1.0 - self.bend_stiffness

        for rod_idx in range(self.num_rods):
            for vert_idx in range(self.num_vertices_per_rod - 2):
                if not (self.valid_mask[rod_idx, vert_idx] and
                        self.valid_mask[rod_idx, vert_idx + 1] and
                        self.valid_mask[rod_idx, vert_idx + 2]):
                    continue

                p0 = self.positions[rod_idx, vert_idx]
                p1 = self.positions[rod_idx, vert_idx + 1]
                p2 = self.positions[rod_idx, vert_idx + 2]

                # Push middle vertex toward line
                midpoint = (p0 + p2) * 0.5
                correction = midpoint - p1

                if not self.fixed_mask[rod_idx, vert_idx + 1]:
                    alpha = 0.1 * (1.0 - compliance)
                    self.positions[rod_idx, vert_idx + 1] += correction * alpha

    def get_positions_flat(self):
        """Get all rod positions as flat array (N, 3)."""
        if not self.initialized:
            return None

        valid_positions = []
        for rod_idx in range(self.num_rods):
            for vert_idx in range(self.num_vertices_per_rod):
                if self.valid_mask[rod_idx, vert_idx]:
                    valid_positions.append(self.positions[rod_idx, vert_idx])

        return np.array(valid_positions) if valid_positions else None

    def get_scales_flat(self):
        """Get all scales as flat array."""
        if not self.initialized:
            return None

        valid_scales = []
        for rod_idx in range(self.num_rods):
            for vert_idx in range(self.num_vertices_per_rod):
                if self.valid_mask[rod_idx, vert_idx]:
                    valid_scales.append(self.scales[rod_idx, vert_idx])

        return np.array(valid_scales) if valid_scales else None

    def compute_skinning_weights(self, mesh_vertices, k_neighbors=4):
        """
        Compute skinning weights from rod vertices to mesh vertices.
        Uses inverse distance weighting to nearest rod vertices.

        Args:
            mesh_vertices: (N, 3) array of mesh vertex positions
            k_neighbors: Number of nearest rod vertices to use

        Returns:
            skinning_weights: (N, k_neighbors) weights
            skinning_indices: (N, k_neighbors) rod vertex indices
        """
        from scipy.spatial import cKDTree

        # Get all rod vertices
        rod_positions = self.get_positions_flat()
        if rod_positions is None:
            return None, None

        # Build KD-tree
        tree = cKDTree(rod_positions)

        # Find k nearest neighbors for each mesh vertex
        distances, indices = tree.query(mesh_vertices, k=k_neighbors)

        # Compute inverse distance weights
        weights = np.zeros_like(distances)
        for i in range(len(mesh_vertices)):
            dists = distances[i]
            # Avoid division by zero
            dists = np.maximum(dists, 1e-6)
            inv_dists = 1.0 / dists
            weights[i] = inv_dists / inv_dists.sum()

        self.skinning_weights = weights
        self.skinning_indices = indices

        return weights, indices

    def skin_mesh(self, rest_mesh_vertices):
        """
        Deform mesh vertices using rod skinning.

        Args:
            rest_mesh_vertices: (N, 3) rest pose mesh vertices

        Returns:
            deformed_vertices: (N, 3) deformed mesh vertices
        """
        if self.skinning_weights is None or self.skinning_indices is None:
            self.compute_skinning_weights(rest_mesh_vertices)

        # Get current rod positions and scales
        rod_positions = self.get_positions_flat()
        rod_scales = self.get_scales_flat()

        if rod_positions is None:
            return rest_mesh_vertices

        # Get rest rod positions
        rest_rod_positions = []
        for rod_idx in range(self.num_rods):
            for vert_idx in range(self.num_vertices_per_rod):
                if self.valid_mask[rod_idx, vert_idx]:
                    rest_rod_positions.append(self.rest_positions[rod_idx, vert_idx])
        rest_rod_positions = np.array(rest_rod_positions)

        # Deform each mesh vertex
        deformed = np.zeros_like(rest_mesh_vertices)

        for i in range(len(rest_mesh_vertices)):
            mesh_rest = rest_mesh_vertices[i]
            weights = self.skinning_weights[i]
            indices = self.skinning_indices[i]

            # Weighted blend of deformations
            deformed_pos = np.zeros(3)
            for j, (idx, w) in enumerate(zip(indices, weights)):
                # Compute deformation from rod vertex
                rod_rest = rest_rod_positions[idx]
                rod_curr = rod_positions[idx]
                rod_scale = rod_scales[idx]

                # Offset from rod vertex in rest pose
                offset = mesh_rest - rod_rest

                # Scale the offset (VIPER volume preservation)
                scaled_offset = offset * rod_scale

                # Apply deformation
                deformed_pos += w * (rod_curr + scaled_offset)

            deformed[i] = deformed_pos

        return deformed

    def reset(self):
        """Reset simulation to rest state."""
        if not self.initialized:
            return

        self.positions = self.rest_positions.copy()
        self.velocities = np.zeros_like(self.velocities)
        self.quaternions = self.rest_quaternions.copy()
        self.scales = np.ones_like(self.scales)
        self.scale_velocities = np.zeros_like(self.scale_velocities)

        if self.use_gpu and TAICHI_AVAILABLE:
            self._sync_to_taichi()

    def generate_rod_mesh(self, rest_radii=None, slices=8):
        """
        Generate a triangle mesh from rod positions and scales.

        Creates a tube mesh around each rod, where the radius at each vertex
        is rest_radius * scale (volume preservation).

        Args:
            rest_radii: List of arrays, one per rod, containing rest radius per vertex.
                       If None, uses default radius of 0.002.
            slices: Number of subdivisions around each tube (default 8)

        Returns:
            tuple: (vertices, faces, normals) as numpy arrays
                   vertices: (N, 3) float32
                   faces: (M, 3) int32
                   normals: (N, 3) float32
        """
        if not self.initialized:
            return None, None, None

        all_vertices = []
        all_faces = []
        all_normals = []
        vertex_offset = 0

        for rod_idx in range(self.num_rods):
            # Get valid vertices for this rod
            valid_count = np.sum(self.valid_mask[rod_idx])
            if valid_count < 2:
                continue

            # Get rod data
            rod_positions = self.positions[rod_idx][self.valid_mask[rod_idx]]
            rod_scales = self.scales[rod_idx][self.valid_mask[rod_idx]]

            # Get rest radii for this rod
            if rest_radii is not None and rod_idx < len(rest_radii):
                rod_rest_radii = rest_radii[rod_idx]
                if len(rod_rest_radii) < len(rod_positions):
                    # Pad with last value
                    rod_rest_radii = np.concatenate([
                        rod_rest_radii,
                        np.full(len(rod_positions) - len(rod_rest_radii), rod_rest_radii[-1] if len(rod_rest_radii) > 0 else 0.002)
                    ])
            else:
                rod_rest_radii = np.full(len(rod_positions), 0.002)

            # Compute radius at each vertex: rest_radius * scale
            rod_radii = rod_rest_radii[:len(rod_positions)] * rod_scales

            # Generate tube mesh for this rod
            rod_verts, rod_faces, rod_normals = self._generate_tube_mesh(
                rod_positions, rod_radii, slices
            )

            if rod_verts is not None and len(rod_verts) > 0:
                all_vertices.append(rod_verts)
                all_faces.append(rod_faces + vertex_offset)
                all_normals.append(rod_normals)
                vertex_offset += len(rod_verts)

        if not all_vertices:
            return None, None, None

        vertices = np.vstack(all_vertices).astype(np.float32)
        faces = np.vstack(all_faces).astype(np.int32)
        normals = np.vstack(all_normals).astype(np.float32)

        return vertices, faces, normals

    def _generate_tube_mesh(self, positions, radii, slices=8):
        """
        Generate a tube mesh along a polyline with varying radii.

        Args:
            positions: (N, 3) array of positions along the tube centerline
            radii: (N,) array of radii at each position
            slices: Number of subdivisions around the tube

        Returns:
            tuple: (vertices, faces, normals)
        """
        n_points = len(positions)
        if n_points < 2:
            return None, None, None

        # Pre-compute tangent frames along the curve (twist-minimizing)
        tangents = []
        for i in range(n_points):
            if i == 0:
                t = positions[1] - positions[0]
            elif i == n_points - 1:
                t = positions[-1] - positions[-2]
            else:
                t = positions[i + 1] - positions[i - 1]
            t_len = np.linalg.norm(t)
            if t_len > 1e-8:
                t = t / t_len
            else:
                t = np.array([0, 0, 1])
            tangents.append(t)

        # Build perpendicular frames with twist minimization
        perps = []
        perp2s = []

        # First frame
        t = tangents[0]
        if abs(t[1]) < 0.9:
            perp = np.cross(t, np.array([0, 1, 0]))
        else:
            perp = np.cross(t, np.array([1, 0, 0]))
        perp = perp / np.linalg.norm(perp)
        perp2 = np.cross(t, perp)
        perps.append(perp)
        perp2s.append(perp2)

        # Propagate frame with twist minimization
        for i in range(1, n_points):
            t = tangents[i]
            prev_perp = perps[-1]

            # Project previous perp onto plane perpendicular to new tangent
            perp = prev_perp - np.dot(prev_perp, t) * t
            perp_len = np.linalg.norm(perp)
            if perp_len > 1e-8:
                perp = perp / perp_len
            else:
                if abs(t[1]) < 0.9:
                    perp = np.cross(t, np.array([0, 1, 0]))
                else:
                    perp = np.cross(t, np.array([1, 0, 0]))
                perp = perp / np.linalg.norm(perp)

            perp2 = np.cross(t, perp)
            perps.append(perp)
            perp2s.append(perp2)

        # Generate vertices: rings around each centerline point
        vertices = []
        normals = []

        for i in range(n_points):
            p = positions[i]
            r = radii[i]
            perp = perps[i]
            perp2 = perp2s[i]

            for j in range(slices):
                angle = 2.0 * np.pi * j / slices
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                # Normal direction (outward)
                normal = cos_a * perp + sin_a * perp2
                # Vertex position
                vertex = p + r * normal

                vertices.append(vertex)
                normals.append(normal)

        vertices = np.array(vertices)
        normals = np.array(normals)

        # Generate faces: connect adjacent rings
        faces = []
        for i in range(n_points - 1):
            for j in range(slices):
                # Current ring indices
                curr0 = i * slices + j
                curr1 = i * slices + (j + 1) % slices
                # Next ring indices
                next0 = (i + 1) * slices + j
                next1 = (i + 1) * slices + (j + 1) % slices

                # Two triangles per quad
                faces.append([curr0, next0, curr1])
                faces.append([curr1, next0, next1])

        # Add end caps
        # Start cap (ring 0)
        center_start = np.mean(vertices[:slices], axis=0)
        start_cap_idx = len(vertices)
        vertices = np.vstack([vertices, center_start.reshape(1, 3)])
        normals = np.vstack([normals, (-tangents[0]).reshape(1, 3)])
        for j in range(slices):
            j_next = (j + 1) % slices
            faces.append([start_cap_idx, j_next, j])

        # End cap (last ring)
        end_ring_start = (n_points - 1) * slices
        center_end = np.mean(vertices[end_ring_start:end_ring_start + slices], axis=0)
        end_cap_idx = len(vertices)
        vertices = np.vstack([vertices, center_end.reshape(1, 3)])
        normals = np.vstack([normals, tangents[-1].reshape(1, 3)])
        for j in range(slices):
            j_next = (j + 1) % slices
            faces.append([end_cap_idx, end_ring_start + j, end_ring_start + j_next])

        faces = np.array(faces)

        return vertices, faces, normals

    def get_debug_info(self):
        """Get debug information about simulation state."""
        if not self.initialized:
            return {}

        return {
            'num_rods': self.num_rods,
            'num_vertices_per_rod': self.num_vertices_per_rod,
            'total_vertices': self.total_vertices,
            'avg_scale': np.mean(self.scales[self.valid_mask]),
            'min_scale': np.min(self.scales[self.valid_mask]),
            'max_scale': np.max(self.scales[self.valid_mask]),
            'avg_velocity': np.mean(np.linalg.norm(self.velocities, axis=-1)[self.valid_mask]),
            'use_gpu': self.use_gpu
        }

    def get_constraint_residuals(self):
        """
        Compute constraint residuals for debugging convergence.

        Returns:
            dict with constraint error statistics
        """
        if not self.initialized:
            return {}

        stretch_errors = []
        volume_errors = []
        bend_errors = []

        for rod_idx in range(self.num_rods):
            # Stretch constraint residuals
            for seg_idx in range(self.num_vertices_per_rod - 1):
                if not self.valid_mask[rod_idx, seg_idx] or not self.valid_mask[rod_idx, seg_idx + 1]:
                    continue

                p0 = self.positions[rod_idx, seg_idx]
                p1 = self.positions[rod_idx, seg_idx + 1]
                current_length = np.linalg.norm(p1 - p0)
                rest_length = self.rest_lengths[rod_idx, seg_idx]

                if rest_length > 1e-10:
                    stretch_error = abs(current_length - rest_length) / rest_length
                    stretch_errors.append(stretch_error)

                # Volume constraint residuals
                if self.enable_volume_constraint:
                    s0 = self.scales[rod_idx, seg_idx]
                    s1 = self.scales[rod_idx, seg_idx + 1]
                    avg_scale = (s0 + s1) * 0.5
                    current_volume = avg_scale * avg_scale * current_length
                    rest_volume = rest_length  # Normalized rest volume (scale=1)

                    if rest_volume > 1e-10:
                        volume_error = abs(current_volume - rest_volume) / rest_volume
                        volume_errors.append(volume_error)

            # Bend constraint residuals
            for vert_idx in range(self.num_vertices_per_rod - 2):
                if (not self.valid_mask[rod_idx, vert_idx] or
                    not self.valid_mask[rod_idx, vert_idx + 1] or
                    not self.valid_mask[rod_idx, vert_idx + 2]):
                    continue

                p0 = self.positions[rod_idx, vert_idx]
                p1 = self.positions[rod_idx, vert_idx + 1]
                p2 = self.positions[rod_idx, vert_idx + 2]

                v1 = p1 - p0
                v2 = p2 - p1
                l1 = np.linalg.norm(v1)
                l2 = np.linalg.norm(v2)

                if l1 > 1e-10 and l2 > 1e-10:
                    cos_angle = np.dot(v1, v2) / (l1 * l2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    bend_error = abs(np.pi - angle)  # Deviation from straight
                    bend_errors.append(bend_error)

        return {
            'stretch_error_mean': np.mean(stretch_errors) if stretch_errors else 0.0,
            'stretch_error_max': np.max(stretch_errors) if stretch_errors else 0.0,
            'volume_error_mean': np.mean(volume_errors) if volume_errors else 0.0,
            'volume_error_max': np.max(volume_errors) if volume_errors else 0.0,
            'bend_error_mean': np.mean(bend_errors) if bend_errors else 0.0,
            'bend_error_max': np.max(bend_errors) if bend_errors else 0.0,
        }


# Convenience function to create VIPER simulation from MeshLoader waypoints
def create_viper_from_mesh_loader(mesh_loader, use_gpu=True):
    """
    Create VIPER rod simulation from MeshLoader waypoints.

    Args:
        mesh_loader: MeshLoader instance with waypoints
        use_gpu: Use GPU acceleration

    Returns:
        ViperRodSimulation instance or None if no waypoints
    """
    if not hasattr(mesh_loader, 'waypoints') or not mesh_loader.waypoints:
        print("VIPER: No waypoints found in mesh_loader")
        return None

    viper = ViperRodSimulation(use_gpu=use_gpu)

    # Convert waypoints to rod format
    # Each waypoint group becomes a rod
    if viper.build_from_waypoints(mesh_loader.waypoints):
        return viper

    return None
