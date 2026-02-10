# Muscle Mesh functionality for the viewer
# Contains SoftBodySimulation class and MuscleMeshMixin for MeshLoader

import numpy as np
from OpenGL.GL import *
import scipy.sparse
import scipy.sparse.linalg
from collections import defaultdict
import matplotlib.cm as cm

from .viper_rods import ViperRodSimulation

# Scalar field constants
SCALAR_FIELD_KMIN = 1.0
SCALAR_FIELD_KMAX = 10.0
COLOR_MAP = cm.get_cmap("turbo")


# ============================================================================
# SCALAR FIELD UTILITY FUNCTIONS
# ============================================================================

def cotangent_weight_matrix(vertices, faces):
    """
    Compute the cotangent weight matrix using vectorized numpy operations.
    Much faster than the loop-based version.
    """
    n = len(vertices)

    # Extract face vertex indices (handle Nx3x1 or Nx3 format)
    if faces.ndim == 3:
        face_indices = faces[:, :, 0].astype(int)
    else:
        face_indices = faces.astype(int)

    # Get vertex positions for all faces (F x 3)
    v0 = vertices[face_indices[:, 0]]
    v1 = vertices[face_indices[:, 1]]
    v2 = vertices[face_indices[:, 2]]

    # Edge vectors
    e0 = v2 - v1  # edge opposite to v0
    e1 = v0 - v2  # edge opposite to v1
    e2 = v1 - v0  # edge opposite to v2

    # Compute cotangent: cot = cos/sin = dot / |cross|
    def compute_cot(edge_a, edge_b):
        dot = np.sum(edge_a * edge_b, axis=1)
        cross_norm = np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
        return dot / (cross_norm + 1e-10)

    # Cotangent at each vertex for opposite edge
    cot0 = compute_cot(-e2, e1)  # angle at v0, for edge (v1, v2)
    cot1 = compute_cot(-e0, e2)  # angle at v1, for edge (v2, v0)
    cot2 = compute_cot(-e1, e0)  # angle at v2, for edge (v0, v1)

    # Build sparse matrix indices and values
    # Each edge gets 0.5 * cot contribution from each adjacent face
    i_indices = np.concatenate([
        face_indices[:, 1], face_indices[:, 2],  # edge (v1, v2)
        face_indices[:, 2], face_indices[:, 0],  # edge (v2, v0)
        face_indices[:, 0], face_indices[:, 1],  # edge (v0, v1)
    ])
    j_indices = np.concatenate([
        face_indices[:, 2], face_indices[:, 1],  # edge (v1, v2)
        face_indices[:, 0], face_indices[:, 2],  # edge (v2, v0)
        face_indices[:, 1], face_indices[:, 0],  # edge (v0, v1)
    ])
    values = np.concatenate([
        0.5 * cot0, 0.5 * cot0,
        0.5 * cot1, 0.5 * cot1,
        0.5 * cot2, 0.5 * cot2,
    ])

    # Create sparse matrix (duplicate entries are automatically summed)
    W = scipy.sparse.csr_matrix((values, (i_indices, j_indices)), shape=(n, n))

    return W


def solve_scalar_field(vertices, faces, origin_indices, insertion_indices, kmin=1.0, kmax=10.0):
    """
    Solve for the scalar field satisfying the Laplace equation with given constraints.
    """
    n = len(vertices)
    W = cotangent_weight_matrix(vertices, faces)
    L = -scipy.sparse.diags(W.sum(axis=1).A1) + W

    b = np.zeros(n)
    boundary_mask = np.zeros(n, dtype=bool)

    b[origin_indices] = kmin
    b[insertion_indices] = kmax
    boundary_mask[origin_indices] = True
    boundary_mask[insertion_indices] = True

    free_vertices = ~boundary_mask
    A = L[free_vertices][:, free_vertices]
    b_free = -L[free_vertices][:, boundary_mask] @ b[boundary_mask]

    regularization = 1e-8
    A = A + regularization * scipy.sparse.eye(A.shape[0])

    u_free = scipy.sparse.linalg.spsolve(A, b_free)

    u = np.zeros(n)
    u[free_vertices] = u_free
    u[boundary_mask] = b[boundary_mask]

    return u


# ============================================================================
# BOUNDING PLANE UTILITY FUNCTIONS
# ============================================================================

def compute_bounding_plane(vertices_2d):
    """Finds the bounding box of 2D projected vertices."""
    min_x, max_x = np.min(vertices_2d[:, 0]), np.max(vertices_2d[:, 0])
    min_y, max_y = np.min(vertices_2d[:, 1]), np.max(vertices_2d[:, 1])

    return np.array([
        [min_x, min_y], [max_x, min_y],
        [max_x, max_y], [min_x, max_y]
    ])


class SoftBodySimulation:
    """
    Vectorized quasistatic soft body simulation for tetrahedralized muscle meshes.
    Solves for equilibrium positions given fixed boundary constraints.
    Uses edge springs and volume preservation without dynamics/gravity.
    """

    def __init__(self, vertices, tetrahedra, fixed_vertices=None,
                 stiffness=1.0, damping=0.5, volume_stiffness=0.5):
        """
        Initialize quasistatic soft body simulation.

        Args:
            vertices: Nx3 array of vertex positions
            tetrahedra: Mx4 array of tetrahedron vertex indices
            fixed_vertices: List of vertex indices that are fixed (caps)
            stiffness: Spring stiffness (0-1, relative weight)
            damping: Position update damping (0-1, higher = slower convergence)
            volume_stiffness: Volume preservation weight (0-1)
        """
        self.num_vertices = len(vertices)
        self.positions = vertices.astype(np.float64).copy()
        self.rest_positions = vertices.astype(np.float64).copy()
        self.tetrahedra = np.array(tetrahedra, dtype=np.int32)

        # Fixed vertices mask (vectorized)
        self.fixed_mask = np.zeros(self.num_vertices, dtype=bool)
        self.free_mask = np.ones(self.num_vertices, dtype=bool)
        if fixed_vertices is not None:
            fixed_vertices = np.array(fixed_vertices, dtype=np.int32)
            valid = fixed_vertices < self.num_vertices
            self.fixed_mask[fixed_vertices[valid]] = True
            self.free_mask[fixed_vertices[valid]] = False
        self.fixed_indices = np.where(self.fixed_mask)[0]
        self.free_indices = np.where(self.free_mask)[0]

        # Simulation parameters (normalized 0-1)
        # For muscle: high volume stiffness, low edge stiffness
        self.stiffness = np.clip(stiffness, 0.0, 1.0)
        self.damping = np.clip(damping, 0.0, 0.99)
        self.volume_stiffness = np.clip(volume_stiffness, 0.0, 1.0)  # No reduction for muscle

        # Build edge arrays (vectorized)
        self._build_edges()

        # Compute rest state
        self._compute_rest_state()

        # Compute characteristic length scale for clamping
        bbox = np.ptp(self.rest_positions, axis=0)
        self.char_length = np.max(bbox) * 0.1  # 10% of bounding box

        # Fixed vertex target positions
        self.fixed_targets = self.positions[self.fixed_indices].copy() if len(self.fixed_indices) > 0 else None
        self.initial_fixed_targets = self.fixed_targets.copy() if self.fixed_targets is not None else None

        # Bone proximity soft constraints (for vertices near skeleton)
        # One-sided repulsion: only push away when too close, doesn't pull
        self.proximity_indices = None  # Vertex indices with proximity constraints
        self.proximity_targets = None  # Bone surface positions (updated each step)
        self.proximity_weights = None  # Constraint weights (stronger = closer to bone initially)
        self.proximity_rest_distances = None  # Initial distances to bone (minimum allowed)
        self.proximity_stiffness = 0.5  # Overall proximity constraint strength

        # Collision meshes (skeleton trimeshes for collision detection)
        self.collision_meshes = None  # List of trimesh objects
        self.collision_margin = 0.002  # Collision margin (2mm)

    def _build_edges(self):
        """Build unique edge arrays from tetrahedra (vectorized)."""
        # Each tet has 6 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        tet_edge_pairs = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])
        all_edges = self.tetrahedra[:, tet_edge_pairs].reshape(-1, 2)
        # Sort each edge and get unique
        all_edges = np.sort(all_edges, axis=1)
        self.edges = np.unique(all_edges, axis=0)
        self.edge_i = self.edges[:, 0]
        self.edge_j = self.edges[:, 1]

    def _compute_rest_state(self):
        """Compute rest lengths and volumes (vectorized)."""
        # Rest edge lengths
        diff = self.rest_positions[self.edge_j] - self.rest_positions[self.edge_i]
        self.rest_lengths = np.linalg.norm(diff, axis=1)

        # Rest volumes for tetrahedra (vectorized)
        v0 = self.rest_positions[self.tetrahedra[:, 0]]
        v1 = self.rest_positions[self.tetrahedra[:, 1]]
        v2 = self.rest_positions[self.tetrahedra[:, 2]]
        v3 = self.rest_positions[self.tetrahedra[:, 3]]
        self.rest_volumes = np.einsum('ij,ij->i', v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0

    def set_fixed_targets(self, targets):
        """Set target positions for fixed vertices."""
        if targets is not None and len(targets) == len(self.fixed_indices):
            self.fixed_targets = np.array(targets, dtype=np.float64)

    def set_proximity_constraints(self, indices, weights, rest_distances, stiffness=0.5):
        """
        Set bone proximity soft constraints (one-sided repulsion).

        Args:
            indices: Array of vertex indices with proximity constraints
            weights: Array of weights for each vertex (higher = stronger constraint)
            rest_distances: Array of initial distances to bone (minimum allowed distance)
            stiffness: Overall proximity constraint strength (0-1)
        """
        if indices is not None and len(indices) > 0:
            self.proximity_indices = np.array(indices, dtype=np.int32)
            self.proximity_weights = np.array(weights, dtype=np.float64)
            self.proximity_rest_distances = np.array(rest_distances, dtype=np.float64)
            self.proximity_targets = self.positions[self.proximity_indices].copy()
            self.proximity_stiffness = np.clip(stiffness, 0.0, 1.0)
        else:
            self.proximity_indices = None
            self.proximity_weights = None
            self.proximity_rest_distances = None
            self.proximity_targets = None

    def update_proximity_targets(self, targets):
        """Update target positions for proximity constraints (called each frame)."""
        if targets is not None and self.proximity_indices is not None:
            if len(targets) == len(self.proximity_indices):
                self.proximity_targets = np.array(targets, dtype=np.float64)

    def update_parameters(self, stiffness=None, damping=None, volume_stiffness=None, proximity_stiffness=None):
        """Update simulation parameters (values in 0-1 range)."""
        if stiffness is not None:
            self.stiffness = np.clip(stiffness, 0.0, 1.0)
        if proximity_stiffness is not None:
            self.proximity_stiffness = np.clip(proximity_stiffness, 0.0, 1.0)
        if damping is not None:
            self.damping = np.clip(damping, 0.0, 0.99)
        if volume_stiffness is not None:
            self.volume_stiffness = np.clip(volume_stiffness, 0.0, 1.0)

    def set_collision_meshes(self, trimeshes, margin=0.002):
        """Set skeleton trimeshes for collision detection."""
        self.collision_meshes = trimeshes if trimeshes else None
        self.collision_margin = margin

    def resolve_collisions(self, max_iterations=10):
        """
        Collision resolution: push vertices that are INSIDE bone meshes.
        Only pushes when vertex is on the wrong side of the surface (penetrating).

        Returns: number of vertices that were pushed out
        """
        if self.collision_meshes is None or len(self.collision_meshes) == 0:
            return 0

        total_pushed = 0
        free_positions = self.positions[self.free_indices].copy()
        n_free = len(free_positions)

        for iteration in range(max_iterations):
            # Find minimum distance to any bone for each vertex
            min_dist = np.full(n_free, np.inf)
            closest_pts = np.zeros_like(free_positions)
            closest_normals = np.zeros_like(free_positions)

            for mesh in self.collision_meshes:
                if mesh is None:
                    continue
                try:
                    closest_points, distances, face_ids = mesh.nearest.on_surface(free_positions)
                    normals = mesh.face_normals[face_ids]

                    # Update where this mesh is closer
                    closer = distances < min_dist
                    min_dist[closer] = distances[closer]
                    closest_pts[closer] = closest_points[closer]
                    closest_normals[closer] = normals[closer]
                except Exception:
                    continue

            # Check which vertices are actually INSIDE (penetrating)
            # A vertex is inside if the vector from surface to vertex points
            # OPPOSITE to the face normal (negative dot product)
            to_vertex = free_positions - closest_pts
            dot_products = np.einsum('ij,ij->i', to_vertex, closest_normals)

            # Only push vertices that are:
            # 1. Close to surface (within margin) AND
            # 2. On the wrong side (inside the bone) - negative dot product
            penetrating = (min_dist < self.collision_margin) & (dot_products < 0)
            push_count = np.sum(penetrating)

            if iteration == 0 and push_count > 0:
                print(f"  Collision: {push_count} vertices penetrating bone")

            if push_count == 0:
                break

            # Push penetrating vertices to outside surface
            for idx in np.where(penetrating)[0]:
                # Push along face normal to get outside
                push_dir = closest_normals[idx]
                # Push to margin distance outside the surface
                free_positions[idx] = closest_pts[idx] + push_dir * self.collision_margin

            total_pushed += push_count

        # Update actual positions
        self.positions[self.free_indices] = free_positions

        if total_pushed > 0:
            print(f"  Collision: pushed {total_pushed} vertices out of bone")

        return total_pushed

    def solve_to_convergence(self, max_iterations=500, tolerance=1e-6, muscle_mode=False):
        """
        Solve until convergence or max iterations.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (max displacement)
            muscle_mode: If True, use muscle-specific behavior (contour-based, volume preserving)
                        If False, use pure mass-spring (settle mesh first)

        Returns: (number of iterations, final residual)
        """
        # Apply fixed vertex constraints
        if self.fixed_targets is not None and len(self.fixed_indices) > 0:
            self.positions[self.fixed_indices] = self.fixed_targets

        for iteration in range(max_iterations):
            old_positions = self.positions.copy()
            self._relaxation_step_vectorized(muscle_mode=muscle_mode)

            # Check for NaN/Inf and reset if needed
            if not np.all(np.isfinite(self.positions)):
                print(f"  Warning: NaN/Inf detected at iteration {iteration}, resetting")
                self.positions = old_positions
                return iteration, float('inf')

            # Check convergence (max displacement of free vertices)
            if len(self.free_indices) > 0:
                displacement = np.linalg.norm(self.positions[self.free_indices] - old_positions[self.free_indices], axis=1)
                max_disp = np.max(displacement)
            else:
                max_disp = 0.0

            if max_disp < tolerance:
                return iteration + 1, max_disp

        return max_iterations, max_disp

    def solve_two_phase(self, phase1_iterations=100, phase2_iterations=200, tolerance=1e-6):
        """
        Two-phase solving for muscle soft body simulation.

        Phase 1: Pure mass-spring to settle mesh after skeleton binding
                 (maintains rest lengths, stabilizes mesh quality)
        Phase 2: Muscle-specific behavior (contour-based, volume preserving)
                 (cross-contour edges shrink, intra-contour expand)

        Args:
            phase1_iterations: Max iterations for mass-spring settling
            phase2_iterations: Max iterations for muscle-specific behavior
            tolerance: Convergence tolerance

        Returns: (phase1_iters, phase1_residual, phase2_iters, phase2_residual)
        """
        # Phase 1: Pure mass-spring to settle mesh
        iters1, residual1 = self.solve_to_convergence(
            max_iterations=phase1_iterations,
            tolerance=tolerance,
            muscle_mode=False
        )

        # Phase 2: Muscle-specific behavior
        iters2, residual2 = self.solve_to_convergence(
            max_iterations=phase2_iterations,
            tolerance=tolerance,
            muscle_mode=True
        )

        return iters1, residual1, iters2, residual2

    def solve_two_phase_arap(self, phase1_iterations=50, phase2_iterations=10, tolerance=1e-6,
                              collision_meshes=None, collision_margin=0.002):
        """
        Two-phase solving using ARAP for muscle deformation.

        Phase 1: Mass-spring to settle mesh (same as before)
        Phase 2: ARAP with muscle-specific modifications (shape-preserving)

        Args:
            phase1_iterations: Max iterations for mass-spring settling
            phase2_iterations: Max ARAP iterations for muscle behavior
            tolerance: Convergence tolerance
            collision_meshes: List of trimesh objects for collision detection in ARAP
            collision_margin: Distance to push vertices outside collision meshes

        Returns: (phase1_iters, phase1_residual, phase2_iters, phase2_residual)
        """
        # Phase 1: Pure mass-spring to settle mesh
        iters1, residual1 = self.solve_to_convergence(
            max_iterations=phase1_iterations,
            tolerance=tolerance,
            muscle_mode=False
        )

        # Phase 2: ARAP with muscle modifications and collision
        iters2, residual2 = self.solve_arap_muscle(
            max_iterations=phase2_iterations,
            tolerance=tolerance,
            collision_meshes=collision_meshes,
            collision_margin=collision_margin
        )

        return iters1, residual1, iters2, residual2

    def step(self, iterations=1, muscle_mode=False):
        """
        Run fixed number of relaxation iterations.

        Args:
            iterations: Number of iterations to run
            muscle_mode: If True, use muscle-specific behavior
                        If False, use pure mass-spring
        """
        if self.fixed_targets is not None and len(self.fixed_indices) > 0:
            self.positions[self.fixed_indices] = self.fixed_targets

        for _ in range(iterations):
            self._relaxation_step_vectorized(muscle_mode=muscle_mode)

    def _relaxation_step_vectorized(self, muscle_mode=False):
        """
        Spring relaxation with two modes:
        - muscle_mode=False: Pure mass-spring (maintain rest lengths, settle mesh)
        - muscle_mode=True: Muscle-specific (contour-based, volume preserving)
        """
        displacements = np.zeros_like(self.positions)
        weights = np.zeros(self.num_vertices)

        base_weight = self.stiffness

        if base_weight > 0.01:
            p_i = self.positions[self.edge_i]
            p_j = self.positions[self.edge_j]
            diff = p_j - p_i
            lengths = np.linalg.norm(diff, axis=1)

            valid = lengths > 1e-10
            directions = np.zeros_like(diff)
            directions[valid] = diff[valid] / lengths[valid, np.newaxis]

            if not muscle_mode:
                # === PHASE 1: Pure mass-spring for mesh quality ===
                errors = lengths - self.rest_lengths
            else:
                # === PHASE 2: Muscle-specific behavior ===
                has_contour_types = (hasattr(self, 'cross_contour_edges') and
                                     self.cross_contour_edges is not None and
                                     hasattr(self, 'intra_contour_edges') and
                                     self.intra_contour_edges is not None)

                # Start with baseline
                errors = lengths - self.rest_lengths

                if has_contour_types:
                    # Compute axis ratio
                    axis_ratio = 1.0
                    if self.fixed_targets is not None and len(self.fixed_indices) >= 2:
                        fixed_pos = self.positions[self.fixed_indices]
                        rest_fixed = self.rest_positions[self.fixed_indices]
                        current_axis_len = np.linalg.norm(fixed_pos[-1] - fixed_pos[0])
                        rest_axis_len = np.linalg.norm(rest_fixed[-1] - rest_fixed[0])
                        if rest_axis_len > 1e-6:
                            axis_ratio = np.clip(current_axis_len / rest_axis_len, 0.3, 3.0)

                    # Cross-contour: target follows axis ratio (along fiber, can shrink)
                    cross_target = self.rest_lengths * max(axis_ratio, 0.3)
                    cross_errors = lengths - cross_target

                    # MINIMUM EDGE LENGTH: prevent edges from collapsing (causes folding)
                    min_length = self.rest_lengths * 0.4  # Never shorter than 40% of rest
                    too_short = lengths < min_length
                    # Strong push to restore minimum length
                    cross_errors[too_short & self.cross_contour_edges] = (
                        lengths[too_short & self.cross_contour_edges] -
                        min_length[too_short & self.cross_contour_edges]
                    ) * 3.0  # Strong restoration force

                    # Normal shrink behavior for edges above minimum
                    normal_cross = ~too_short & self.cross_contour_edges
                    cross_errors[normal_cross & (cross_errors < 0)] *= 0.2  # Weak shrink resistance

                    errors[self.cross_contour_edges] = cross_errors[self.cross_contour_edges]

                    # Intra-contour: expand when muscle shortens (perpendicular to fiber)
                    if axis_ratio < 0.98:
                        # Moderate expansion when compressed
                        perp_scale = np.sqrt(1.0 / axis_ratio)
                        perp_scale = np.clip(perp_scale, 1.0, 1.25)  # Cap at 25% expansion
                    else:
                        perp_scale = 1.0
                    intra_target = self.rest_lengths * perp_scale
                    intra_errors = lengths - intra_target

                    # MAXIMUM EDGE LENGTH: prevent Saturn rings (over-expansion)
                    max_length = self.rest_lengths * 1.3  # Never longer than 130% of rest
                    too_long = lengths > max_length
                    # Strong pull back for over-expanded edges
                    intra_errors[too_long & self.intra_contour_edges] = (
                        lengths[too_long & self.intra_contour_edges] -
                        max_length[too_long & self.intra_contour_edges]
                    ) * 2.0  # Pull back force

                    # Normal expansion for edges below max
                    normal_intra = ~too_long & self.intra_contour_edges
                    # Moderate expansion push, stronger over-expansion resistance
                    intra_errors[normal_intra & (intra_errors < 0)] *= 1.5
                    intra_errors[normal_intra & (intra_errors > 0)] *= 0.5
                    errors[self.intra_contour_edges] = intra_errors[self.intra_contour_edges]

            # Clamp for stability
            max_error = np.minimum(self.rest_lengths * 0.25, self.char_length * 0.1)
            errors = np.clip(errors, -max_error, max_error)

            corrections = 0.5 * errors[:, np.newaxis] * directions
            edge_weights = np.full(len(self.edges), base_weight)

            np.add.at(displacements, self.edge_i, edge_weights[:, np.newaxis] * corrections)
            np.add.at(displacements, self.edge_j, -edge_weights[:, np.newaxis] * corrections)
            np.add.at(weights, self.edge_i, edge_weights)
            np.add.at(weights, self.edge_j, edge_weights)

        # === Bone proximity soft constraints ===
        if (self.proximity_stiffness > 0.01 and
            self.proximity_indices is not None and
            self.proximity_targets is not None and
            self.proximity_rest_distances is not None and
            len(self.proximity_indices) > 0):

            prox_pos = self.positions[self.proximity_indices]
            diff_from_bone = prox_pos - self.proximity_targets
            current_dist = np.linalg.norm(diff_from_bone, axis=1)

            too_close = current_dist < self.proximity_rest_distances

            if np.any(too_close):
                direction = np.zeros_like(diff_from_bone)
                valid = current_dist > 1e-10
                direction[valid] = diff_from_bone[valid] / current_dist[valid, np.newaxis]

                push_amount = self.proximity_rest_distances - current_dist
                push_amount = np.clip(push_amount, 0, self.char_length * 0.1)

                push_correction = np.zeros_like(prox_pos)
                push_correction[too_close] = direction[too_close] * push_amount[too_close, np.newaxis]

                prox_weight = self.proximity_stiffness * self.proximity_weights[:, np.newaxis]
                np.add.at(displacements, self.proximity_indices, prox_weight * push_correction)
                np.add.at(weights, self.proximity_indices,
                        self.proximity_stiffness * self.proximity_weights * too_close.astype(float))

        # === Volume-preserving outward bulge (ONLY in muscle_mode and when volume shrinks) ===
        if (muscle_mode and
            hasattr(self, 'outward_directions') and self.outward_directions is not None):

            # Compute current and rest total volume
            v0 = self.positions[self.tetrahedra[:, 0]]
            v1 = self.positions[self.tetrahedra[:, 1]]
            v2 = self.positions[self.tetrahedra[:, 2]]
            v3 = self.positions[self.tetrahedra[:, 3]]
            current_volumes = np.einsum('ij,ij->i', v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0
            current_total = np.sum(np.abs(current_volumes))

            rest_total = np.sum(np.abs(self.rest_volumes))

            if rest_total > 1e-12:
                volume_ratio = current_total / rest_total

                # ONLY push outward when volume is BELOW rest (never increase beyond rest)
                if volume_ratio < 0.99:
                    # How much volume is missing? Push proportionally
                    volume_deficit = 1.0 - volume_ratio  # e.g., 0.2 means 20% volume lost

                    # Moderate push (not too aggressive to avoid Saturn rings)
                    push_strength = np.clip(volume_deficit * 2.5, 0.0, 0.8)

                    # Prioritize intra-contour edges (perpendicular to fiber)
                    has_intra_edge = np.zeros(self.num_vertices, dtype=bool)
                    if hasattr(self, 'intra_contour_edges') and self.intra_contour_edges is not None:
                        intra_edge_indices = np.where(self.intra_contour_edges)[0]
                        for edge_idx in intra_edge_indices:
                            has_intra_edge[self.edge_i[edge_idx]] = True
                            has_intra_edge[self.edge_j[edge_idx]] = True

                    for v_idx in self.free_indices:
                        # Slightly higher priority for vertices with intra-contour edges
                        if has_intra_edge[v_idx]:
                            vertex_weight = 1.2
                        else:
                            vertex_weight = 0.8

                        outward_dir = self.outward_directions[v_idx]
                        push_amount = push_strength * vertex_weight * self.char_length * 0.025

                        displacements[v_idx] += push_amount * outward_dir
                        weights[v_idx] += vertex_weight

                # If volume is ABOVE rest, pull inward slightly to prevent over-expansion
                elif volume_ratio > 1.01:
                    volume_excess = volume_ratio - 1.0
                    pull_strength = np.clip(volume_excess * 1.0, 0.0, 0.5)

                    for v_idx in self.free_indices:
                        outward_dir = self.outward_directions[v_idx]
                        pull_amount = pull_strength * self.char_length * 0.01
                        displacements[v_idx] -= pull_amount * outward_dir
                        weights[v_idx] += 0.5

        # === Apply displacements with stricter clamping ===
        valid_weights = weights > 1e-10
        update_mask = self.free_mask & valid_weights

        if np.any(update_mask):
            update = displacements[update_mask] / weights[update_mask, np.newaxis]

            # Per-vertex update clamping
            update_norms = np.linalg.norm(update, axis=1, keepdims=True)
            max_update = self.char_length * 0.03  # Allow larger updates for muscle
            too_large = update_norms > max_update
            if np.any(too_large):
                update = np.where(too_large, update * max_update / (update_norms + 1e-10), update)

            self.positions[update_mask] += (1.0 - self.damping) * update

        # Re-enforce fixed positions
        if self.fixed_targets is not None and len(self.fixed_indices) > 0:
            self.positions[self.fixed_indices] = self.fixed_targets

    # =========================================================================
    # ARAP (As-Rigid-As-Possible) Deformation
    # =========================================================================

    def init_arap(self):
        """
        Initialize ARAP data structures.
        Must be called before using ARAP solver.
        """
        n = self.num_vertices

        # Build neighbor list from edges
        self.arap_neighbors = [[] for _ in range(n)]
        for i, j in zip(self.edge_i, self.edge_j):
            self.arap_neighbors[i].append(j)
            self.arap_neighbors[j].append(i)

        # Uniform weights for tet mesh edges
        self.arap_weights = {}
        for i, j in zip(self.edge_i, self.edge_j):
            self.arap_weights[(i, j)] = 1.0
            self.arap_weights[(j, i)] = 1.0

        # Store rest edge vectors
        self.arap_rest_edges = []
        for i in range(n):
            edges = {}
            for j in self.arap_neighbors[i]:
                edges[j] = self.rest_positions[j] - self.rest_positions[i]
            self.arap_rest_edges.append(edges)

        # Build and pre-factorize system matrix with fixed constraints
        self._build_arap_system()

        self.arap_initialized = True
        print(f"  ARAP initialized: {n} vertices, {len(self.edges)} edges")

    def _build_arap_system(self):
        """Build the ARAP system matrix with fixed vertex constraints baked in."""
        n = self.num_vertices

        # Small regularization to prevent singularity for disconnected vertices
        regularization = 1e-6

        # Build Laplacian: L_ii = sum of weights, L_ij = -w_ij
        L = scipy.sparse.lil_matrix((n, n))
        disconnected_count = 0
        for i in range(n):
            if self.fixed_mask[i]:
                # Fixed vertex: identity row
                L[i, i] = 1.0
            else:
                weight_sum = 0.0
                for j in self.arap_neighbors[i]:
                    w = self.arap_weights[(i, j)]
                    L[i, j] = -w
                    weight_sum += w
                # Add regularization to prevent singularity
                L[i, i] = weight_sum + regularization
                if weight_sum == 0:
                    disconnected_count += 1
                    # Print diagnostic info for disconnected vertex
                    if disconnected_count <= 5:  # Limit output
                        pos = self.rest_positions[i]
                        print(f"    Disconnected vertex {i}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

        if disconnected_count > 0:
            print(f"  WARNING: {disconnected_count} disconnected free vertices detected")
            # Additional diagnostic: find which tetrahedra use these vertices
            disconnected_indices = []
            for i in range(n):
                if not self.fixed_mask[i] and len(self.arap_neighbors[i]) == 0:
                    disconnected_indices.append(i)
            if len(disconnected_indices) > 0:
                # Check if these vertices are in any tetrahedra
                tet_usage = np.zeros(n, dtype=int)
                for tet in self.tetrahedra:
                    for vi in tet:
                        tet_usage[vi] += 1
                for vi in disconnected_indices[:5]:  # Limit output
                    print(f"    Vertex {vi}: used in {tet_usage[vi]} tetrahedra")

        self.arap_L = L.tocsr()

        # Pre-factorize for faster solving
        try:
            self.arap_solver = scipy.sparse.linalg.factorized(self.arap_L)
            self.arap_use_factorized = True
        except Exception:
            self.arap_use_factorized = False

    def _arap_local_step(self, target_edges=None):
        """
        ARAP local step: compute optimal rotation for each vertex cell.
        Solves: R_i = argmin sum_j w_ij ||curr_edge_ij - R_i * rest_edge_ij||^2
        """
        rotations = [np.eye(3)] * self.num_vertices  # Initialize all to identity

        for i in range(self.num_vertices):
            neighbors = self.arap_neighbors[i]
            if len(neighbors) == 0:
                continue  # Already identity

            # Build covariance matrix S = sum_j w_ij * curr_edge * rest_edge^T
            # rest_edge = p_j - p_i (from i to j)
            # curr_edge = p'_j - p'_i
            # Works even for single-neighbor vertices (SVD gives best-fit rotation)
            S = np.zeros((3, 3))
            for j in neighbors:
                w = self.arap_weights[(i, j)]

                if target_edges is not None and j in target_edges[i]:
                    rest_edge = target_edges[i][j]  # p_j - p_i
                else:
                    rest_edge = self.arap_rest_edges[i][j]

                curr_edge = self.positions[j] - self.positions[i]
                # S = P' * P^T where P are rest edges, P' are current edges
                S += w * np.outer(curr_edge, rest_edge)

            # Check for degenerate S matrix
            S_norm = np.linalg.norm(S, 'fro')
            if S_norm < 1e-10:
                continue  # Keep identity rotation

            # SVD: S = U * Sigma * V^T, then R = U * V^T
            try:
                U, sigma, Vt = np.linalg.svd(S)
                # Check for degenerate singular values
                if sigma[0] < 1e-10:
                    continue  # Keep identity
                R = U @ Vt

                # Ensure proper rotation (det = 1, not reflection)
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt

                # Verify R is a valid rotation
                if not np.isfinite(R).all():
                    continue  # Keep identity
                rotations[i] = R
            except Exception:
                pass  # Keep identity rotation

        return rotations

    def _arap_global_step(self, rotations, target_edges=None):
        """
        ARAP global step: solve L * p' = b for new positions.

        RHS: b_i = sum_j (w_ij/2) * (R_i + R_j) * (p_i - p_j)
        Note: (p_i - p_j) = -rest_edge[i][j] since rest_edge[i][j] = p_j - p_i
        """
        n = self.num_vertices

        # Must match regularization in _build_arap_system
        regularization = 1e-6

        # Build RHS
        b = np.zeros((n, 3))
        for i in range(n):
            if self.fixed_mask[i]:
                # Fixed vertex: RHS = target position
                idx = np.where(self.fixed_indices == i)[0]
                if len(idx) > 0 and self.fixed_targets is not None:
                    b[i] = self.fixed_targets[idx[0]]
                else:
                    b[i] = self.rest_positions[i]
            else:
                for j in self.arap_neighbors[i]:
                    w = self.arap_weights[(i, j)]

                    if target_edges is not None and j in target_edges[i]:
                        rest_edge_ij = target_edges[i][j]  # p_j - p_i
                    else:
                        rest_edge_ij = self.arap_rest_edges[i][j]

                    # (p_i - p_j) = -rest_edge_ij
                    edge_i_to_j = -rest_edge_ij

                    # Average rotation * edge from i to j
                    R_avg = 0.5 * (rotations[i] + rotations[j])
                    b[i] += w * (R_avg @ edge_i_to_j)

                # Add regularization term (anchors to rest position)
                b[i] += regularization * self.rest_positions[i]

        # Solve system
        new_positions = self.positions.copy()  # Start with current positions as fallback
        for coord in range(3):
            try:
                if self.arap_use_factorized:
                    result = self.arap_solver(b[:, coord])
                else:
                    result = scipy.sparse.linalg.spsolve(
                        self.arap_L, b[:, coord]
                    )
                # Check result validity before assigning
                if np.isfinite(result).all():
                    new_positions[:, coord] = result
                else:
                    print(f"  WARNING: Non-finite values in ARAP solve for coord {coord}")
            except Exception as e:
                print(f"  ARAP solve failed: {e}")
                # Keep current positions for this coordinate

        return new_positions

    def solve_arap(self, max_iterations=10, tolerance=1e-6, target_edges=None,
                   collision_meshes=None, collision_margin=0.002):
        """
        Solve ARAP deformation using alternating local-global optimization.
        Optionally includes collision detection in the iteration loop.

        Args:
            max_iterations: Maximum ARAP iterations
            tolerance: Convergence tolerance
            target_edges: Optional scaled target edges for muscle behavior
            collision_meshes: List of trimesh objects for collision detection
            collision_margin: Distance to push vertices outside collision meshes
        """
        if not hasattr(self, 'arap_initialized') or not self.arap_initialized:
            self.init_arap()

        # Ensure fixed vertices are at target
        if self.fixed_targets is not None and len(self.fixed_indices) > 0:
            self.positions[self.fixed_indices] = self.fixed_targets

        for iteration in range(max_iterations):
            old_positions = self.positions.copy()

            # Local step: find best rotations for current positions
            rotations = self._arap_local_step(target_edges)

            # Global step: solve for positions given rotations
            self.positions = self._arap_global_step(rotations, target_edges)

            # Check for NaN/Inf and recover if needed
            if not np.isfinite(self.positions).all():
                print(f"  ERROR: Non-finite positions at iteration {iteration}, reverting")
                self.positions = old_positions.copy()
                break

            # Re-enforce fixed positions
            if self.fixed_targets is not None and len(self.fixed_indices) > 0:
                self.positions[self.fixed_indices] = self.fixed_targets

            # Collision resolution within ARAP loop (multiple passes)
            if collision_meshes is not None and len(collision_meshes) > 0:
                total_pushed = 0
                for pass_i in range(3):  # Multiple passes for deep penetrations
                    pushed = self._resolve_collisions_arap(collision_meshes, collision_margin)
                    total_pushed += pushed
                    if pushed == 0:
                        break
                if iteration == 0 and total_pushed > 0:
                    print(f"    Collision: pushed {total_pushed} verts (margin={collision_margin:.4f})")

            # Check convergence
            if len(self.free_indices) > 0:
                disp = np.linalg.norm(
                    self.positions[self.free_indices] - old_positions[self.free_indices],
                    axis=1
                )
                max_disp = np.max(disp)
            else:
                max_disp = 0.0

            if max_disp < tolerance:
                return iteration + 1, max_disp

        # Fix isolated vertices (0 neighbors) by finding closest connected vertex
        for i in self.free_indices:
            if len(self.arap_neighbors[i]) == 0:
                rest_pos = self.rest_positions[i]
                best_dist = float('inf')
                best_j = -1
                for j in range(self.num_vertices):
                    if j != i and len(self.arap_neighbors[j]) > 0:
                        d = np.linalg.norm(self.rest_positions[j] - rest_pos)
                        if d < best_dist:
                            best_dist = d
                            best_j = j
                if best_j >= 0:
                    disp = self.positions[best_j] - self.rest_positions[best_j]
                    self.positions[i] = rest_pos + disp
                    print(f"  Fixed isolated vertex {i} by copying displacement from vertex {best_j}")

        # Check for other stuck vertices (didn't move from rest)
        total_disp_from_rest = np.linalg.norm(self.positions - self.rest_positions, axis=1)
        stuck_threshold = 1e-6  # Essentially didn't move
        for i in self.free_indices:
            if total_disp_from_rest[i] < stuck_threshold:
                n_neighbors = len(self.arap_neighbors[i])
                if n_neighbors > 0:
                    neighbor_positions = [self.positions[j] for j in self.arap_neighbors[i]]
                    neighbor_avg = np.mean(neighbor_positions, axis=0)
                    # Move toward neighbors
                    self.positions[i] = 0.3 * self.positions[i] + 0.7 * neighbor_avg
                    print(f"  Fixed stuck vertex {i} (neighbors={n_neighbors}) by moving toward neighbors")

        # Final collision pass (not undone by ARAP)
        if collision_meshes is not None and len(collision_meshes) > 0:
            final_pushed = 0
            for _ in range(5):
                pushed = self._resolve_collisions_arap(collision_meshes, collision_margin)
                final_pushed += pushed
                if pushed == 0:
                    break
            if final_pushed > 0:
                print(f"    Final collision: pushed {final_pushed} verts")

        return max_iterations, max_disp

    def _resolve_collisions_arap(self, collision_meshes, margin=0.002):
        """
        Resolve collisions by pushing free vertices out of collision meshes.
        Called within ARAP iteration loop.
        Uses face normal to determine inside/outside (works for non-watertight meshes).
        Also checks edge midpoints and face centroids for more robust collision.
        """
        total_pushed = 0
        for mesh in collision_meshes:
            if mesh is None:
                continue

            try:
                # === 1. Check free vertices ===
                free_positions = self.positions[self.free_indices]
                n_free = len(free_positions)

                # === 2. Compute edge midpoints (sample every 4th edge for speed) ===
                edge_midpoints = []
                edge_vertex_pairs = []  # (i, j) for each midpoint
                step = max(1, len(self.edge_i) // 500)  # Limit to ~500 edges
                for idx in range(0, len(self.edge_i), step):
                    i, j = self.edge_i[idx], self.edge_j[idx]
                    mid = (self.positions[i] + self.positions[j]) * 0.5
                    edge_midpoints.append(mid)
                    edge_vertex_pairs.append((i, j))
                edge_midpoints = np.array(edge_midpoints) if edge_midpoints else np.zeros((0, 3))
                n_edges = len(edge_midpoints)

                # === 3. Compute face centroids (if tet_faces available, sample) ===
                face_centroids = []
                face_vertex_lists = []
                if hasattr(self, 'tet_faces') and self.tet_faces is not None and len(self.tet_faces) > 0:
                    step = max(1, len(self.tet_faces) // 300)  # Limit to ~300 faces
                    for idx in range(0, len(self.tet_faces), step):
                        face = self.tet_faces[idx]
                        centroid = np.mean(self.positions[face], axis=0)
                        face_centroids.append(centroid)
                        face_vertex_lists.append(face)
                face_centroids = np.array(face_centroids) if face_centroids else np.zeros((0, 3))
                n_faces = len(face_centroids)

                # === 4. Batch all points for single query ===
                all_points = np.vstack([free_positions, edge_midpoints, face_centroids]) if (n_edges > 0 or n_faces > 0) else free_positions

                closest_points, distances, face_ids = mesh.nearest.on_surface(all_points)
                face_normals = mesh.face_normals[face_ids]
                to_point = all_points - closest_points
                dot_products = np.sum(to_point * face_normals, axis=1)

                inside_mask = dot_products < 0
                too_close = distances < margin
                needs_push = inside_mask | too_close

                # === 5. Process vertex collisions ===
                vertex_push = needs_push[:n_free]
                if np.any(vertex_push):
                    push_indices = np.where(vertex_push)[0]
                    new_positions = free_positions.copy()
                    new_positions[push_indices] = (
                        closest_points[push_indices] +
                        face_normals[push_indices] * margin
                    )
                    self.positions[self.free_indices] = new_positions
                    total_pushed += len(push_indices)

                # === 6. Process edge midpoint collisions -> push both vertices ===
                if n_edges > 0:
                    edge_push = needs_push[n_free:n_free + n_edges]
                    edge_push_idx = np.where(edge_push)[0]
                    for idx in edge_push_idx:
                        i, j = edge_vertex_pairs[idx]
                        push_dir = face_normals[n_free + idx]
                        push_dist = margin - distances[n_free + idx] if not inside_mask[n_free + idx] else distances[n_free + idx] + margin
                        # Push both vertices half the distance
                        self.positions[i] += push_dir * (push_dist * 0.5)
                        self.positions[j] += push_dir * (push_dist * 0.5)
                        total_pushed += 2

                # === 7. Process face centroid collisions -> push all face vertices ===
                if n_faces > 0:
                    face_push = needs_push[n_free + n_edges:]
                    face_push_idx = np.where(face_push)[0]
                    for idx in face_push_idx:
                        face_verts = face_vertex_lists[idx]
                        push_dir = face_normals[n_free + n_edges + idx]
                        push_dist = margin - distances[n_free + n_edges + idx] if not inside_mask[n_free + n_edges + idx] else distances[n_free + n_edges + idx] + margin
                        # Push all face vertices by 1/3 the distance
                        for vi in face_verts:
                            self.positions[vi] += push_dir * (push_dist * 0.35)
                        total_pushed += len(face_verts)

            except Exception as e:
                print(f"  Collision error: {e}")
                continue

        return total_pushed

    def solve_arap_muscle(self, max_iterations=10, tolerance=1e-6,
                          collision_meshes=None, collision_margin=0.002):
        """
        ARAP with muscle-specific target edge lengths.
        Optionally includes collision detection in the iteration loop.

        Args:
            max_iterations: Maximum ARAP iterations
            tolerance: Convergence tolerance
            collision_meshes: List of trimesh objects for collision detection
            collision_margin: Distance to push vertices outside collision meshes
        """
        if not hasattr(self, 'arap_initialized') or not self.arap_initialized:
            self.init_arap()

        # Compute axis ratio
        axis_ratio = 1.0
        if self.fixed_targets is not None and len(self.fixed_indices) >= 2:
            fixed_pos = self.fixed_targets
            rest_fixed = self.rest_positions[self.fixed_indices]
            current_len = np.linalg.norm(fixed_pos[-1] - fixed_pos[0])
            rest_len = np.linalg.norm(rest_fixed[-1] - rest_fixed[0])
            if rest_len > 1e-6:
                axis_ratio = np.clip(current_len / rest_len, 0.5, 2.0)

        # Build target edges based on muscle behavior
        target_edges = None
        if (hasattr(self, 'cross_contour_edges') and self.cross_contour_edges is not None and
            hasattr(self, 'intra_contour_edges') and self.intra_contour_edges is not None):

            # Perpendicular scale for volume preservation
            if axis_ratio < 0.98:
                perp_scale = np.sqrt(1.0 / axis_ratio)
                perp_scale = np.clip(perp_scale, 1.0, 1.2)
            else:
                perp_scale = 1.0

            # Create scaled target edges
            target_edges = [{} for _ in range(self.num_vertices)]

            for edge_idx, (i, j) in enumerate(zip(self.edge_i, self.edge_j)):
                rest_ij = self.arap_rest_edges[i][j]
                rest_ji = self.arap_rest_edges[j][i]

                if self.cross_contour_edges[edge_idx]:
                    # Cross-contour: scale with axis ratio
                    scale = axis_ratio
                elif self.intra_contour_edges[edge_idx]:
                    # Intra-contour: expand perpendicular
                    scale = perp_scale
                else:
                    scale = 1.0

                target_edges[i][j] = rest_ij * scale
                target_edges[j][i] = rest_ji * scale

        return self.solve_arap(max_iterations, tolerance, target_edges,
                               collision_meshes, collision_margin)

    # ============================================================================
    # HYBRID ARAP + VIPER-LIKE VOLUME PRESERVATION
    # ============================================================================

    def init_hybrid_mode(self, waypoints, waypoints_original):
        """
        Initialize hybrid mode by computing vertex-to-fiber mapping.
        Must be called before solve_arap_hybrid.

        Args:
            waypoints: Current waypoints [stream_idx][level_idx] = (num_fibers, 3)
            waypoints_original: Original waypoints (same structure)
        """
        # Store original waypoints for ratio computation
        self.hybrid_waypoints_original = waypoints_original

        # Compute rest fiber lengths
        self.fiber_rest_lengths = self._compute_fiber_lengths(waypoints_original)

        # Compute vertex-to-fiber membership mapping
        self._compute_vertex_fiber_membership(waypoints_original)

        self.hybrid_initialized = True
        print(f"  Hybrid mode initialized: {len(self.fiber_rest_lengths)} fiber streams")

    def _compute_fiber_lengths(self, waypoints):
        """Compute total length from origin to insertion for each fiber stream."""
        lengths = []
        for stream_idx in range(len(waypoints)):
            stream = waypoints[stream_idx]
            if len(stream) < 2:
                lengths.append(1.0)
                continue

            # Get centroids of first and last contour levels
            first_level = np.array(stream[0])  # (num_fibers, 3)
            last_level = np.array(stream[-1])   # (num_fibers, 3)

            # Use centroid of all fibers at each level
            first_centroid = np.mean(first_level, axis=0)
            last_centroid = np.mean(last_level, axis=0)

            length = np.linalg.norm(last_centroid - first_centroid)
            lengths.append(max(length, 1e-6))

        return np.array(lengths)

    def _compute_vertex_fiber_membership(self, waypoints, k=3):
        """
        Map each tet vertex to its k nearest fiber streams.
        Uses inverse distance weighting to fiber centerlines.

        Args:
            waypoints: Waypoint structure [stream_idx][level_idx] = (num_fibers, 3)
            k: Number of nearest streams to consider
        """
        num_streams = len(waypoints)
        if num_streams == 0:
            self.vertex_fiber_membership = None
            self.vertex_fiber_weights = None
            return

        k = min(k, num_streams)

        # Compute centerline for each stream (average of all waypoints)
        stream_centerlines = []
        for stream_idx in range(num_streams):
            stream = waypoints[stream_idx]
            all_points = []
            for level in stream:
                level_arr = np.array(level)
                if level_arr.ndim == 1:
                    all_points.append(level_arr)
                else:
                    all_points.extend(level_arr)
            if len(all_points) > 0:
                centerline = np.mean(all_points, axis=0)
            else:
                centerline = np.zeros(3)
            stream_centerlines.append(centerline)
        stream_centerlines = np.array(stream_centerlines)  # (num_streams, 3)

        # For each vertex, find k nearest streams
        self.vertex_fiber_membership = np.zeros((self.num_vertices, k), dtype=np.int32)
        self.vertex_fiber_weights = np.zeros((self.num_vertices, k), dtype=np.float64)

        for v_idx in range(self.num_vertices):
            v_pos = self.rest_positions[v_idx]

            # Compute distances to all stream centerlines
            dists = np.linalg.norm(stream_centerlines - v_pos, axis=1)

            # Get k nearest
            nearest_indices = np.argsort(dists)[:k]
            nearest_dists = dists[nearest_indices]

            # Inverse distance weighting
            # Add small epsilon to avoid division by zero
            inv_dists = 1.0 / (nearest_dists + 1e-6)
            weights = inv_dists / np.sum(inv_dists)

            self.vertex_fiber_membership[v_idx] = nearest_indices
            self.vertex_fiber_weights[v_idx] = weights

    def compute_fiber_ratios(self, waypoints):
        """
        Compute stretch ratio for each fiber stream from endpoint positions.

        Args:
            waypoints: Current waypoints [stream_idx][level_idx] = (num_fibers, 3)

        Returns:
            ratios: (num_streams,) array of stretch ratios
        """
        current_lengths = self._compute_fiber_lengths(waypoints)

        if not hasattr(self, 'fiber_rest_lengths') or self.fiber_rest_lengths is None:
            return np.ones(len(waypoints))

        ratios = current_lengths / self.fiber_rest_lengths
        # Clip to reasonable range to prevent extreme deformations
        ratios = np.clip(ratios, 0.5, 2.0)

        return ratios

    def compute_vertex_scales(self, fiber_ratios):
        """
        Interpolate per-vertex scale factors from fiber ratios.
        Uses volume preservation formula: scale = sqrt(1 / ratio)

        Args:
            fiber_ratios: (num_streams,) array of stretch ratios

        Returns:
            vertex_scales: (num_vertices,) array of scale factors
        """
        vertex_scales = np.ones(self.num_vertices)

        if self.vertex_fiber_weights is None or self.vertex_fiber_membership is None:
            return vertex_scales

        for v_idx in range(self.num_vertices):
            # Get fiber membership for this vertex
            indices = self.vertex_fiber_membership[v_idx]
            weights = self.vertex_fiber_weights[v_idx]

            # Weighted average of nearby fiber ratios
            weighted_ratio = np.sum(weights * fiber_ratios[indices])

            # Volume preserving scale: scale^2 * ratio = 1
            # Therefore: scale = sqrt(1 / ratio)
            vertex_scales[v_idx] = np.sqrt(1.0 / max(weighted_ratio, 0.5))

        return vertex_scales

    def _build_hybrid_target_edges(self, fiber_ratios, vertex_scales, volume_stiffness=0.8):
        """
        Build target edge vectors for hybrid ARAP.

        Args:
            fiber_ratios: (num_streams,) stretch ratios
            vertex_scales: (num_vertices,) volume preservation scales
            volume_stiffness: Blend factor for volume preservation (0-1)

        Returns:
            target_edges: List of dicts for ARAP solver
        """
        target_edges = [{} for _ in range(self.num_vertices)]

        has_edge_types = (hasattr(self, 'cross_contour_edges') and
                         self.cross_contour_edges is not None and
                         hasattr(self, 'intra_contour_edges') and
                         self.intra_contour_edges is not None)

        for edge_idx, (i, j) in enumerate(zip(self.edge_i, self.edge_j)):
            rest_ij = self.arap_rest_edges[i][j]
            rest_ji = self.arap_rest_edges[j][i]

            if has_edge_types:
                if self.cross_contour_edges[edge_idx]:
                    # Cross-contour (along fiber): scale by interpolated ratio
                    ratio_i = self._get_vertex_ratio(i, fiber_ratios)
                    ratio_j = self._get_vertex_ratio(j, fiber_ratios)
                    scale = (ratio_i + ratio_j) / 2.0
                elif self.intra_contour_edges[edge_idx]:
                    # Intra-contour (perpendicular): scale by volume preservation
                    scale_i = vertex_scales[i]
                    scale_j = vertex_scales[j]
                    avg_scale = (scale_i + scale_j) / 2.0
                    # Blend with identity based on volume_stiffness
                    scale = 1.0 + volume_stiffness * (avg_scale - 1.0)
                else:
                    scale = 1.0
            else:
                # No edge classification - use average behavior
                ratio_i = self._get_vertex_ratio(i, fiber_ratios)
                ratio_j = self._get_vertex_ratio(j, fiber_ratios)
                avg_ratio = (ratio_i + ratio_j) / 2.0
                scale = avg_ratio

            target_edges[i][j] = rest_ij * scale
            target_edges[j][i] = rest_ji * scale

        return target_edges

    def _get_vertex_ratio(self, vertex_idx, fiber_ratios):
        """Get interpolated fiber ratio for a vertex."""
        if self.vertex_fiber_weights is None or self.vertex_fiber_membership is None:
            return 1.0

        weights = self.vertex_fiber_weights[vertex_idx]
        indices = self.vertex_fiber_membership[vertex_idx]
        return np.sum(weights * fiber_ratios[indices])

    def solve_arap_hybrid(self, waypoints, waypoints_original=None,
                          max_iterations=10, tolerance=1e-6,
                          collision_meshes=None, collision_margin=0.002,
                          volume_stiffness=0.8):
        """
        ARAP with per-fiber volume preservation (VIPER-like).

        Cross-contour edges: target = rest_length * ratio (stretch/compress with fiber)
        Intra-contour edges: target = rest_length * scale (perpendicular expansion/contraction)

        Args:
            waypoints: Current waypoints [stream_idx][level_idx] = (num_fibers, 3)
            waypoints_original: Original waypoints (optional, uses stored if None)
            max_iterations: Maximum ARAP iterations
            tolerance: Convergence tolerance
            collision_meshes: List of trimesh objects for collision detection
            collision_margin: Distance to push vertices outside collision meshes
            volume_stiffness: Weight for volume preservation (0-1)

        Returns:
            (iterations, residual): Number of iterations and final residual
        """
        if not hasattr(self, 'arap_initialized') or not self.arap_initialized:
            self.init_arap()

        # Initialize hybrid mode if needed
        if waypoints_original is not None:
            if not hasattr(self, 'hybrid_initialized') or not self.hybrid_initialized:
                self.init_hybrid_mode(waypoints, waypoints_original)

        # Use stored original if not provided
        if waypoints_original is None:
            if hasattr(self, 'hybrid_waypoints_original'):
                waypoints_original = self.hybrid_waypoints_original
            else:
                # Fall back to standard ARAP
                return self.solve_arap_muscle(max_iterations, tolerance,
                                              collision_meshes, collision_margin)

        # Compute fiber ratios from current waypoints
        fiber_ratios = self.compute_fiber_ratios(waypoints)

        # Compute per-vertex scales for volume preservation
        vertex_scales = self.compute_vertex_scales(fiber_ratios)

        # Build target edges with hybrid scaling
        target_edges = self._build_hybrid_target_edges(fiber_ratios, vertex_scales, volume_stiffness)

        # Run standard ARAP with modified targets
        return self.solve_arap(max_iterations, tolerance, target_edges,
                               collision_meshes, collision_margin)

    def reset(self):
        """Reset simulation to rest state."""
        self.positions = self.rest_positions.copy()
        if self.initial_fixed_targets is not None:
            self.fixed_targets = self.initial_fixed_targets.copy()
        self._neighbor_cache = None  # Clear cache on reset

    def get_positions(self):
        """Get current vertex positions."""
        return self.positions.copy()

    def set_positions(self, positions):
        """Set current vertex positions."""
        self.positions = np.array(positions, dtype=np.float64)

    def set_contour_edge_types(self, cross_contour_mask, intra_contour_mask):
        """
        Set edge types based on contour structure for muscle-like behavior.

        Args:
            cross_contour_mask: Boolean array - True for edges connecting different contours
                               (along fiber direction, can shrink)
            intra_contour_mask: Boolean array - True for edges within same contour
                               (perpendicular to fibers, should expand when muscle shortens)
        """
        self.cross_contour_edges = np.array(cross_contour_mask, dtype=bool)
        self.intra_contour_edges = np.array(intra_contour_mask, dtype=bool)

    def set_outward_directions(self, outward_dirs):
        """
        Set pre-computed outward direction for each vertex.
        Used for volume-preserving bulge when muscle shortens.

        Args:
            outward_dirs: Nx3 array of unit vectors pointing outward from muscle axis
        """
        if outward_dirs is not None and len(outward_dirs) == self.num_vertices:
            self.outward_directions = np.array(outward_dirs, dtype=np.float64)
        else:
            self.outward_directions = None


class MuscleMeshMixin:
    """
    Mixin class containing muscle mesh specific functionality for MeshLoader.
    This includes contour processing, tetrahedralization, soft body simulation,
    fiber architecture, and waypoint computation methods.
    """

    def _init_muscle_properties(self):
        """Initialize muscle-specific properties. Called from MeshLoader.__init__"""
        # For Muscle Meshes
        self.frames = []
        self.bboxes = []
        self.contour_matches = []
        self.bounding_planes = []

        self.scalar_field = None
        self._face_scalar_min = None  # Invalidate precomputed ranges
        self._face_scalar_max = None
        self.vertex_colors = None

        # Scalar field animation state
        self._scalar_anim_active = False
        self._scalar_anim_progress = 0.0
        self._scalar_anim_target_colors = None
        self._scalar_anim_normalized_u = None
        self._scalar_replayed = False
        self.contours = None
        self.contour_mesh_vertices = None
        self.contour_mesh_faces = None
        self.contour_mesh_normals = None
        self.is_draw_contour_mesh = False
        self.contour_mesh_color = np.array([0.8, 0.5, 0.5])
        self.contour_mesh_transparency = 0.8

        # Tetrahedron mesh for soft body simulation
        self.tet_vertices = None
        self.tet_faces = None  # Backwards compat alias to tet_render_faces
        self.tet_render_faces = None  # Original contour faces for rendering/collision
        self.tet_sim_faces = None  # Tet boundary faces for simulation
        self.tet_tetrahedra = None
        self.tet_cap_face_indices = []
        self.tet_anchor_vertices = []
        self.tet_surface_face_count = 0
        self.is_draw_tet_mesh = False
        self.is_draw_tet_edges = False
        self.is_draw_constraints = False

        # Soft body simulation properties (quasistatic, on-demand)
        self.soft_body = None  # SoftBodySimulation instance
        self.soft_body_stiffness = 0.5
        self.soft_body_damping = 0.3
        self.soft_body_volume_stiffness = 0.0
        self.soft_body_fixed_vertices = []
        self.soft_body_collision = True
        self.soft_body_collision_margin = 0.008  # 8mm margin
        self.use_arap = True

        # Cap-to-skeleton mapping
        self.tet_cap_attachments = []
        self.specific_contour = None
        self.specific_contour_value = 1.0
        self.contour_value_min = 1.1
        self.contour_value_max = 9.9
        self.normalized_contours = []

        self.structure_vectors = []

        self.contours_discarded = None
        self.bounding_planes_discarded = None

        self.fiber_architecture = [self._sobol_sampling_barycentric_default(16)]
        self.is_draw_fiber_architecture = False
        self.is_one_fiber = False
        self.sampling_method = 'sobol_unit_square'
        self.cutting_method = 'bp'

        self.waypoints = []
        self.waypoints_from_tet_sim = True

        self.link_mode = 'mean'
        self.min_contour_distance = 0.005  # Minimum distance between contours in a stream (5mm default)
        self.contour_spacing_scale = 0.8  # < 1.0 = more contours, > 1.0 = fewer contours
        self.bounding_box_method = 'farthest_vertex'  # 'farthest_vertex' or 'pca'

        self.attach_skeletons = []
        self.attach_skeletons_sub = []
        self.attach_skeleton_names = []

        # VIPER rod simulation properties
        self.viper_sim = None
        self.viper_only_mode = False
        self.is_draw_viper = False
        self.is_draw_viper_rod_mesh = False
        self.viper_skinning_computed = False
        self.viper_rest_mesh_vertices = None

    def _sobol_sampling_barycentric_default(self, num_samples):
        """Default sobol sampling for initialization."""
        try:
            from scipy.stats import qmc
            seed = getattr(self, 'fiber_sampling_seed', 42)
            sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
            samples = sampler.random(num_samples)
            barycentric = []
            for s in samples:
                u, v = s
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                w = 1 - u - v
                barycentric.append([u, v, w])
            return np.array(barycentric)
        except Exception:
            # Fallback to uniform random
            barycentric = []
            for _ in range(num_samples):
                u, v = np.random.random(2)
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                w = 1 - u - v
                barycentric.append([u, v, w])
            return np.array(barycentric)

    # ========================================================================
    # Contour-to-Tet Mapping Methods
    # ========================================================================

    def build_contour_to_tet_mapping(self):
        """
        Build a mapping from contour mesh vertices to tet vertex indices.
        Call this after both build_contour_mesh and tetrahedralization.
        """
        from scipy.spatial import cKDTree

        if self.contour_mesh_vertices is None or len(self.contour_mesh_vertices) == 0:
            print("No contour mesh vertices")
            return
        if self.tet_vertices is None or len(self.tet_vertices) == 0:
            print("No tet vertices")
            return

        # Build KD-tree from tet vertices
        tet_tree = cKDTree(self.tet_vertices)

        # Find nearest tet vertex for each contour mesh vertex
        self.contour_to_tet_indices = []
        matched = 0
        for i, cv in enumerate(self.contour_mesh_vertices):
            dist, idx = tet_tree.query(cv)
            if dist < 1e-4:  # Close enough to be considered the same vertex
                self.contour_to_tet_indices.append(idx)
                matched += 1
            else:
                self.contour_to_tet_indices.append(-1)  # No match

        print(f"Contour-to-tet mapping: {matched}/{len(self.contour_mesh_vertices)} vertices matched")

    def compute_tet_edge_contour_types(self):
        """
        Classify tet mesh edges based on contour structure.

        Cross-contour edges: connect vertices from different contour levels
                            (along fiber direction, can shrink)
        Intra-contour edges: connect vertices within same contour level
                            (perpendicular to fibers, resist compression)

        Returns:
            (cross_contour_mask, intra_contour_mask): Boolean arrays for edge classification
        """
        if self.soft_body is None:
            print("  Edge classification: no soft body")
            return None, None

        if not hasattr(self, 'vertex_contour_level') or self.vertex_contour_level is None:
            print("  Edge classification: no vertex_contour_level (build contour mesh first)")
            return None, None

        if not hasattr(self, 'contour_to_tet_indices') or self.contour_to_tet_indices is None:
            self.build_contour_to_tet_mapping()

        # Build reverse mapping: tet_vertex_idx -> contour_level
        n_tet_verts = len(self.tet_vertices)
        tet_vertex_level = np.full(n_tet_verts, -1.0, dtype=np.float32)

        # First, assign known levels from contour mesh
        max_level = 0
        for contour_idx, tet_idx in enumerate(self.contour_to_tet_indices):
            if tet_idx >= 0 and tet_idx < n_tet_verts:
                level = self.vertex_contour_level[contour_idx]
                tet_vertex_level[tet_idx] = float(level)
                max_level = max(max_level, level)

        # Compute muscle axis from fixed vertices
        tet_verts = np.array(self.tet_vertices)
        fixed_indices = list(self.soft_body_fixed_vertices) if hasattr(self, 'soft_body_fixed_vertices') else []

        if len(fixed_indices) >= 2:
            # Cluster fixed vertices into origin and insertion based on level
            origin_verts = []
            insertion_verts = []
            for fi in fixed_indices:
                if tet_vertex_level[fi] >= 0:
                    if tet_vertex_level[fi] < max_level / 2:
                        origin_verts.append(tet_verts[fi])
                    else:
                        insertion_verts.append(tet_verts[fi])

            if len(origin_verts) > 0 and len(insertion_verts) > 0:
                origin_center = np.mean(origin_verts, axis=0)
                insertion_center = np.mean(insertion_verts, axis=0)
            else:
                origin_center = tet_verts[fixed_indices[0]]
                insertion_center = tet_verts[fixed_indices[-1]]

            muscle_axis = insertion_center - origin_center
            axis_len = np.linalg.norm(muscle_axis)

            if axis_len > 1e-6:
                muscle_axis_norm = muscle_axis / axis_len

                # Interpolate level for internal vertices based on position along axis
                for v_idx in range(n_tet_verts):
                    if tet_vertex_level[v_idx] < 0:  # Internal vertex
                        to_vertex = tet_verts[v_idx] - origin_center
                        t = np.dot(to_vertex, muscle_axis_norm) / axis_len
                        t = np.clip(t, 0.0, 1.0)
                        tet_vertex_level[v_idx] = t * max_level

        # Classify each edge based on level difference
        edges = self.soft_body.edges
        n_edges = len(edges)
        cross_contour_mask = np.zeros(n_edges, dtype=bool)
        intra_contour_mask = np.zeros(n_edges, dtype=bool)

        level_threshold = 0.5  # Half a level difference

        for edge_idx, (v_i, v_j) in enumerate(edges):
            level_i = tet_vertex_level[v_i]
            level_j = tet_vertex_level[v_j]

            if level_i < 0 or level_j < 0:
                continue

            level_diff = abs(level_i - level_j)

            if level_diff < level_threshold:
                intra_contour_mask[edge_idx] = True
            else:
                cross_contour_mask[edge_idx] = True

        n_cross = np.sum(cross_contour_mask)
        n_intra = np.sum(intra_contour_mask)
        n_neither = n_edges - n_cross - n_intra

        print(f"  Edge classification: {n_cross} cross-contour, {n_intra} intra-contour, {n_neither} unclassified")

        return cross_contour_mask, intra_contour_mask

    def compute_outward_directions(self):
        """
        Compute outward direction for each tet vertex.
        Outward = perpendicular to muscle axis, pointing away from axis center.

        Returns:
            Nx3 array of unit vectors pointing outward
        """
        if self.soft_body is None:
            return None

        tet_verts = np.array(self.tet_vertices)
        n_verts = len(tet_verts)
        outward_dirs = np.zeros((n_verts, 3), dtype=np.float64)

        fixed_indices = list(self.soft_body_fixed_vertices) if hasattr(self, 'soft_body_fixed_vertices') else []

        if len(fixed_indices) < 2:
            print("  Outward directions: not enough fixed vertices")
            return None

        # Get origin and insertion centers using contour level
        if hasattr(self, 'vertex_contour_level') and self.vertex_contour_level is not None:
            max_level = np.max(self.vertex_contour_level)
            origin_verts = []
            insertion_verts = []

            tet_vertex_level = np.full(n_verts, -1.0, dtype=np.float32)
            if hasattr(self, 'contour_to_tet_indices'):
                for contour_idx, tet_idx in enumerate(self.contour_to_tet_indices):
                    if tet_idx >= 0 and tet_idx < n_verts:
                        tet_vertex_level[tet_idx] = self.vertex_contour_level[contour_idx]

            for fi in fixed_indices:
                level = tet_vertex_level[fi]
                if level >= 0:
                    if level < max_level / 2:
                        origin_verts.append(tet_verts[fi])
                    else:
                        insertion_verts.append(tet_verts[fi])

            if len(origin_verts) > 0 and len(insertion_verts) > 0:
                origin_center = np.mean(origin_verts, axis=0)
                insertion_center = np.mean(insertion_verts, axis=0)
            else:
                origin_center = tet_verts[fixed_indices[0]]
                insertion_center = tet_verts[fixed_indices[-1]]
        else:
            origin_center = tet_verts[fixed_indices[0]]
            insertion_center = tet_verts[fixed_indices[-1]]

        muscle_axis = insertion_center - origin_center
        axis_len = np.linalg.norm(muscle_axis)
        if axis_len < 1e-6:
            print("  Outward directions: muscle axis too short")
            return None

        muscle_axis_norm = muscle_axis / axis_len

        for v_idx in range(n_verts):
            pos = tet_verts[v_idx]
            to_vertex = pos - origin_center
            t = np.dot(to_vertex, muscle_axis_norm)
            t = np.clip(t, 0, axis_len)
            axis_point = origin_center + t * muscle_axis_norm

            outward = pos - axis_point
            outward_len = np.linalg.norm(outward)

            if outward_len > 1e-8:
                outward_dirs[v_idx] = outward / outward_len
            else:
                if abs(muscle_axis_norm[0]) < 0.9:
                    perp = np.cross(muscle_axis_norm, [1, 0, 0])
                else:
                    perp = np.cross(muscle_axis_norm, [0, 1, 0])
                perp_len = np.linalg.norm(perp)
                if perp_len > 1e-8:
                    outward_dirs[v_idx] = perp / perp_len
                else:
                    outward_dirs[v_idx] = [1, 0, 0]

        print(f"  Computed outward directions for {n_verts} vertices")
        return outward_dirs

    def update_contour_mesh_from_tet(self):
        """
        Update contour mesh vertices from deformed tet vertices.
        Call this after soft body simulation to sync the visualization.
        """
        if not hasattr(self, 'contour_to_tet_indices') or self.contour_to_tet_indices is None:
            self.build_contour_to_tet_mapping()

        if not hasattr(self, 'contour_to_tet_indices') or self.contour_to_tet_indices is None:
            return

        if self.tet_vertices is None:
            return

        updated = 0
        for i, tet_idx in enumerate(self.contour_to_tet_indices):
            if tet_idx >= 0 and tet_idx < len(self.tet_vertices):
                self.contour_mesh_vertices[i] = self.tet_vertices[tet_idx]
                updated += 1

        if updated > 0:
            self._compute_contour_mesh_normals()
            print(f"Updated {updated} contour mesh vertices from tet")

    # ========================================================================
    # Soft Body Simulation Methods
    # ========================================================================

    def run_soft_body_to_convergence(self, skeleton_meshes, skeleton, max_iterations=500, tolerance=1e-6,
                                    enable_collision=False, collision_margin=0.002,
                                    collision_mesh_override=None, verbose=True, fast_collision=False,
                                    use_arap=False):
        """
        Run quasistatic soft body simulation until convergence.
        Uses LBS for initial positioning, then volume-preserving relaxation.
        Optional post-process collision resolution.

        Args:
            skeleton_meshes: Dict of skeleton name -> MeshLoader
            skeleton: DART skeleton object
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (max vertex displacement)
            enable_collision: Whether to resolve collisions after convergence
            collision_margin: Distance to push vertices outside skeleton (meters)
            collision_mesh_override: Optional list of trimesh objects to use instead
            verbose: Whether to print progress messages
            fast_collision: IGNORED (reserved for future use)
            use_arap: If True, use ARAP for Phase 2 (shape-preserving deformation)

        Returns:
            (iterations, residual): Number of iterations and final residual
        """
        if self.soft_body is None:
            print("Soft body not initialized. Call init_soft_body() first.")
            return 0, 0.0

        # Update parameters
        self.soft_body.update_parameters(
            stiffness=self.soft_body_stiffness,
            damping=self.soft_body_damping,
            volume_stiffness=self.soft_body_volume_stiffness
        )

        if verbose:
            print(f"Running soft body simulation (skeleton binding + contour-aware springs)...")
            print(f"  Local anchors: {len(getattr(self, 'soft_body_local_anchors', {}))}")

        # === STEP 1: Move all vertices based on skeleton bindings ===
        skeleton_moved = False
        if hasattr(self, 'tet_skeleton_bindings') and self.tet_skeleton_bindings is not None:
            updated = self._update_tet_positions_from_skeleton(skeleton)
            skeleton_moved = (updated > 0)
            if verbose:
                print(f"  Moved {updated} vertices based on skeleton bindings")
        else:
            # Fallback to old LBS approach
            self._apply_simple_initial_transform(skeleton)
            if verbose:
                print(f"  Applied LBS initialization to all vertices")

        # === STEP 2: Update fixed vertex targets (origins/insertions) ===
        self._update_fixed_targets_from_skeleton(skeleton_meshes, skeleton)

        if verbose and self.soft_body.fixed_targets is not None:
            rest_fixed = self.soft_body.rest_positions[self.soft_body.fixed_indices]
            diff = np.linalg.norm(self.soft_body.fixed_targets - rest_fixed, axis=1)
            max_diff = np.max(diff)
            print(f"  Fixed vertex displacement: max={max_diff:.6f}, mean={np.mean(diff):.6f}")
            if max_diff > 1e-6:
                skeleton_moved = True

        # === STEP 3: Prepare collision meshes (if enabled) ===
        collision_trimeshes = []
        collision_in_arap = False
        if enable_collision:
            if verbose:
                print(f"  Collision enabled, margin={collision_margin:.4f}")
            if collision_mesh_override is not None:
                collision_trimeshes = collision_mesh_override
                if verbose:
                    print(f"  Using {len(collision_trimeshes)} override meshes")
            elif skeleton_meshes is not None and skeleton is not None:
                collision_trimeshes = self._build_transformed_collision_meshes(
                    skeleton_meshes, skeleton, verbose=verbose
                )
            # Also add DART body shape collision meshes (BoxShape primitives)
            if skeleton is not None:
                dart_shape_meshes = self._build_dart_shape_collision_meshes(skeleton, verbose=verbose)
                collision_trimeshes.extend(dart_shape_meshes)

                if verbose and len(collision_trimeshes) > 0:
                    all_bone_verts = np.vstack([m.vertices for m in collision_trimeshes])
                    bone_min = np.min(all_bone_verts, axis=0)
                    bone_max = np.max(all_bone_verts, axis=0)
                    print(f"  Collision: bone bbox = [{bone_min[0]:.3f},{bone_min[1]:.3f},{bone_min[2]:.3f}] to [{bone_max[0]:.3f},{bone_max[1]:.3f},{bone_max[2]:.3f}]")
                    muscle_min = np.min(self.soft_body.positions, axis=0)
                    muscle_max = np.max(self.soft_body.positions, axis=0)
                    print(f"  Collision: muscle bbox = [{muscle_min[0]:.3f},{muscle_min[1]:.3f},{muscle_min[2]:.3f}] to [{muscle_max[0]:.3f},{muscle_max[1]:.3f},{muscle_max[2]:.3f}]")

        # === STEP 4: Two-phase relaxation ===
        phase1_iters = max(max_iterations // 3, 50)
        effective_margin = collision_margin * 2

        if use_arap:
            phase2_iters = min(15, max(max_iterations // 10, 5))
            arap_collision_meshes = collision_trimeshes if len(collision_trimeshes) > 0 else None
            if arap_collision_meshes is not None:
                collision_in_arap = True
                if verbose:
                    print(f"  Collision integrated into ARAP loop")

            # Check for hybrid simulation mode
            simulation_mode = getattr(self, 'simulation_mode', 'arap')
            if simulation_mode == 'hybrid' and hasattr(self, 'waypoints') and len(self.waypoints) > 0:
                # Initialize hybrid mode if needed
                if not hasattr(self.soft_body, 'hybrid_initialized') or not self.soft_body.hybrid_initialized:
                    waypoints_orig = getattr(self, 'waypoints_original', self.waypoints)
                    self.soft_body.init_hybrid_mode(self.waypoints, waypoints_orig)

                # Run phase 1 (mass-spring settling)
                iters1, res1 = self.soft_body.solve_to_convergence(phase1_iters, tolerance)

                # Run phase 2 (hybrid ARAP with volume preservation)
                volume_stiffness = getattr(self, 'hybrid_volume_stiffness', 0.8)
                waypoints_orig = getattr(self, 'waypoints_original', self.waypoints)
                iters2, res2 = self.soft_body.solve_arap_hybrid(
                    self.waypoints, waypoints_orig,
                    max_iterations=phase2_iters,
                    tolerance=tolerance,
                    collision_meshes=arap_collision_meshes,
                    collision_margin=effective_margin,
                    volume_stiffness=volume_stiffness
                )
                phase2_method = "Hybrid ARAP+Volume"
            else:
                iters1, res1, iters2, res2 = self.soft_body.solve_two_phase_arap(
                    phase1_iterations=phase1_iters,
                    phase2_iterations=phase2_iters,
                    tolerance=tolerance,
                    collision_meshes=arap_collision_meshes,
                    collision_margin=effective_margin
                )
                phase2_method = "ARAP"
        else:
            phase2_iters = max_iterations - phase1_iters
            iters1, res1, iters2, res2 = self.soft_body.solve_two_phase(
                phase1_iterations=phase1_iters,
                phase2_iterations=phase2_iters,
                tolerance=tolerance
            )
            phase2_method = "mass-spring"

        iterations = iters1 + iters2
        residual = res2

        if verbose:
            print(f"  Phase 1 (mass-spring): {iters1} iters, residual={res1:.2e}")
            print(f"  Phase 2 ({phase2_method}): {iters2} iters, residual={res2:.2e}")

        # === STEP 5: Post-process collision resolution (if not done in ARAP) ===
        collision_pushed = 0
        if enable_collision and len(collision_trimeshes) > 0 and not collision_in_arap:
            self.soft_body.set_collision_meshes(collision_trimeshes, margin=effective_margin)
            collision_pushed = self.soft_body.resolve_collisions(max_iterations=15)
            if verbose:
                print(f"  Collision: post-process pushed {collision_pushed} vertices (margin={effective_margin:.4f})")

        # Update tetrahedron mesh vertices for rendering
        self.tet_vertices = self.soft_body.get_positions().astype(np.float32)
        if not getattr(self, '_baking_mode', False):
            self._prepare_tet_draw_arrays()

        # === STEP 6: Update waypoints from tets ===
        if getattr(self, 'waypoints_from_tet_sim', True):
            if hasattr(self, 'waypoints') and len(self.waypoints) > 0:
                # Choose waypoint update method
                use_mvc = getattr(self, 'use_mvc_waypoint_update', False)

                if use_mvc:
                    # MVC-based: recompute waypoints from deformed contours
                    # Build contour mapping if not already done
                    if not hasattr(self, 'contour_to_tet_mapping') or self.contour_to_tet_mapping is None:
                        self.build_contour_vertex_mapping()
                    # Recompute waypoints using MVC on deformed contours
                    self.recompute_waypoints_from_deformed_contours()
                else:
                    # Barycentric-based: interpolate waypoints from tet deformation
                    if not hasattr(self, 'waypoint_bary_coords') or len(self.waypoint_bary_coords) == 0:
                        self._compute_waypoint_barycentric_coords(skeleton_meshes, skeleton)
                    if hasattr(self, 'waypoint_bary_coords') and len(self.waypoint_bary_coords) > 0:
                        self._update_waypoints_from_tet(skeleton)

        return iterations, residual

    def _smooth_soft_body_positions(self, iterations=3, factor=0.3):
        """
        Apply Laplacian smoothing to soft body positions to remove zigzag artifacts.
        Only smooths free vertices, preserves fixed vertices.
        """
        if self.soft_body is None:
            return

        positions = self.soft_body.positions.copy()
        edges = self.soft_body.edges
        free_mask = self.soft_body.free_mask

        # Build adjacency list
        num_verts = len(positions)
        neighbors = [[] for _ in range(num_verts)]
        for i, j in edges:
            neighbors[i].append(j)
            neighbors[j].append(i)

        for _ in range(iterations):
            new_positions = positions.copy()

            for v_idx in range(num_verts):
                if not free_mask[v_idx]:  # Skip fixed vertices
                    continue
                if len(neighbors[v_idx]) == 0:
                    continue

                neighbor_positions = positions[neighbors[v_idx]]
                centroid = np.mean(neighbor_positions, axis=0)
                new_positions[v_idx] = positions[v_idx] + factor * (centroid - positions[v_idx])

            positions = new_positions

        self.soft_body.positions = positions

    def reset_soft_body(self):
        """Reset soft body simulation to rest state."""
        if self.soft_body is not None:
            self.soft_body.reset()
            self.tet_vertices = self.soft_body.get_positions().astype(np.float32)
            self._prepare_tet_draw_arrays()
            # Reset waypoints from original
            if hasattr(self, 'waypoints_original') and self.waypoints_original is not None:
                self.waypoints = [
                    [wp.copy() for wp in group]
                    for group in self.waypoints_original
                ]
            # Reset contour mesh vertices if available
            if hasattr(self, 'contour_mesh_vertices_original') and self.contour_mesh_vertices_original is not None:
                self.contour_mesh_vertices = self.contour_mesh_vertices_original.copy()

    # ========== VIPER Rod Simulation Methods ==========

    def init_viper(self, skeleton_meshes=None, skeleton=None):
        """
        Initialize VIPER rod simulation from waypoints.
        Alternative to tet-based soft body simulation.

        Args:
            skeleton_meshes: Dict of skeleton name -> MeshLoader (for body name lookup)
            skeleton: DART skeleton object (for initial body transforms)
        """
        if not self.viper_available:
            print("VIPER not available (requires Taichi)")
            return False

        if not hasattr(self, 'waypoints') or len(self.waypoints) == 0:
            print("No waypoints available. Generate waypoints first.")
            return False

        # Convert waypoints from [stream][level] = array(num_fibers, 3)
        # to [fiber] = list of 3D points along the fiber
        viper_waypoints = []
        for stream_idx, stream in enumerate(self.waypoints):
            if len(stream) == 0:
                continue
            # Get number of fibers from first level
            first_level = stream[0]
            if isinstance(first_level, np.ndarray) and first_level.ndim == 2:
                num_fibers = first_level.shape[0]
            else:
                # Fallback: treat stream as single fiber
                viper_waypoints.append([np.array(pt) for pt in stream])
                continue

            # Create one rod per fiber, collecting points across levels
            for fiber_idx in range(num_fibers):
                fiber_points = []
                for level in stream:
                    if isinstance(level, np.ndarray) and level.ndim == 2 and fiber_idx < level.shape[0]:
                        fiber_points.append(level[fiber_idx].copy())
                if len(fiber_points) >= 2:
                    viper_waypoints.append(fiber_points)

        if len(viper_waypoints) == 0:
            print("No valid fibers found in waypoints")
            return False

        # Store the converted waypoints for visualization
        self.viper_waypoints = viper_waypoints

        # Compute rest radii from contour areas for each rod vertex
        # radius = sqrt(contour_area / pi) gives equivalent circular cross-section
        self.viper_rest_radii = []
        rod_idx = 0
        for stream_idx, stream in enumerate(self.waypoints):
            if len(stream) == 0:
                continue

            # Get contour areas for this stream
            contour_areas = []
            if hasattr(self, 'contours') and self.contours is not None and stream_idx < len(self.contours):
                stream_contours = self.contours[stream_idx]
                for contour in stream_contours:
                    if contour is not None and len(contour) >= 3:
                        # Compute polygon area using shoelace formula (project to best-fit plane)
                        pts = np.array(contour)
                        # Use PCA to get 2D projection
                        mean = np.mean(pts, axis=0)
                        centered = pts - mean
                        _, _, Vt = np.linalg.svd(centered)
                        pts_2d = centered @ Vt[:2].T
                        # Shoelace formula
                        x, y = pts_2d[:, 0], pts_2d[:, 1]
                        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        contour_areas.append(area)
                    else:
                        contour_areas.append(0.0)

            # Get number of fibers
            first_level = stream[0]
            if isinstance(first_level, np.ndarray) and first_level.ndim == 2:
                num_fibers = first_level.shape[0]
            else:
                num_fibers = 1

            # Create rest radii for each rod in this stream
            for fiber_idx in range(num_fibers):
                rod_radii = []
                for level_idx in range(len(stream)):
                    if level_idx < len(contour_areas) and contour_areas[level_idx] > 0:
                        # Divide by number of fibers to get per-fiber area
                        fiber_area = contour_areas[level_idx] / num_fibers
                        radius = np.sqrt(fiber_area / np.pi)
                    else:
                        radius = self.viper_rod_radius  # Fallback to default
                    rod_radii.append(radius)
                if len(rod_radii) >= 2:
                    self.viper_rest_radii.append(rod_radii)
                    rod_idx += 1

        # Store endpoint attachment info for skeleton binding
        # Maps each rod to its origin and insertion body names
        # If skeleton provided, compute bone-local positions for proper transform
        self.viper_endpoint_attachments = []
        rod_idx = 0
        for stream_idx, stream in enumerate(self.waypoints):
            if len(stream) == 0:
                continue

            # Get body names for this stream from attach_skeleton_names
            origin_body_name = None
            insertion_body_name = None
            if hasattr(self, 'attach_skeleton_names') and stream_idx < len(self.attach_skeleton_names):
                names = self.attach_skeleton_names[stream_idx]
                if len(names) >= 2:
                    origin_body_name = names[0] if names[0] else None
                    insertion_body_name = names[1] if names[1] else None

            # Count rods in this stream
            first_level = stream[0]
            if isinstance(first_level, np.ndarray) and first_level.ndim == 2:
                num_fibers = first_level.shape[0]
            else:
                num_fibers = 1

            # Store attachment for each rod in this stream
            for fiber_idx in range(num_fibers):
                if rod_idx < len(viper_waypoints):
                    rod = viper_waypoints[rod_idx]
                    origin_world_pos = np.array(rod[0], dtype=np.float64)
                    insertion_world_pos = np.array(rod[-1], dtype=np.float64)

                    # Convert world positions to bone-local positions if skeleton available
                    origin_local_pos = origin_world_pos.copy()
                    insertion_local_pos = insertion_world_pos.copy()

                    if skeleton is not None:
                        # Origin: transform world pos to bone-local
                        if origin_body_name:
                            try:
                                body_node = skeleton.getBodyNode(origin_body_name)
                                if body_node is not None:
                                    T = body_node.getWorldTransform()
                                    R = T.rotation()
                                    t = T.translation()
                                    # local = R^T @ (world - t)
                                    origin_local_pos = R.T @ (origin_world_pos - t)
                            except:
                                pass

                        # Insertion: transform world pos to bone-local
                        if insertion_body_name:
                            try:
                                body_node = skeleton.getBodyNode(insertion_body_name)
                                if body_node is not None:
                                    T = body_node.getWorldTransform()
                                    R = T.rotation()
                                    t = T.translation()
                                    insertion_local_pos = R.T @ (insertion_world_pos - t)
                            except:
                                pass

                    self.viper_endpoint_attachments.append({
                        'rod_idx': rod_idx,
                        'origin_body': origin_body_name,
                        'insertion_body': insertion_body_name,
                        'origin_local_pos': origin_local_pos,
                        'insertion_local_pos': insertion_local_pos,
                    })
                    rod_idx += 1

        # Create VIPER simulation
        self.viper_sim = ViperRodSimulation(use_gpu=True)
        # Quasistatic parameters (like ARAP) - no dynamics, just constraint solving
        self.viper_sim.damping = 0.0  # No momentum - pure position-based
        self.viper_sim.gravity = np.array([0.0, 0.0, 0.0])  # No gravity
        self.viper_sim.iterations = 20  # More iterations for better convergence
        self.viper_sim.stretch_stiffness = 1.0  # Strong stretch constraint
        self.viper_sim.volume_stiffness = 0.9  # Strong volume preservation
        self.viper_sim.bend_stiffness = 0.3  # Moderate bending (allow some flexibility)

        # Build rods from converted waypoints
        success = self.viper_sim.build_from_waypoints(viper_waypoints, fixed_indices=None)

        if not success:
            print("Failed to build VIPER rods from waypoints")
            self.viper_sim = None
            return False

        # Compute skinning weights from rods to mesh
        if hasattr(self, 'tet_vertices') and self.tet_vertices is not None:
            mesh_verts = np.array(self.tet_vertices, dtype=np.float32)
            self.viper_sim.compute_skinning_weights(mesh_verts, k_neighbors=4)
            self.viper_skinning_computed = True
            self.viper_rest_mesh_vertices = mesh_verts.copy()

        print(f"VIPER initialized: {len(self.viper_waypoints)} rods, {sum(len(rod) for rod in self.viper_waypoints)} total points")
        if self.viper_endpoint_attachments:
            print(f"  Endpoint attachments: origin={self.viper_endpoint_attachments[0].get('origin_body')}, "
                  f"insertion={self.viper_endpoint_attachments[0].get('insertion_body')}")

        # Auto-enable tet mesh visualization if is_draw_viper_mesh is True
        if getattr(self, 'is_draw_viper_mesh', False):
            self.is_draw_tet_mesh = True

        # Generate initial VIPER rod mesh
        self.update_viper_rod_mesh()

        return True

    def update_viper_endpoints_from_skeleton(self, skeleton):
        """
        Update VIPER rod endpoint positions based on current skeleton transforms.
        This should be called before each VIPER step when skeleton moves.

        Args:
            skeleton: DART skeleton object
        """
        if self.viper_sim is None:
            return
        if skeleton is None:
            return
        if not hasattr(self, 'viper_endpoint_attachments') or len(self.viper_endpoint_attachments) == 0:
            return

        origin_targets = []
        insertion_targets = []

        for attach in self.viper_endpoint_attachments:
            rod_idx = attach['rod_idx']

            # Update origin endpoint
            origin_body_name = attach.get('origin_body')
            if origin_body_name:
                try:
                    body_node = skeleton.getBodyNode(origin_body_name)
                    if body_node is not None:
                        world_transform = body_node.getWorldTransform()
                        rotation = world_transform.rotation()
                        translation = world_transform.translation()
                        local_pos = attach['origin_local_pos']
                        new_pos = rotation @ local_pos + translation
                        origin_targets.append((rod_idx, new_pos))
                except Exception as e:
                    pass  # Skip if body node not found

            # Update insertion endpoint
            insertion_body_name = attach.get('insertion_body')
            if insertion_body_name:
                try:
                    body_node = skeleton.getBodyNode(insertion_body_name)
                    if body_node is not None:
                        world_transform = body_node.getWorldTransform()
                        rotation = world_transform.rotation()
                        translation = world_transform.translation()
                        local_pos = attach['insertion_local_pos']
                        new_pos = rotation @ local_pos + translation
                        insertion_targets.append((rod_idx, new_pos))
                except Exception as e:
                    pass  # Skip if body node not found

        # Update VIPER simulation endpoint targets
        if origin_targets or insertion_targets:
            self.viper_sim.set_endpoint_targets(origin_targets, insertion_targets)

    def set_viper_collision_meshes(self, skeleton_meshes):
        """
        Set collision meshes for VIPER simulation from skeleton meshes.

        Args:
            skeleton_meshes: Dict of skeleton mesh objects (from mesh_loader)
        """
        if self.viper_sim is None or skeleton_meshes is None:
            return

        collision_meshes = []
        for name, mesh in skeleton_meshes.items():
            if hasattr(mesh, 'trimesh') and mesh.trimesh is not None:
                collision_meshes.append(mesh.trimesh)
            elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                collision_meshes.append((mesh.vertices, mesh.faces))

        if collision_meshes:
            self.viper_sim.set_collision_meshes(collision_meshes)

    def run_viper_step(self, dt=None, skeleton=None, skeleton_meshes=None):
        """
        Run a single VIPER simulation step.

        Args:
            dt: Time step (uses default if None)
            skeleton: DART skeleton object (for updating endpoints from bone transforms)
            skeleton_meshes: Dict of skeleton mesh objects for collision (optional)

        Returns:
            Maximum displacement for convergence check
        """
        if self.viper_sim is None:
            return 0.0

        # Update endpoints from skeleton BEFORE stepping
        if skeleton is not None:
            self.update_viper_endpoints_from_skeleton(skeleton)

        # Update collision meshes if provided
        if skeleton_meshes is not None and self.viper_sim.enable_collision:
            self.set_viper_collision_meshes(skeleton_meshes)

        max_disp = self.viper_sim.step(dt)

        # Update mesh from rod positions (skinned tet mesh)
        self.update_mesh_from_viper()

        # Update VIPER rod-based mesh
        self.update_viper_rod_mesh()

        # Update waypoints from rod positions
        self.update_waypoints_from_viper()

        return max_disp

    def update_mesh_from_viper(self):
        """Update tet mesh vertices from VIPER rod positions using skinning."""
        if self.viper_sim is None or not self.viper_skinning_computed:
            return

        if not hasattr(self, 'viper_rest_mesh_vertices'):
            return

        # Get deformed mesh positions
        deformed_verts = self.viper_sim.skin_mesh(self.viper_rest_mesh_vertices)

        if deformed_verts is not None:
            self.tet_vertices = deformed_verts.astype(np.float32)
            self._prepare_tet_draw_arrays()

    def reset_viper(self):
        """Reset VIPER simulation to rest state."""
        if self.viper_sim is not None:
            self.viper_sim.reset()
            # Update mesh and waypoints
            self.update_mesh_from_viper()
            self.update_viper_rod_mesh()
            self.update_waypoints_from_viper()

    def update_viper_rod_mesh(self):
        """
        Update the VIPER rod-based mesh from current rod positions and scales.
        This creates a triangle mesh directly from the rod geometry.
        """
        if self.viper_sim is None:
            return

        # Get rest radii from contour areas if available
        rest_radii = getattr(self, 'viper_rest_radii', None)

        # Generate mesh from rods
        vertices, faces, normals = self.viper_sim.generate_rod_mesh(
            rest_radii=rest_radii,
            slices=12
        )

        if vertices is not None:
            self.viper_mesh_vertices = vertices
            self.viper_mesh_faces = faces
            self.viper_mesh_normals = normals
            self._prepare_viper_mesh_draw_arrays()

    def _prepare_viper_mesh_draw_arrays(self):
        """Prepare OpenGL draw arrays for VIPER rod mesh."""
        if not hasattr(self, 'viper_mesh_vertices') or self.viper_mesh_vertices is None:
            return

        # Create flattened arrays for drawing
        verts = self.viper_mesh_vertices
        faces = self.viper_mesh_faces
        norms = self.viper_mesh_normals

        # Create per-face vertex arrays
        self.viper_mesh_draw_vertices = verts[faces.flatten()].astype(np.float32)
        self.viper_mesh_draw_normals = norms[faces.flatten()].astype(np.float32)

    def run_viper_to_convergence(self, max_iterations=500, tolerance=1e-6, skeleton=None, skeleton_meshes=None):
        """
        Run VIPER simulation until convergence or max iterations.

        Args:
            max_iterations: Maximum number of steps
            tolerance: Convergence threshold for max displacement
            skeleton: DART skeleton object (for updating endpoints from bone transforms)
            skeleton_meshes: Dict of skeleton mesh objects for collision (optional)

        Returns:
            Number of iterations taken
        """
        if self.viper_sim is None:
            print("VIPER not initialized")
            return 0

        for i in range(max_iterations):
            max_disp = self.run_viper_step(skeleton=skeleton, skeleton_meshes=skeleton_meshes)

            if max_disp < tolerance:
                print(f"VIPER converged in {i+1} iterations (max_disp={max_disp:.2e})")
                return i + 1

        print(f"VIPER reached max iterations ({max_iterations}), max_disp={max_disp:.2e}")
        return max_iterations

    def set_viper_target_position(self, rod_idx, vertex_idx, target_pos):
        """
        Set target position for a VIPER rod vertex (for bone attachment).

        Args:
            rod_idx: Index of the rod
            vertex_idx: Index of vertex within the rod
            target_pos: Target position (3D array)
        """
        if self.viper_sim is None:
            return

        # Convert local indices to global index
        global_idx = 0
        for i in range(rod_idx):
            if i < len(self.waypoints):
                global_idx += len(self.waypoints[i])
        global_idx += vertex_idx

        self.viper_sim.update_fixed_position(global_idx, np.array(target_pos, dtype=np.float32))

    # ========== End VIPER Methods ==========

    # ========== Contour Cutting Methods ==========

    def _cut_contour_voronoi(self, contour, contour_match, parent_means, basis_z, smooth_window=5):
        """
        Cut a contour into multiple streams using Voronoi-style assignment with smoothing.

        Args:
            contour: list of 3D points forming the contour
            contour_match: list of (P, Q) pairs from bounding plane matching
            parent_means: list of 3D mean points for each parent stream
            basis_z: normal vector of the contour plane
            smooth_window: window size for smoothing assignments

        Returns:
            new_contours: list of contour point lists, one per parent stream
        """
        n = len(contour)
        num_streams = len(parent_means)

        if num_streams < 2:
            return [list(contour)]

        if n == 0 or len(contour_match) == 0:
            # Return copies of the nearest parent mean as fallback
            return [[parent_means[s].copy()] for s in range(num_streams)]

        # Project parent means onto the contour plane
        centroid = np.mean(contour, axis=0)
        projected_parents = [p - np.dot(p - centroid, basis_z) * basis_z for p in parent_means]

        # Initial assignment: each point to nearest parent stream
        assignments = []
        for point in contour:
            distances = [np.linalg.norm(point - pp) for pp in projected_parents]
            assignments.append(np.argmin(distances))

        # Smooth assignments to remove zig-zag using majority voting
        smoothed = self._smooth_assignments_circular(assignments, smooth_window)

        # Find transition points and build contours for each stream
        new_contours = [[] for _ in range(num_streams)]
        Ps = [pair[0] for pair in contour_match]

        prev_assignment = smoothed[-1]
        for i in range(n):
            curr_assignment = smoothed[i]

            # If transition between streams, add midpoint to both
            if prev_assignment != curr_assignment:
                prev_P = Ps[i - 1]
                curr_P = Ps[i]
                mid_P = (prev_P + curr_P) / 2

                new_contours[prev_assignment].append(mid_P)
                new_contours[curr_assignment].append(mid_P)

            new_contours[curr_assignment].append(Ps[i])
            prev_assignment = curr_assignment

        # Ensure no stream is empty - if a stream has no points, assign closest contour points
        for stream_idx in range(num_streams):
            if len(new_contours[stream_idx]) == 0:
                # Find the closest points from the contour to this stream's parent mean
                parent_mean = projected_parents[stream_idx]
                distances = [np.linalg.norm(np.array(p) - parent_mean) for p in Ps]
                closest_idx = np.argmin(distances)
                # Add a small segment around the closest point
                new_contours[stream_idx].append(Ps[closest_idx])
                if closest_idx > 0:
                    new_contours[stream_idx].insert(0, Ps[closest_idx - 1])
                if closest_idx < len(Ps) - 1:
                    new_contours[stream_idx].append(Ps[closest_idx + 1])
                print(f"Warning: Stream {stream_idx} had no points, added fallback points")

        return new_contours

    def _smooth_assignments_circular(self, assignments, window_size):
        """
        Smooth stream assignments using circular majority voting.
        Removes isolated assignments that cause zig-zag patterns.

        Args:
            assignments: list of stream indices for each contour point
            window_size: size of voting window (should be odd)

        Returns:
            smoothed assignments list
        """
        n = len(assignments)
        if n == 0:
            return assignments

        smoothed = list(assignments)
        half_window = window_size // 2

        # Multiple passes for better smoothing
        for _ in range(3):
            new_smoothed = list(smoothed)
            for i in range(n):
                # Gather votes from circular window
                votes = {}
                for j in range(-half_window, half_window + 1):
                    idx = (i + j) % n
                    stream = smoothed[idx]
                    votes[stream] = votes.get(stream, 0) + 1

                # Majority vote
                new_smoothed[i] = max(votes.keys(), key=lambda k: votes[k])
            smoothed = new_smoothed

        return smoothed

    def _cut_contour_angular(self, contour, contour_match, parent_means, basis_z):
        """
        Cut a contour into multiple streams using angular/radial sectors from centroid.

        Each stream gets a pie-slice shaped sector based on direction to its parent mean.
        This produces clean, non-zig-zagged cuts.

        Args:
            contour: list of 3D points forming the contour
            contour_match: list of (P, Q) pairs from bounding plane matching
            parent_means: list of 3D mean points for each parent stream
            basis_z: normal vector of the contour plane

        Returns:
            new_contours: list of contour point lists, one per parent stream
        """
        n = len(contour)
        num_streams = len(parent_means)

        if num_streams < 2:
            return [list(contour)]

        if n == 0 or len(contour_match) == 0:
            return [[parent_means[s].copy()] for s in range(num_streams)]

        # Project parent means onto the contour plane
        centroid = np.mean(contour, axis=0)
        projected_parents = [p - np.dot(p - centroid, basis_z) * basis_z for p in parent_means]

        # Build local 2D coordinate system on the plane
        # basis_x: direction to first parent mean
        ref_dir = projected_parents[0] - centroid
        ref_norm = np.linalg.norm(ref_dir)
        if ref_norm < 1e-10:
            # Fallback: use arbitrary direction perpendicular to basis_z
            basis_x = np.array([1, 0, 0]) - np.dot([1, 0, 0], basis_z) * basis_z
            if np.linalg.norm(basis_x) < 1e-10:
                basis_x = np.array([0, 1, 0]) - np.dot([0, 1, 0], basis_z) * basis_z
            basis_x = basis_x / np.linalg.norm(basis_x)
        else:
            basis_x = ref_dir / ref_norm
        basis_y = np.cross(basis_z, basis_x)

        # Calculate angle for each parent mean from centroid
        parent_angles = []
        for pp in projected_parents:
            direction = pp - centroid
            x_component = np.dot(direction, basis_x)
            y_component = np.dot(direction, basis_y)
            angle = np.arctan2(y_component, x_component)
            parent_angles.append(angle)

        # Sort streams by angle
        stream_order = np.argsort(parent_angles)
        sorted_angles = [parent_angles[i] for i in stream_order]

        # Calculate boundary angles (midpoints between consecutive parent angles)
        boundary_angles = []
        for i in range(num_streams):
            angle_a = sorted_angles[i]
            angle_b = sorted_angles[(i + 1) % num_streams]

            # Handle wrap-around
            if angle_b < angle_a:
                angle_b += 2 * np.pi

            mid_angle = (angle_a + angle_b) / 2
            if mid_angle > np.pi:
                mid_angle -= 2 * np.pi
            boundary_angles.append(mid_angle)

        # Assign each contour point to a stream based on its angle
        Ps = [pair[0] for pair in contour_match]
        new_contours = [[] for _ in range(num_streams)]

        prev_stream = None
        for i, P in enumerate(Ps):
            direction = P - centroid
            x_component = np.dot(direction, basis_x)
            y_component = np.dot(direction, basis_y)
            point_angle = np.arctan2(y_component, x_component)

            # Find which sector this point belongs to
            assigned_stream = None
            for j in range(num_streams):
                # Get boundary range for stream at sorted position j
                start_boundary = boundary_angles[(j - 1) % num_streams]
                end_boundary = boundary_angles[j]

                # Normalize angle for comparison
                test_angle = point_angle

                # Handle wrap-around cases
                if start_boundary > end_boundary:
                    # Sector crosses -pi/pi boundary
                    if test_angle >= start_boundary or test_angle < end_boundary:
                        assigned_stream = stream_order[j]
                        break
                else:
                    if start_boundary <= test_angle < end_boundary:
                        assigned_stream = stream_order[j]
                        break

            if assigned_stream is None:
                # Fallback: assign to nearest parent
                distances = [np.linalg.norm(P - pp) for pp in projected_parents]
                assigned_stream = np.argmin(distances)

            # Add transition midpoint if stream changes
            if prev_stream is not None and prev_stream != assigned_stream:
                prev_P = Ps[i - 1]
                mid_P = (prev_P + P) / 2
                new_contours[prev_stream].append(mid_P)
                new_contours[assigned_stream].append(mid_P)

            new_contours[assigned_stream].append(P)
            prev_stream = assigned_stream

        # Ensure no stream is empty
        for stream_idx in range(num_streams):
            if len(new_contours[stream_idx]) == 0:
                parent_mean = projected_parents[stream_idx]
                distances = [np.linalg.norm(np.array(p) - parent_mean) for p in Ps]
                closest_idx = np.argmin(distances)
                new_contours[stream_idx].append(Ps[closest_idx])
                if closest_idx > 0:
                    new_contours[stream_idx].insert(0, Ps[closest_idx - 1])
                if closest_idx < len(Ps) - 1:
                    new_contours[stream_idx].append(Ps[closest_idx + 1])
                print(f"Warning: Stream {stream_idx} had no points in angular cutting, added fallback")

        return new_contours

    def _cut_contour_gradient(self, contour, contour_match, parent_means, basis_z):
        """
        Cut a contour by finding natural transition points where nearest-parent changes.

        This method:
        1. Computes distance from each contour point to each parent mean
        2. Finds transition points where the nearest parent changes
        3. Uses these natural boundaries as cut points
        4. Guarantees contiguous segments (no zig-zag)

        Args:
            contour: list of 3D points forming the contour
            contour_match: list of (P, Q) pairs from bounding plane matching
            parent_means: list of 3D mean points for each parent stream
            basis_z: normal vector of the contour plane

        Returns:
            new_contours: list of contour point lists, one per parent stream
        """
        n = len(contour)
        num_streams = len(parent_means)

        if num_streams < 2:
            return [list(contour)]

        if n == 0 or len(contour_match) == 0:
            return [[parent_means[s].copy()] for s in range(num_streams)]

        Ps = [pair[0] for pair in contour_match]
        n_pts = len(Ps)

        # Project parent means onto the contour plane
        centroid = np.mean(contour, axis=0)
        projected_parents = [p - np.dot(p - centroid, basis_z) * basis_z for p in parent_means]

        # Compute distance from each point to each parent
        distances = np.zeros((n_pts, num_streams))
        for i, P in enumerate(Ps):
            for s, pp in enumerate(projected_parents):
                distances[i, s] = np.linalg.norm(np.array(P) - pp)

        # Find nearest parent for each point
        nearest = np.argmin(distances, axis=1)

        # Find all transition points (where nearest parent changes)
        transitions = []
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            if nearest[i] != nearest[next_i]:
                # Transition between i and next_i
                # Store: (index, from_stream, to_stream)
                transitions.append((i, nearest[i], nearest[next_i]))

        # If no transitions (all points nearest to same parent), assign all to that parent
        if len(transitions) == 0:
            new_contours = [[] for _ in range(num_streams)]
            dominant = nearest[0]
            new_contours[dominant] = list(Ps)
            # Give other streams at least one point
            for s in range(num_streams):
                if s != dominant and len(new_contours[s]) == 0:
                    dist_to_s = distances[:, s]
                    closest = np.argmin(dist_to_s)
                    new_contours[s].append(Ps[closest])
            return new_contours

        # Build contours by following transitions
        # Start from first transition and collect points for each stream
        new_contours = [[] for _ in range(num_streams)]

        # Sort transitions by index
        transitions.sort(key=lambda x: x[0])

        # Each segment goes from one transition to the next
        for t_idx in range(len(transitions)):
            start_trans = transitions[t_idx]
            end_trans = transitions[(t_idx + 1) % len(transitions)]

            start_idx = (start_trans[0] + 1) % n_pts  # First point after transition
            end_idx = end_trans[0]  # Last point before next transition
            stream = start_trans[2]  # Stream we're transitioning TO

            # Collect points in this segment
            if end_idx >= start_idx:
                for i in range(start_idx, end_idx + 1):
                    new_contours[stream].append(Ps[i])
            else:
                # Wrap around
                for i in range(start_idx, n_pts):
                    new_contours[stream].append(Ps[i])
                for i in range(0, end_idx + 1):
                    new_contours[stream].append(Ps[i])

            # Add transition midpoint to both streams for smooth connection
            mid_P = (Ps[start_trans[0]] + Ps[(start_trans[0] + 1) % n_pts]) / 2
            if len(new_contours[start_trans[1]]) > 0:
                new_contours[start_trans[1]].append(mid_P)
            new_contours[stream].insert(0, mid_P)

        # Ensure no stream is empty
        for stream_idx in range(num_streams):
            if len(new_contours[stream_idx]) == 0:
                parent_mean = projected_parents[stream_idx]
                dist_to_stream = [np.linalg.norm(np.array(p) - parent_mean) for p in Ps]
                closest_idx = np.argmin(dist_to_stream)
                new_contours[stream_idx].append(Ps[closest_idx])
                print(f"Warning: Stream {stream_idx} had no points in gradient cutting, added fallback")

        return new_contours

    def _cut_contour_ratio(self, contour, contour_match, parent_means, basis_z, target_ratios, stream_order):
        """
        Cut a contour into segments proportional to target ratios.

        This method:
        1. Orders streams spatially along the contour using parent means
        2. Divides the contour into segments of specified proportions
        3. Guarantees contiguous segments (no zig-zag)

        Args:
            contour: list of 3D points forming the contour
            contour_match: list of (P, Q) pairs from bounding plane matching
            parent_means: list of 3D mean points for each parent stream
            basis_z: normal vector of the contour plane
            target_ratios: list of target proportions for each stream (should sum to 1)
            stream_order: list of stream indices in spatial order

        Returns:
            new_contours: list of contour point lists, one per parent stream
        """
        n = len(contour)
        num_streams = len(parent_means)

        if num_streams < 2:
            return [list(contour)]

        if n == 0 or len(contour_match) == 0:
            return [[parent_means[s].copy()] for s in range(num_streams)]

        Ps = [pair[0] for pair in contour_match]
        n_pts = len(Ps)

        # Project parent means onto the contour plane
        centroid = np.mean(contour, axis=0)
        projected_parents = [p - np.dot(p - centroid, basis_z) * basis_z for p in parent_means]

        # Find starting point: closest point to first stream's parent mean
        first_stream = stream_order[0]
        first_parent = projected_parents[first_stream]
        distances = [np.linalg.norm(np.array(P) - first_parent) for P in Ps]
        start_idx = np.argmin(distances)

        # Reorder Ps to start from start_idx
        reordered_indices = [(start_idx + i) % n_pts for i in range(n_pts)]

        # Calculate cumulative ratios for cut points
        cumul_ratios = np.cumsum(target_ratios)
        cumul_ratios[-1] = 1.0  # Ensure last is exactly 1

        # Calculate cut indices based on ratios
        cut_indices = []
        for i in range(num_streams - 1):
            cut_idx = int(cumul_ratios[i] * n_pts)
            cut_indices.append(min(cut_idx, n_pts - 1))

        # Build contours for each stream
        new_contours = [[] for _ in range(num_streams)]

        prev_cut = 0
        for i, stream_idx in enumerate(stream_order):
            if i < num_streams - 1:
                next_cut = cut_indices[i]
            else:
                next_cut = n_pts

            # Collect points for this stream
            for j in range(prev_cut, next_cut):
                original_idx = reordered_indices[j]
                new_contours[stream_idx].append(Ps[original_idx])

            # Add midpoint at cut boundary for smooth transition
            if i < num_streams - 1 and next_cut < n_pts and prev_cut < n_pts:
                idx_before = reordered_indices[next_cut - 1] if next_cut > 0 else reordered_indices[0]
                idx_after = reordered_indices[next_cut] if next_cut < n_pts else reordered_indices[0]
                mid_P = (Ps[idx_before] + Ps[idx_after]) / 2
                new_contours[stream_idx].append(mid_P)
                # Next stream also gets this midpoint
                next_stream = stream_order[i + 1]
                new_contours[next_stream].insert(0, mid_P)

            prev_cut = next_cut

        # Ensure no stream is empty
        for stream_idx in range(num_streams):
            if len(new_contours[stream_idx]) == 0:
                parent_mean = projected_parents[stream_idx]
                dist_to_stream = [np.linalg.norm(np.array(p) - parent_mean) for p in Ps]
                closest_idx = np.argmin(dist_to_stream)
                new_contours[stream_idx].append(Ps[closest_idx])
                print(f"Warning: Stream {stream_idx} had no points in ratio cutting, added fallback")

        return new_contours

    def _ensure_contiguous_assignments(self, assignments):
        """
        Ensure each stream's points form a contiguous segment.
        If a stream has multiple disconnected segments, merge them.

        Args:
            assignments: list of stream indices

        Returns:
            cleaned assignments with contiguous segments per stream
        """
        n = len(assignments)
        if n == 0:
            return assignments

        # Find all unique streams
        streams = list(set(assignments))

        # For each stream, find its segments
        stream_segments = {s: [] for s in streams}

        current_stream = assignments[0]
        segment_start = 0

        for i in range(1, n + 1):
            idx = i % n
            if i == n or assignments[idx] != current_stream:
                stream_segments[current_stream].append((segment_start, i - 1))
                if i < n:
                    current_stream = assignments[idx]
                    segment_start = i

        # If any stream has multiple segments, keep only the largest
        result = list(assignments)
        for stream, segments in stream_segments.items():
            if len(segments) > 1:
                # Find largest segment
                largest = max(segments, key=lambda s: s[1] - s[0])
                # Mark other segments for reassignment
                for seg in segments:
                    if seg != largest:
                        for i in range(seg[0], seg[1] + 1):
                            # Assign to nearest other stream
                            if i > 0:
                                result[i] = result[i - 1]
                            elif i < n - 1:
                                result[i] = result[i + 1]

        return result

    # ========== End Contour Cutting Methods ==========

    # ========== Scalar Field Methods ==========

    def reset_process(self):
        """Reset all processing-related variables to initial state."""
        # Scalar field
        self.scalar_field = None
        self.vertex_colors = None
        self.is_draw_scalar_field = False

        # Scalar field animation state
        self._scalar_anim_active = False
        self._scalar_anim_progress = 0.0
        self._scalar_anim_target_colors = None
        self._scalar_anim_normalized_u = None
        self._scalar_replayed = False

        # Clear precomputed face scalar ranges
        if hasattr(self, '_face_scalar_ranges'):
            self._face_scalar_ranges = None
        if hasattr(self, '_face_scalar_min'):
            self._face_scalar_min = None
        if hasattr(self, '_face_scalar_max'):
            self._face_scalar_max = None

        # Contours and bounding planes
        self.contours = None
        self.bounding_planes = []
        self.contours_discarded = None
        self.bounding_planes_discarded = None
        self.draw_contour_stream = None
        self.is_draw_contours = False
        self.is_draw_bounding_box = False

        # Contour animation state
        self._contour_anim_active = False
        self._contour_anim_progress = 0.0
        self._contour_anim_total = 0
        self._contour_anim_original_indices = []
        self._contour_anim_bp_scale = {}
        self._contour_replayed = False
        self._find_contours_count = 0

        # Fill gaps animation state
        self._fill_gaps_inserted_indices = []
        self._fill_gaps_anim_active = False
        self._fill_gaps_anim_progress = 0.0
        self._fill_gaps_anim_step = 0
        self._fill_gaps_replayed = False

        # Animation highlight
        self._anim_highlight_fades = {}

        # Transitions animation state
        self._transitions_inserted_indices = []
        self._transitions_anim_active = False
        self._transitions_anim_progress = 0.0
        self._transitions_anim_step = 0
        self._transitions_replayed = False

        # Smoothing animation state
        self._smooth_anim_active = False
        self._smooth_anim_progress = 0.0
        self._smooth_bp_before = None
        self._smooth_bp_after = None
        self._smooth_swing_data = None
        self._smooth_twist_data = None
        self._smooth_replayed = False

        # Stream smooth animation state (post-cut)
        self._stream_smooth_anim_active = False
        self._stream_smooth_anim_progress = 0.0
        self._stream_smooth_bp_before = None
        self._stream_smooth_bp_after = None
        self._stream_smooth_swing_data = None
        self._stream_smooth_twist_data = None
        self._stream_smooth_num_levels = 0
        self._stream_smooth_replayed = False

        # BP color override during smooth animations
        self._smooth_anim_bp_colors = None

        # Reset transparency
        self.transparency = 1.0
        if self.vertex_colors is not None:
            self.vertex_colors[:, 3] = 1.0

        # Cut animation state
        self._cut_anim_active = False
        self._cut_anim_progress = 0.0
        self._cut_color_before = None
        self._cut_color_after = None
        self._cut_bp_before = None
        self._cut_bp_after = None
        self._cut_anim_contour_colors = None
        self._cut_replayed = False
        self._cut_bp_switched = False
        self._cut_bp_changed_levels = set()
        self._cut_num_levels_before = 0
        self._precut_contours = None
        self._precut_bounding_planes = None
        self._precut_draw_contour_stream = None
        self._cut_anim_deferred_backup = None
        self._smooth_bp_before_level = None
        self._smooth_bp_after_level = None
        self._smooth_swing_data_level = None
        self._smooth_twist_data_level = None
        self._smooth_num_levels_level = 0
        self._smooth_bp_before_stream = None
        self._smooth_bp_after_stream = None
        self._smooth_swing_data_stream = None
        self._smooth_twist_data_stream = None
        self._smooth_num_levels_stream = 0

        # Level select shrink animation state
        self._level_select_anim_active = False
        self._level_select_anim_progress = 0.0
        self._level_select_anim_scales = None
        self._level_select_anim_unselected = None
        self._level_select_anim_num_levels = 0
        self._level_select_replayed = False
        self._level_select_anim_pending_resume = False
        self._level_select_anim_original = None

        # Level selection backup
        self._contours_backup = None
        self._bounding_planes_backup = None
        self.selected_stream_levels = None

        # Smoothening state
        if hasattr(self, '_smoothen_anchor_level'):
            self._smoothen_anchor_level = None

        # Contour mesh
        self.contour_mesh_vertices = None
        self.contour_mesh_faces = None
        self.contour_mesh_normals = None
        self.is_draw_contour_mesh = False

        # Tetrahedron mesh
        self.tet_vertices = None
        self.tet_faces = None  # Backwards compat alias to tet_render_faces
        self.tet_render_faces = None  # Original contour faces for rendering/collision
        self.tet_sim_faces = None  # Tet boundary faces for simulation
        self.tet_tetrahedra = None
        self.tet_cap_face_indices = []
        self.tet_anchor_vertices = []
        self.tet_surface_face_count = 0
        self.is_draw_tet_mesh = False
        self.is_draw_tet_edges = False
        self.tet_cap_attachments = []

        # Soft body
        self.soft_body = None

        # Fiber architecture
        self.fiber_architecture = [self._sobol_sampling_barycentric_default(16)]
        self.is_draw_fiber_architecture = False

        # Stream data
        self.stream_contours = None
        self.stream_bounding_planes = None
        self.stream_groups = None
        self.max_stream_count = None

        # Manual cutting state
        self._manual_cut_pending = False
        self._manual_cut_data = None
        self._manual_cut_line = None

        # Shared cut edge vertices (for tet mesh connectivity)
        self.shared_cut_vertices = []

        # Other
        self.structure_vectors = []
        self.specific_contour = None
        self.normalized_contours = []
        self.waypoints = []

        print("Process reset complete")

    def compute_scalar_field(self, defer=False):
        """Compute scalar field from origin/insertion edge groups.
        If defer=True, save animation data but keep visual state unchanged until replay."""
        origin_indices = []
        insertion_indices = []

        for i, edge_group in enumerate(self.edge_groups):
            if self.edge_classes[i] == 'origin':
                origin_indices += edge_group
            else:
                insertion_indices += edge_group

        u = solve_scalar_field(self.vertices, self.faces_3, origin_indices, insertion_indices)
        self.scalar_field = u
        # Invalidate precomputed face scalar ranges (will be recomputed on next find_contour)
        self._face_scalar_min = None
        self._face_scalar_max = None

        u_min, u_max = np.min(u), np.max(u)
        normalized_u = (u - u_min) / (u_max - u_min) if u_max > u_min else np.zeros_like(u)

        target_colors = COLOR_MAP(1 - normalized_u)[:, :4]
        target_colors = np.array(target_colors, dtype=np.float32)
        target_colors[:, 3] = self.transparency

        # Save animation data for replay
        self._scalar_anim_target_colors = target_colors[self.faces_3[:, :, 0].flatten()]
        self._scalar_anim_normalized_u = normalized_u[self.faces_3[:, :, 0].flatten()]
        self._scalar_anim_active = False
        self._scalar_replayed = False

        if defer:
            # Keep default muscle color, don't show scalar field yet
            self.is_draw_scalar_field = False
        else:
            # Apply colors immediately
            self.vertex_colors = self._scalar_anim_target_colors.copy()
            self.is_draw_scalar_field = True
            self._scalar_replayed = True

    def replay_scalar_animation(self):
        """Start replaying the scalar field color flood animation."""
        if self._scalar_anim_target_colors is None:
            return
        self._scalar_anim_progress = 0.0
        self._scalar_anim_active = True
        self.is_draw_scalar_field = True
        # Reset to default muscle color
        n = len(self._scalar_anim_target_colors)
        self.vertex_colors = np.tile(
            np.array([self.color[0], self.color[1], self.color[2], self.transparency], dtype=np.float32),
            (n, 1)
        )

    def update_scalar_animation(self, dt):
        """Advance the scalar field color reveal animation. Returns True while active."""
        if not self._scalar_anim_active:
            return False

        self._scalar_anim_progress += dt * 0.5  # 2 seconds for full reveal
        if self._scalar_anim_progress >= 1.0:
            self._scalar_anim_progress = 1.0
            self._scalar_anim_active = False
            self._scalar_replayed = True
            self.vertex_colors = self._scalar_anim_target_colors.copy()
            return False

        # Reveal vertices whose normalized scalar value (0=origin, 1=insertion) <= progress
        mask = self._scalar_anim_normalized_u <= self._scalar_anim_progress
        default_color = np.array([self.color[0], self.color[1], self.color[2], self.transparency], dtype=np.float32)
        self.vertex_colors[mask] = self._scalar_anim_target_colors[mask]
        self.vertex_colors[~mask] = default_color
        return True

    # ========== End Scalar Field Methods ==========

    # ========== Bounding Plane Methods ==========

    def save_bounding_planes(self, contour_points, scalar_value, prev_bounding_plane=None, bounding_plane_info_orig=None, use_independent_axes=False):
        """Save bounding plane information for a contour.

        Args:
            use_independent_axes: If True, use farthest vertex pair for axes without
                                  aligning to any previous reference. Used for origin/insertion
                                  contours that should have independent orientations.
        """
        # Import geometric utilities from contour_mesh
        from .contour_mesh import compute_newell_normal, compute_best_fitting_plane, compute_polygon_area, compute_minimum_area_bbox

        # Validate and fix contour_points to ensure it's a proper 2D array with shape (N, 3)
        contour_points = np.asarray(contour_points)
        if contour_points.ndim == 0:
            # 0-dimensional array - create minimal valid contour
            contour_points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]])
        elif contour_points.ndim == 1:
            # 1D array - might be single vertex or flat array
            if len(contour_points) >= 9:
                # Try to reshape as (N, 3)
                n_verts = len(contour_points) // 3
                contour_points = contour_points[:n_verts*3].reshape(n_verts, 3)
            elif len(contour_points) >= 3:
                # Single vertex - duplicate to make valid contour
                v = contour_points[:3]
                contour_points = np.array([v, v + [0.001, 0, 0], v + [0, 0.001, 0]])
            else:
                contour_points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]])
        elif contour_points.ndim == 2:
            if contour_points.shape[1] < 3:
                # Pad columns with zeros
                padded = np.zeros((contour_points.shape[0], 3))
                padded[:, :contour_points.shape[1]] = contour_points
                contour_points = padded
            elif contour_points.shape[1] > 3:
                contour_points = contour_points[:, :3]

        # Ensure at least 3 vertices
        while len(contour_points) < 3:
            contour_points = np.vstack([contour_points, contour_points[-1] + [0.001, 0, 0]])

        has_prev_reference = False  # Track if we have a real previous reference
        if use_independent_axes:
            # Force independent axes - don't use any previous reference
            temp_newell = compute_newell_normal(contour_points)
            prev_basis_z = temp_newell / (np.linalg.norm(temp_newell) + 1e-10)
            # Create arbitrary orthonormal x/y (will be overridden by farthest_vertex)
            arbitrary = np.array([1, 0, 0])
            if abs(np.dot(arbitrary, prev_basis_z)) > 0.9:
                arbitrary = np.array([0, 1, 0])
            prev_basis_x = arbitrary - np.dot(arbitrary, prev_basis_z) * prev_basis_z
            prev_basis_x = prev_basis_x / (np.linalg.norm(prev_basis_x) + 1e-10)
            prev_basis_y = np.cross(prev_basis_z, prev_basis_x)
            prev_newell = prev_basis_z.copy()
            has_prev_reference = False
        elif prev_bounding_plane is not None:
            prev_basis_x = prev_bounding_plane['basis_x']
            prev_basis_y = prev_bounding_plane['basis_y']
            prev_basis_z = prev_bounding_plane['basis_z']
            prev_newell = prev_bounding_plane['newell_normal']
            has_prev_reference = True
        elif len(self.bounding_planes) > 0:
            prev_basis_x = self.bounding_planes[-1][0]['basis_x']
            prev_basis_y = self.bounding_planes[-1][0]['basis_y']
            prev_basis_z = self.bounding_planes[-1][0]['basis_z']
            prev_newell = self.bounding_planes[-1][0]['newell_normal']
            has_prev_reference = True
        else:
            # No reference - compute initial z from contour normal, x/y will be determined later
            temp_newell = compute_newell_normal(contour_points)
            prev_basis_z = temp_newell / (np.linalg.norm(temp_newell) + 1e-10)
            # Create arbitrary orthonormal x/y (will be overridden by farthest_vertex)
            arbitrary = np.array([1, 0, 0])
            if abs(np.dot(arbitrary, prev_basis_z)) > 0.9:
                arbitrary = np.array([0, 1, 0])
            prev_basis_x = arbitrary - np.dot(arbitrary, prev_basis_z) * prev_basis_z
            prev_basis_x = prev_basis_x / (np.linalg.norm(prev_basis_x) + 1e-10)
            prev_basis_y = np.cross(prev_basis_z, prev_basis_x)
            prev_newell = prev_basis_z.copy()

        newell_normal = compute_newell_normal(contour_points)
        mean = np.mean(contour_points, axis=0)

        # Align newell_normal with previous basis_z (not just prev_newell)
        # This ensures z-axis consistency for newly found/refined contours
        if np.dot(newell_normal, prev_basis_z) < 0:
            newell_normal *= -1

        basis_z = newell_normal

        # Compute farthest vertex pair in 3D for all methods
        # This information is useful for visualization regardless of which method is used
        from scipy.spatial.distance import cdist
        farthest_pair = None
        if len(contour_points) > 2:
            dists = cdist(contour_points, contour_points)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            farthest_pair = (contour_points[i].copy(), contour_points[j].copy())

        # Get bounding box method
        bbox_method = getattr(self, 'bounding_box_method', 'farthest_vertex')

        if bounding_plane_info_orig is not None:
            # For cut planes, use bbox method to find optimal orientation
            print(f"  [DEBUG] save_bounding_planes: Using bounding_plane_info_orig path (minimum area bbox)")
            basis_z = bounding_plane_info_orig['basis_z']
            newell_normal = bounding_plane_info_orig['newell_normal']

            # Use bbox method for cut planes
            # Create initial orthonormal basis on the plane
            arbitrary = np.array([1, 0, 0])
            if abs(np.dot(arbitrary, basis_z)) > 0.9:
                arbitrary = np.array([0, 1, 0])
            temp_x = arbitrary - np.dot(arbitrary, basis_z) * basis_z
            temp_x = temp_x / (np.linalg.norm(temp_x) + 1e-10)
            temp_y = np.cross(basis_z, temp_x)

            # Project contour points to 2D on this plane
            projected_2d = np.array([[np.dot(v - mean, temp_x), np.dot(v - mean, temp_y)] for v in contour_points])

            # Use minimum area bbox to find optimal orientation
            bbox_result = compute_minimum_area_bbox(projected_2d)
            bbox_basis_x_2d = bbox_result['basis_x']
            bbox_basis_y_2d = bbox_result['basis_y']

            # Convert back to 3D
            basis_x = bbox_basis_x_2d[0] * temp_x + bbox_basis_x_2d[1] * temp_y
            basis_x = basis_x / (np.linalg.norm(basis_x) + 1e-10)
            basis_y = np.cross(basis_z, basis_x)
            basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-10)

            # Align with previous bounding plane orientation (sign flip only)
            prev_x = bounding_plane_info_orig['basis_x']
            proj_prev_x = prev_x - np.dot(prev_x, basis_z) * basis_z
            proj_norm = np.linalg.norm(proj_prev_x)
            if proj_norm > 1e-6:
                proj_prev_x = proj_prev_x / proj_norm
                if np.dot(basis_x, proj_prev_x) < 0:
                    basis_x = -basis_x
                    basis_y = np.cross(basis_z, basis_x)
                    basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-10)

        elif bbox_method == 'bbox':
            # Minimum area bounding box using rotating calipers
            print(f"  [DEBUG] save_bounding_planes: Using bbox method (minimum area)")
            # Create initial orthonormal basis on the plane
            arbitrary = np.array([1, 0, 0])
            if abs(np.dot(arbitrary, basis_z)) > 0.9:
                arbitrary = np.array([0, 1, 0])
            temp_x = arbitrary - np.dot(arbitrary, basis_z) * basis_z
            temp_x = temp_x / (np.linalg.norm(temp_x) + 1e-10)
            temp_y = np.cross(basis_z, temp_x)

            # Project contour points to 2D on this plane
            projected_2d = np.array([[np.dot(v - mean, temp_x), np.dot(v - mean, temp_y)] for v in contour_points])

            # Use minimum area bbox to find optimal orientation
            bbox_result = compute_minimum_area_bbox(projected_2d)
            bbox_basis_x_2d = bbox_result['basis_x']
            bbox_basis_y_2d = bbox_result['basis_y']

            # Convert back to 3D
            basis_x = bbox_basis_x_2d[0] * temp_x + bbox_basis_x_2d[1] * temp_y
            basis_x = basis_x / (np.linalg.norm(basis_x) + 1e-10)
            basis_y = np.cross(basis_z, basis_x)
            basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-10)

            # Align with previous bounding plane orientation (sign flip only)
            proj_prev_x = prev_basis_x - np.dot(prev_basis_x, basis_z) * basis_z
            proj_norm = np.linalg.norm(proj_prev_x)
            if proj_norm > 1e-6:
                proj_prev_x = proj_prev_x / proj_norm
                if np.dot(basis_x, proj_prev_x) < 0:
                    basis_x = -basis_x
                    basis_y = np.cross(basis_z, basis_x)
                    basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-10)

        elif bbox_method == 'pca':
            # Project contour points onto the plane and do 2D PCA to find principal direction
            print(f"  [DEBUG] save_bounding_planes: Using pca method")
            # This ensures basis_x aligns with the long axis of the contour

            # Create initial orthonormal basis on the plane
            arbitrary = np.array([1, 0, 0])
            if abs(np.dot(arbitrary, basis_z)) > 0.9:
                arbitrary = np.array([0, 1, 0])
            temp_x = arbitrary - np.dot(arbitrary, basis_z) * basis_z
            temp_x = temp_x / np.linalg.norm(temp_x)
            temp_y = np.cross(basis_z, temp_x)

            # Project contour points to 2D on this plane
            projected_2d = np.array([[np.dot(v - mean, temp_x), np.dot(v - mean, temp_y)] for v in contour_points])

            # 2D PCA to find principal direction within the plane
            centered_2d = projected_2d - np.mean(projected_2d, axis=0)
            _, _, eigenvectors_2d = np.linalg.svd(centered_2d)
            principal_2d = eigenvectors_2d[0]  # First principal component in 2D

            # Convert back to 3D basis_x
            basis_x = principal_2d[0] * temp_x + principal_2d[1] * temp_y
            basis_x = basis_x / np.linalg.norm(basis_x)
            basis_y = np.cross(basis_z, basis_x)
            basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-10)

        elif bbox_method == 'farthest_vertex':
            # x: parallel to farthest vertex direction (projected onto plane)
            # z: Newell normal
            # y: cross(z, x)

            # Step 1: Use pre-computed farthest vertex pair for x-axis direction
            if farthest_pair is not None:
                farthest_dir = farthest_pair[1] - farthest_pair[0]
                farthest_len = np.linalg.norm(farthest_dir)

                if farthest_len > 1e-10:
                    farthest_dir = farthest_dir / farthest_len
                else:
                    farthest_dir = prev_basis_x.copy()
            else:
                farthest_dir = prev_basis_x.copy()

            # Step 2: Project farthest direction onto plane perpendicular to basis_z (Newell normal)
            basis_x = farthest_dir - np.dot(farthest_dir, basis_z) * basis_z
            x_norm = np.linalg.norm(basis_x)
            if x_norm > 1e-10:
                basis_x = basis_x / x_norm
            else:
                # Farthest direction is parallel to z, use arbitrary perpendicular
                arbitrary = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(arbitrary, basis_z)) > 0.9:
                    arbitrary = np.array([0.0, 1.0, 0.0])
                basis_x = arbitrary - np.dot(arbitrary, basis_z) * basis_z
                basis_x = basis_x / (np.linalg.norm(basis_x) + 1e-10)

            # Step 3: y = cross(z, x)
            basis_y = np.cross(basis_z, basis_x)
            basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-10)

            # Step 4: Only sign flip for continuity (no rotation)
            if has_prev_reference:
                # Project prev_basis_x onto current plane
                prev_x_proj = prev_basis_x - np.dot(prev_basis_x, basis_z) * basis_z
                prev_x_norm = np.linalg.norm(prev_x_proj)
                if prev_x_norm > 1e-10:
                    prev_x_proj = prev_x_proj / prev_x_norm
                    # Only flip sign if facing opposite direction
                    if np.dot(basis_x, prev_x_proj) < 0:
                        basis_x = -basis_x
                        basis_y = np.cross(basis_z, basis_x)
                        basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-10)

        # Reproject with final basis
        projected_2d = np.array([[np.dot(v - mean, basis_x), np.dot(v - mean, basis_y)] for v in contour_points])
        area = compute_polygon_area(projected_2d)

        min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
        min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])
        x_len = max_x - min_x
        y_len = max_y - min_y

        ratio_threshold = 2.0
        square_like = max(x_len, y_len) / min(x_len, y_len) < ratio_threshold if min(x_len, y_len) > 1e-10 else False

        # Use the axes as found (farthest_vertex or pca), no override for square-like
        # Square-like handling is done during smoothening, not during initial finding

        # Final re-orthogonalization to ensure exactly 90-degree corners
        basis_x = basis_x / (np.linalg.norm(basis_x) + 1e-10)
        basis_y = np.cross(basis_z, basis_x)
        basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-10)

        # Re-project contour points with orthonormal basis to get correct extents
        projected_2d = np.array([[np.dot(v - mean, basis_x), np.dot(v - mean, basis_y)] for v in contour_points])
        min_x, max_x = np.min(projected_2d[:, 0]), np.max(projected_2d[:, 0])
        min_y, max_y = np.min(projected_2d[:, 1]), np.max(projected_2d[:, 1])

        bounding_plane_2d = np.array([
            [min_x, min_y], [max_x, min_y],
            [max_x, max_y], [min_x, max_y]
        ])

        # Optimize plane position along z-axis to minimize total distance from vertices
        # Compute z-coordinates of all vertices relative to current mean
        z_coords = np.array([np.dot(v - mean, basis_z) for v in contour_points])
        # Use median for L1 (sum of absolute distances) minimization
        optimal_z_offset = np.median(z_coords)
        # Shift the plane center to optimal position
        optimal_mean = mean + optimal_z_offset * basis_z

        bounding_plane = np.array([optimal_mean + x * basis_x + y * basis_y for x, y in bounding_plane_2d])
        projected_2d = np.array([optimal_mean + x * basis_x + y * basis_y for x, y in projected_2d])

        # Debug: verify 90-degree corners
        dot_xy = np.dot(basis_x, basis_y)
        # Compute actual corner angles
        angles = []
        for i in range(4):
            p0 = bounding_plane[(i - 1) % 4]
            p1 = bounding_plane[i]
            p2 = bounding_plane[(i + 1) % 4]
            v1 = p0 - p1
            v2 = p2 - p1
            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angles.append(np.degrees(np.arccos(np.clip(cos_ang, -1, 1))))

        # Preserve order if contours have been normalized
        preserve = getattr(self, '_contours_normalized', False)
        new_contour, contour_match = self.find_contour_match(contour_points, bounding_plane, preserve_order=preserve)

        return new_contour, {
            'basis_x': basis_x,
            'basis_y': basis_y,
            'basis_z': basis_z,
            'mean': optimal_mean,  # Use optimized position
            'bounding_plane': bounding_plane,
            'projected_2d': projected_2d,
            'area': area,
            'contour_match': contour_match,
            'scalar_value': scalar_value,
            'square_like': square_like,
            'newell_normal': newell_normal,
            'farthest_pair': farthest_pair,  # 3D vertex pair used for x-axis
        }

    def _trim_independent_section(self, contours, bounding_planes, max_spacing_threshold):
        """
        Trim an independent section by removing contours while keeping spacing below threshold.
        Also maintains corresponding bounding planes.
        Always keeps first and last contour.

        Args:
            contours: List of contours to trim
            bounding_planes: List of corresponding bounding plane info
            max_spacing_threshold: Maximum allowed spacing after removal

        Returns:
            Tuple of (trimmed_contours, trimmed_bounding_planes)
        """
        if len(contours) <= 2:
            return list(contours), list(bounding_planes)

        # Greedy removal: try to remove each middle contour if spacing stays below threshold
        trimmed_contours = [contours[0]]
        trimmed_bounding = [bounding_planes[0]]

        for i in range(1, len(contours) - 1):
            prev_contour = trimmed_contours[-1]
            next_contour = contours[i + 1]

            # Check if we can skip this contour
            prev_centroid = np.mean(prev_contour, axis=0)
            next_centroid = np.mean(next_contour, axis=0)
            skip_spacing = np.linalg.norm(next_centroid - prev_centroid)

            if skip_spacing <= max_spacing_threshold:
                # Can skip this contour
                continue
            else:
                # Must keep this contour
                trimmed_contours.append(contours[i])
                trimmed_bounding.append(bounding_planes[i])

        # Always keep last
        trimmed_contours.append(contours[-1])
        trimmed_bounding.append(bounding_planes[-1])

        return trimmed_contours, trimmed_bounding

    def sobol_sampling_min_contour(self, bounding_plane_stream, num_samples):
        """
        Sample 2D positions inside the smallest contour of a stream using Sobol + rejection.
        These positions are valid for all contours in the stream.

        Args:
            bounding_plane_stream: List of bounding plane dicts for one stream
            num_samples: Number of samples to generate

        Returns:
            np.array of shape (num_samples, 2) with positions in [0,1]^2
        """
        from scipy.stats.qmc import Sobol
        from shapely.geometry import Point, Polygon

        # Find the smallest contour (by area)
        min_area = np.inf
        smallest_plane = None
        for bp in bounding_plane_stream:
            if bp['area'] < min_area:
                min_area = bp['area']
                smallest_plane = bp

        # Get 2D projected contour
        projected_2d = smallest_plane['projected_2d']

        # Normalize to unit square [0,1]^2 for Sobol compatibility
        min_pt = np.min(projected_2d, axis=0)
        max_pt = np.max(projected_2d, axis=0)
        extent = max_pt - min_pt

        # Handle degenerate case
        if extent[0] < 1e-10 or extent[1] < 1e-10:
            return self.sobol_sampling_barycentric(num_samples)

        normalized_contour = (projected_2d - min_pt) / extent

        poly = Polygon(normalized_contour)
        if not poly.is_valid:
            poly = poly.buffer(0)  # Fix invalid polygons

        # Shrink polygon to keep samples away from boundary
        shrink_poly = poly.buffer(-0.1)  # 10% margin from boundary
        if shrink_poly.is_empty or not shrink_poly.is_valid:
            shrink_poly = poly  # Fallback to original if shrinking fails

        # Rejection sampling with Sobol
        seed = getattr(self, 'fiber_sampling_seed', 42)
        sobol = Sobol(d=2, scramble=True, seed=seed)
        samples = []
        max_iterations = 100
        iteration = 0

        while len(samples) < num_samples and iteration < max_iterations:
            batch_size = (num_samples - len(samples)) * 4
            candidates = sobol.random(batch_size)
            for c in candidates:
                if shrink_poly.contains(Point(c[0], c[1])):
                    samples.append(c)
                    if len(samples) >= num_samples:
                        break
            iteration += 1

        # If we couldn't get enough samples, fill with centroid or fallback
        if len(samples) < num_samples:
            centroid = np.array(poly.centroid.coords[0])
            while len(samples) < num_samples:
                samples.append(centroid)

        return np.array(samples[:num_samples])

    # ========== End Bounding Plane Methods ==========

    # ========== Soft Body Initialization Methods ==========

    def init_soft_body(self, skeleton_meshes=None, skeleton=None, mesh_info=None):
        """
        Initialize quasistatic soft body simulation from tetrahedron mesh.
        Uses LBS for initial positioning, then converges with volume preservation.
        NO collision detection - much faster.
        
        Args:
            skeleton_meshes: Dict of skeleton name -> MeshLoader (for body name lookup)
            skeleton: DART skeleton object (for initial body transforms)
            mesh_info: Dict mapping DART body names to OBJ paths (from env.mesh_info)
        """
        if self.tet_vertices is None or self.tet_tetrahedra is None:
            print("No tetrahedron mesh available. Run tetrahedralization first.")
            return False

        # Get fixed vertices from cap faces (origin/insertion anchors)
        fixed_vertices = set()
        cap_face_set = set(self.tet_cap_face_indices)
        for face_idx in cap_face_set:
            if face_idx < len(self.tet_faces):
                for vi in self.tet_faces[face_idx]:
                    fixed_vertices.add(vi)

        # Also add anchor vertices
        if hasattr(self, 'tet_anchor_vertices'):
            for vi in self.tet_anchor_vertices:
                fixed_vertices.add(vi)

        # Fix contours at origin/insertion
        # Always fix level 0 (origin) and level N-1 (insertion)
        # Additionally fix nearby levels within distance threshold
        self.contour_level_vertices = {}  # vi -> (stream_idx, end_type) for bone attachment
        distance_threshold = 0.003  # 3mm from anchor - for additional nearby levels

        if hasattr(self, 'contours') and self.contours is not None and len(self.contours) > 0:
            # Collect positions from origin/insertion contours
            # end_type: 0 = origin side, 1 = insertion side
            contour_info = []  # (position, stream_idx, end_type)

            for stream_idx, stream_contours in enumerate(self.contours):
                num_levels = len(stream_contours)
                if num_levels < 2:
                    continue

                # Get origin and insertion anchor positions (center of first/last contour)
                origin_center = np.mean(stream_contours[0], axis=0)
                insertion_center = np.mean(stream_contours[-1], axis=0)

                # Track which levels are fixed
                origin_count = 0
                insertion_count = 0

                for level_idx, contour in enumerate(stream_contours):
                    contour_center = np.mean(contour, axis=0)

                    # ALWAYS fix level 0 (origin) and level N-1 (insertion)
                    if level_idx == 0:
                        for pos in contour:
                            contour_info.append((np.array(pos), stream_idx, 0))
                        origin_count += 1
                    elif level_idx == num_levels - 1:
                        for pos in contour:
                            contour_info.append((np.array(pos), stream_idx, 1))
                        insertion_count += 1
                    else:
                        # For intermediate levels, use distance threshold
                        dist_to_origin = np.linalg.norm(contour_center - origin_center)
                        dist_to_insertion = np.linalg.norm(contour_center - insertion_center)

                        # Fix to origin if close to origin (and closer to origin than insertion)
                        if dist_to_origin < distance_threshold and dist_to_origin <= dist_to_insertion:
                            for pos in contour:
                                contour_info.append((np.array(pos), stream_idx, 0))
                            origin_count += 1
                        # Fix to insertion if close to insertion (and closer to insertion than origin)
                        elif dist_to_insertion < distance_threshold and dist_to_insertion < dist_to_origin:
                            for pos in contour:
                                contour_info.append((np.array(pos), stream_idx, 1))
                            insertion_count += 1

                print(f"  Stream {stream_idx}: fixing {origin_count} origin levels, {insertion_count} insertion levels")

            # Match tet vertices to contour positions
            if len(contour_info) > 0:
                contour_positions = np.array([info[0] for info in contour_info])
                matched_count = 0
                for vi, tet_pos in enumerate(self.tet_vertices):
                    dists = np.linalg.norm(contour_positions - tet_pos, axis=1)
                    min_idx = np.argmin(dists)
                    if dists[min_idx] < 1e-4:  # Very tight threshold for exact match
                        fixed_vertices.add(vi)
                        _, stream_idx, end_type = contour_info[min_idx]
                        self.contour_level_vertices[vi] = (stream_idx, end_type)
                        matched_count += 1
                print(f"  Fixed {matched_count} vertices from contours near origin/insertion")

        self.soft_body_fixed_vertices = list(fixed_vertices)

        # Create quasistatic soft body simulation instance
        self.soft_body = SoftBodySimulation(
            vertices=self.tet_vertices.astype(np.float64),
            tetrahedra=self.tet_tetrahedra,
            fixed_vertices=self.soft_body_fixed_vertices,
            stiffness=self.soft_body_stiffness,
            damping=self.soft_body_damping,
            volume_stiffness=self.soft_body_volume_stiffness
        )

        # Build mapping from skeleton mesh name to DART body node name
        # Multiple lookup strategies for robustness
        mesh_to_body = {}
        mesh_to_body_normalized = {}  # Normalized names (lowercase, no underscores)
        if mesh_info is not None:
            for body_name, obj_path in mesh_info.items():
                if obj_path:
                    mesh_name = obj_path.split('/')[-1].replace('.obj', '').replace('.OBJ', '')
                    mesh_to_body[mesh_name] = body_name
                    # Also store normalized version for fuzzy matching
                    normalized = mesh_name.lower().replace('_', '').replace('-', '')
                    mesh_to_body_normalized[normalized] = body_name
            print(f"  Mesh to body mapping: {len(mesh_to_body)} entries")

        def find_dart_body(skeleton, mesh_name):
            """Find DART body node for a given mesh name using multiple strategies."""
            # Strategy 1: Direct lookup in mesh_to_body
            body_name = mesh_to_body.get(mesh_name)
            if body_name:
                body_node = skeleton.getBodyNode(body_name)
                if body_node is not None:
                    return body_node, body_name

            # Strategy 2: Normalized lookup (case-insensitive, no underscores)
            normalized = mesh_name.lower().replace('_', '').replace('-', '')
            body_name = mesh_to_body_normalized.get(normalized)
            if body_name:
                body_node = skeleton.getBodyNode(body_name)
                if body_node is not None:
                    return body_node, body_name

            # Strategy 3: Direct body name lookup (mesh_name might be body name)
            body_node = skeleton.getBodyNode(mesh_name)
            if body_node is not None:
                return body_node, mesh_name

            # Strategy 4: Search all bodies for substring match
            for i in range(skeleton.getNumBodyNodes()):
                bn = skeleton.getBodyNode(i)
                bn_name = bn.getName()
                bn_normalized = bn_name.lower().replace('_', '').replace('-', '')

                # Check if normalized names match
                if normalized == bn_normalized:
                    return bn, bn_name

                # Check if one contains the other
                if normalized in bn_normalized or bn_normalized in normalized:
                    return bn, bn_name

                # Check common naming patterns (e.g., "L_Femur" vs "FemurL")
                # Remove L/R prefix/suffix and compare
                mesh_core = normalized.replace('l', '').replace('r', '')
                bn_core = bn_normalized.replace('l', '').replace('r', '')
                if len(mesh_core) > 2 and len(bn_core) > 2:
                    if mesh_core == bn_core:
                        return bn, bn_name

            return None, None

        # Store REST POSE transforms for skeleton bones
        self.skeleton_rest_transforms = {}
        if skeleton is not None:
            for i in range(skeleton.getNumBodyNodes()):
                body_node = skeleton.getBodyNode(i)
                body_name = body_node.getName()
                world_transform = body_node.getWorldTransform()
                rotation = world_transform.rotation()
                translation = world_transform.translation()
                self.skeleton_rest_transforms[body_name] = (rotation.copy(), translation.copy())
            print(f"  Stored {len(self.skeleton_rest_transforms)} skeleton rest transforms")

        # Store initial body transforms and compute local anchor positions
        self.soft_body_initial_transforms = {}
        self.soft_body_local_anchors = {}

        if skeleton is not None and skeleton_meshes is not None and hasattr(self, 'tet_cap_attachments'):
            skeleton_names = list(skeleton_meshes.keys())

            for attachment in self.tet_cap_attachments:
                anchor_idx, stream_idx, end_type, skel_mesh_idx, subpart_idx = attachment

                if skel_mesh_idx >= len(skeleton_names):
                    continue

                mesh_name = skeleton_names[skel_mesh_idx]

                try:
                    # Use robust body finding
                    body_node, body_name = find_dart_body(skeleton, mesh_name)

                    if body_node is None:
                        print(f"  Could not find DART body for mesh '{mesh_name}'")
                        continue

                    world_transform = body_node.getWorldTransform()
                    rotation = world_transform.rotation()
                    translation = world_transform.translation()

                    if body_name not in self.soft_body_initial_transforms:
                        self.soft_body_initial_transforms[body_name] = (rotation.copy(), translation.copy())

                    anchor_world_pos = self.soft_body.rest_positions[anchor_idx]
                    local_pos = rotation.T @ (anchor_world_pos - translation)

                    self.soft_body_local_anchors[anchor_idx] = (body_name, local_pos.copy())

                    print(f"  Anchor {anchor_idx} -> mesh '{mesh_name}' -> body '{body_name}'")

                except Exception as e:
                    print(f"  Failed to attach anchor {anchor_idx}: {e}")
                    continue

            # Assign ALL fixed vertices to their appropriate body
            # For muscles attaching to multiple bones (like psoas major),
            # each fixed vertex finds its nearest bone independently
            self._assign_fixed_vertices_to_nearest_bones(skeleton, skeleton_meshes, mesh_to_body)
            print(f"  Assigned {len(self.soft_body_local_anchors)} fixed vertices to bodies")

        # Compute LBS skinning weights for ALL vertices (not just fixed)
        self._compute_skinning_weights(skeleton, mesh_to_body, list(skeleton_meshes.keys()) if skeleton_meshes else [])

        print(f"Initialized quasistatic soft body simulation:")
        print(f"  Vertices: {self.soft_body.num_vertices}")
        print(f"  Fixed vertices: {len(self.soft_body_fixed_vertices)}")
        print(f"  Edges: {len(self.soft_body.edges)}")
        print(f"  Tetrahedra: {len(self.soft_body.tetrahedra)}")
        print(f"  Cap attachments: {len(self.tet_cap_attachments)}")
        print(f"  Body attachments: {len(self.soft_body_local_anchors)}")
        print(f"  Skinning bones: {len(getattr(self, 'skinning_bones', []))}")

        # Store original waypoints for reset (before any simulation)
        if hasattr(self, 'waypoints') and self.waypoints is not None and len(self.waypoints) > 0:
            if not hasattr(self, 'waypoints_original') or self.waypoints_original is None:
                self.waypoints_original = [
                    [np.array(wp).copy() for wp in stream]
                    for stream in self.waypoints
                ]

        # Compute barycentric coordinates for waypoints
        if hasattr(self, 'waypoints') and len(self.waypoints) > 0:
            if getattr(self, 'waypoints_from_tet_sim', True):
                self._compute_waypoint_barycentric_coords(skeleton_meshes, skeleton)
            else:
                print(f"  Skipping waypoint embedding (waypoints are imported)")

        # Compute skeleton bindings for all tet vertices
        self._compute_tet_skeleton_bindings(skeleton_meshes, skeleton)

        # Classify edges based on contour structure for muscle-like behavior
        cross_mask, intra_mask = self.compute_tet_edge_contour_types()
        if cross_mask is not None and intra_mask is not None:
            self.soft_body.set_contour_edge_types(cross_mask, intra_mask)

        # Compute outward directions for volume-preserving bulge
        outward_dirs = self.compute_outward_directions()
        if outward_dirs is not None:
            self.soft_body.set_outward_directions(outward_dirs)

        print(f"Soft body initialized")
        return True

    def sobol_sampling_barycentric(self, num_samples):
        """Generate Sobol samples in 0.95x0.95 unit square with margin from edges."""
        from scipy.stats.qmc import Sobol

        # Generate Sobol samples in unit square [0,1]^2
        seed = getattr(self, 'fiber_sampling_seed', 42)
        sobol = Sobol(d=2, scramble=True, seed=seed)
        samples = sobol.random(num_samples)

        # Scale to [0.025, 0.975] to keep samples inside 0.95x0.95 area
        margin = 0.025
        samples = margin + samples * (1 - 2 * margin)

        return samples

    def sobol_sampling_square(self, num_samples, margin=0.05):
        """Generate Sobol samples in unit square [0,1]^2 with optional margin.

        Args:
            num_samples: Number of samples to generate
            margin: Margin from edges (default 0.05 gives [0.05, 0.95] range)

        Returns:
            (num_samples, 2) array of samples in [margin, 1-margin]^2
        """
        from scipy.stats.qmc import Sobol

        seed = getattr(self, 'fiber_sampling_seed', 42)
        sobol = Sobol(d=2, scramble=True, seed=seed)
        samples = sobol.random(num_samples)

        # Scale to [margin, 1-margin] to keep samples slightly inside the boundary
        samples = margin + samples * (1 - 2 * margin)

        return samples
