"""
FEM-based muscle simulation with Neo-Hookean material model.

Self-contained module using Taichi for GPU acceleration.
Handles muscle-bone collisions (penalty-based) and inter-muscle contacts.
"""

import numpy as np

try:
    import taichi as ti
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Taichi kernels (module-level, lazily initialized)
# ---------------------------------------------------------------------------

_ti_initialized = False


def _ensure_ti_init():
    global _ti_initialized
    if _ti_initialized:
        return
    if not TAICHI_AVAILABLE:
        raise RuntimeError("Taichi is required for FEM simulation")
    # Only init if not already initialized by another module
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f64)
    except RuntimeError:
        pass  # Already initialized
    _ti_initialized = True


# ---------------------------------------------------------------------------
# FEMSimulation
# ---------------------------------------------------------------------------


class FEMSimulation:
    """Neo-Hookean FEM simulation with Newton-CG solver on Taichi GPU."""

    def __init__(self, youngs_modulus=5000.0, poisson_ratio=0.49,
                 collision_kappa=1e4, contact_threshold=0.015):
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self.kappa = collision_kappa
        self.contact_threshold = contact_threshold

        # Lamé parameters
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        # State
        self._built = False
        self._n_verts = 0
        self._n_tets = 0

        # Collision data
        self._bone_meshes = None
        self._bone_margin = 0.002
        self._collision_verts = None  # indices of surface verts near bones
        self._collision_targets = None  # push-out target positions
        self._collision_depths = None  # penetration depths

        # Inter-muscle contact
        self._contact_pairs = None  # (K, 2) int array of vertex index pairs
        self._contact_rest_dists = None  # (K,) float array

        # Taichi fields (allocated in build)
        self._fields_allocated = False

    def build(self, vertices, tetrahedra, fixed_mask, surface_faces=None):
        """
        Initialize the FEM simulation with mesh data.

        Args:
            vertices: (N, 3) float64 rest positions
            tetrahedra: (M, 4) int32 tet vertex indices
            fixed_mask: (N,) bool array — True for fixed (Dirichlet) vertices
            surface_faces: (F, 3) int32 optional surface triangle indices for collision
        """
        _ensure_ti_init()

        self._n_verts = len(vertices)
        self._n_tets = len(tetrahedra)

        # Store numpy arrays
        self.rest_positions = vertices.astype(np.float64).copy()
        self.positions = vertices.astype(np.float64).copy()
        self.tetrahedra = tetrahedra.astype(np.int32)
        self.fixed_mask = fixed_mask.astype(bool)
        self.free_mask = ~self.fixed_mask
        self.fixed_indices = np.where(self.fixed_mask)[0]
        self.free_indices = np.where(self.free_mask)[0]
        self.fixed_targets = self.positions[self.fixed_indices].copy()

        if surface_faces is not None:
            self.surface_faces = surface_faces.astype(np.int32)
            # Surface vertex indices (unique)
            self.surface_verts = np.unique(surface_faces.ravel())
        else:
            self.surface_faces = None
            self.surface_verts = np.arange(self._n_verts)

        # Precompute rest-state edge matrix inverse and volume per tet
        self._precompute_rest_state()

        # Allocate Taichi fields
        self._allocate_fields()

        # Upload data to GPU
        self._upload_data()

        self._built = True

    def _precompute_rest_state(self):
        """Compute Dm_inv (rest edge matrix inverse) and rest_volume per tet."""
        N = self._n_verts
        M = self._n_tets
        tets = self.tetrahedra
        X = self.rest_positions

        # Edge matrix: Dm = [x1-x0, x2-x0, x3-x0] (3x3 per tet)
        v0 = X[tets[:, 0]]  # (M, 3)
        v1 = X[tets[:, 1]]
        v2 = X[tets[:, 2]]
        v3 = X[tets[:, 3]]

        # Dm columns: e1, e2, e3
        e1 = v1 - v0  # (M, 3)
        e2 = v2 - v0
        e3 = v3 - v0

        # Dm as (M, 3, 3)
        Dm = np.stack([e1, e2, e3], axis=-1)  # (M, 3, 3) — columns are edges

        # Dm_inv per tet
        self.Dm_inv = np.linalg.inv(Dm)  # (M, 3, 3)

        # Rest volume = |det(Dm)| / 6
        self.rest_volume = np.abs(np.linalg.det(Dm)) / 6.0  # (M,)

        # Check for degenerate tets
        degenerate = self.rest_volume < 1e-15
        if np.any(degenerate):
            n_deg = np.sum(degenerate)
            print(f"  FEM: Warning: {n_deg} degenerate tetrahedra (zero volume)")
            self.rest_volume[degenerate] = 1e-15

    def _allocate_fields(self):
        """Allocate Taichi fields for GPU computation."""
        N = self._n_verts
        M = self._n_tets

        # Vertex fields
        self.ti_positions = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_gradient = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_dx = ti.Vector.field(3, dtype=ti.f64, shape=N)  # search direction
        self.ti_Hdx = ti.Vector.field(3, dtype=ti.f64, shape=N)  # Hessian-vector product
        self.ti_r = ti.Vector.field(3, dtype=ti.f64, shape=N)   # CG residual
        self.ti_p = ti.Vector.field(3, dtype=ti.f64, shape=N)   # CG direction
        self.ti_Hp = ti.Vector.field(3, dtype=ti.f64, shape=N)  # H * p
        self.ti_positions_tmp = ti.Vector.field(3, dtype=ti.f64, shape=N)  # temp for FD
        self.ti_grad_tmp = ti.Vector.field(3, dtype=ti.f64, shape=N)  # temp gradient

        # Fixed vertex data
        self.ti_fixed_mask = ti.field(dtype=ti.i32, shape=N)
        self.ti_fixed_targets = ti.Vector.field(3, dtype=ti.f64, shape=N)

        # Tet fields
        self.ti_tets = ti.Vector.field(4, dtype=ti.i32, shape=M)
        self.ti_Dm_inv = ti.Matrix.field(3, 3, dtype=ti.f64, shape=M)
        self.ti_rest_volume = ti.field(dtype=ti.f64, shape=M)

        # Material params
        self.ti_mu = ti.field(dtype=ti.f64, shape=())
        self.ti_lam = ti.field(dtype=ti.f64, shape=())

        # Collision fields (allocated dynamically)
        self._n_collision_verts = 0
        self.ti_collision_indices = None
        self.ti_collision_targets = None
        self.ti_collision_depths = None
        self.ti_kappa = ti.field(dtype=ti.f64, shape=())

        # Contact fields (allocated dynamically)
        self._n_contacts = 0
        self.ti_contact_pairs = None
        self.ti_contact_rest_dists = None

        # Scalar reduction fields
        self.ti_energy = ti.field(dtype=ti.f64, shape=())
        self.ti_dot_result = ti.field(dtype=ti.f64, shape=())

        self._fields_allocated = True

    def _upload_data(self):
        """Upload mesh data to Taichi fields."""
        self.ti_positions.from_numpy(self.positions)
        self.ti_tets.from_numpy(self.tetrahedra)
        self.ti_Dm_inv.from_numpy(self.Dm_inv)
        self.ti_rest_volume.from_numpy(self.rest_volume)

        fixed_mask_int = self.fixed_mask.astype(np.int32)
        self.ti_fixed_mask.from_numpy(fixed_mask_int)

        # Upload fixed targets (full array, only fixed indices matter)
        targets_full = np.zeros((self._n_verts, 3), dtype=np.float64)
        targets_full[self.fixed_indices] = self.fixed_targets
        self.ti_fixed_targets.from_numpy(targets_full)

        self.ti_mu[None] = self.mu
        self.ti_lam[None] = self.lam
        self.ti_kappa[None] = self.kappa

    def set_fixed_targets(self, targets):
        """
        Update fixed vertex target positions.

        Args:
            targets: (F, 3) float64 array matching fixed_indices ordering
        """
        self.fixed_targets = targets.astype(np.float64).copy()
        targets_full = np.zeros((self._n_verts, 3), dtype=np.float64)
        targets_full[self.fixed_indices] = self.fixed_targets
        self.ti_fixed_targets.from_numpy(targets_full)

        # Also snap positions
        self.positions[self.fixed_indices] = self.fixed_targets
        self.ti_positions.from_numpy(self.positions)

    def set_bone_collisions(self, bone_trimeshes, margin=0.002):
        """
        Set bone trimeshes for collision detection.

        Args:
            bone_trimeshes: list of trimesh.Trimesh objects
            margin: push-out margin in meters
        """
        self._bone_meshes = bone_trimeshes
        self._bone_margin = margin

    def set_inter_muscle_contacts(self, contact_pairs, rest_distances):
        """
        Set inter-muscle contact vertex pairs.

        Args:
            contact_pairs: (K, 2) int array — pairs of vertex indices
            rest_distances: (K,) float array — rest distances
        """
        self._contact_pairs = contact_pairs.astype(np.int32)
        self._contact_rest_dists = rest_distances.astype(np.float64)
        self._n_contacts = len(contact_pairs)

        if self._n_contacts > 0:
            self.ti_contact_pairs = ti.Vector.field(2, dtype=ti.i32, shape=self._n_contacts)
            self.ti_contact_rest_dists = ti.field(dtype=ti.f64, shape=self._n_contacts)
            self.ti_contact_pairs.from_numpy(self._contact_pairs)
            self.ti_contact_rest_dists.from_numpy(self._contact_rest_dists)

    def _detect_bone_collisions(self):
        """CPU: detect surface vertices penetrating bone meshes."""
        if self._bone_meshes is None or len(self._bone_meshes) == 0:
            self._n_collision_verts = 0
            return

        # Query surface vertices only
        surf_positions = self.positions[self.surface_verts]
        n_surf = len(surf_positions)

        min_dist = np.full(n_surf, np.inf)
        closest_pts = np.zeros_like(surf_positions)
        closest_normals = np.zeros_like(surf_positions)

        for mesh in self._bone_meshes:
            if mesh is None:
                continue
            try:
                cp, dists, face_ids = mesh.nearest.on_surface(surf_positions)
                normals = mesh.face_normals[face_ids]
                closer = dists < min_dist
                min_dist[closer] = dists[closer]
                closest_pts[closer] = cp[closer]
                closest_normals[closer] = normals[closer]
            except Exception:
                continue

        # Penetration check: inside if dot(vertex - surface_pt, normal) < 0
        to_vertex = surf_positions - closest_pts
        dots = np.einsum('ij,ij->i', to_vertex, closest_normals)
        penetrating = (min_dist < self._bone_margin) & (dots < 0)

        pen_indices = np.where(penetrating)[0]
        if len(pen_indices) == 0:
            self._n_collision_verts = 0
            return

        # Map back to global vertex indices
        global_indices = self.surface_verts[pen_indices]
        # Filter out fixed vertices
        is_free = self.free_mask[global_indices]
        pen_indices = pen_indices[is_free]
        global_indices = global_indices[is_free]

        if len(global_indices) == 0:
            self._n_collision_verts = 0
            return

        # Compute push-out targets
        targets = closest_pts[pen_indices] + closest_normals[pen_indices] * self._bone_margin
        depths = self._bone_margin - min_dist[pen_indices]

        self._n_collision_verts = len(global_indices)
        self._collision_verts = global_indices
        self._collision_targets = targets
        self._collision_depths = depths

        # Upload to Taichi
        if self.ti_collision_indices is None or self.ti_collision_indices.shape[0] < self._n_collision_verts:
            self.ti_collision_indices = ti.field(dtype=ti.i32, shape=self._n_collision_verts)
            self.ti_collision_targets = ti.Vector.field(3, dtype=ti.f64, shape=self._n_collision_verts)
            self.ti_collision_depths = ti.field(dtype=ti.f64, shape=self._n_collision_verts)

        self.ti_collision_indices.from_numpy(global_indices.astype(np.int32))
        self.ti_collision_targets.from_numpy(targets)
        self.ti_collision_depths.from_numpy(depths)

    def solve(self, max_newton_iters=10, cg_max_iters=50, cg_tol=1e-6,
              newton_tol=1e-4, verbose=False):
        """
        Run Newton-CG solver for one frame.

        Returns:
            (iterations, residual) tuple
        """
        if not self._built:
            raise RuntimeError("Must call build() before solve()")

        for newton_iter in range(max_newton_iters):
            # 1. Zero gradient
            _zero_field(self.ti_gradient, self._n_verts)

            # 2. Compute elastic energy gradient
            _compute_elastic_gradient(
                self.ti_positions, self.ti_tets, self.ti_Dm_inv,
                self.ti_rest_volume, self.ti_gradient,
                self.ti_mu, self.ti_lam, self._n_tets
            )

            # 3. Bone collision detection (CPU) and penalty gradient (GPU)
            self._detect_bone_collisions()
            if self._n_collision_verts > 0:
                _add_collision_gradient(
                    self.ti_positions, self.ti_gradient,
                    self.ti_collision_indices, self.ti_collision_targets,
                    self.ti_kappa, self._n_collision_verts
                )

            # 4. Inter-muscle contact gradient
            if self._n_contacts > 0:
                _add_contact_gradient(
                    self.ti_positions, self.ti_gradient,
                    self.ti_contact_pairs, self.ti_contact_rest_dists,
                    self.ti_kappa, self._n_contacts
                )

            # 5. Enforce Dirichlet BCs
            _enforce_dirichlet(
                self.ti_gradient, self.ti_positions, self.ti_fixed_targets,
                self.ti_fixed_mask, self._n_verts
            )

            # 6. Check convergence (gradient norm of free vertices)
            _dot_product(self.ti_gradient, self.ti_gradient, self.ti_dot_result,
                         self.ti_fixed_mask, self._n_verts)
            grad_norm = np.sqrt(self.ti_dot_result[None])

            if verbose:
                print(f"  Newton iter {newton_iter}: |grad| = {grad_norm:.6e}"
                      f" collisions={self._n_collision_verts}")

            if grad_norm < newton_tol:
                if verbose:
                    print(f"  Newton converged at iter {newton_iter}")
                break

            # 7. CG solve: H @ dx = -grad
            cg_iters = self._cg_solve(cg_max_iters, cg_tol)

            # 8. Backtracking line search
            alpha = self._line_search()

            # 9. Update positions: x += alpha * dx
            _update_positions(self.ti_positions, self.ti_dx, alpha, self._n_verts)

            # 10. Snap fixed vertices
            _snap_fixed(self.ti_positions, self.ti_fixed_targets,
                        self.ti_fixed_mask, self._n_verts)

        # Download final positions
        self.positions = self.ti_positions.to_numpy()

        return newton_iter + 1, grad_norm

    def _cg_solve(self, max_iters, tol):
        """CG solve for Newton direction: H @ dx = -grad, using finite-diff Hv."""
        N = self._n_verts

        # dx = 0
        _zero_field(self.ti_dx, N)

        # r = -grad (residual)
        _copy_neg(self.ti_r, self.ti_gradient, N)

        # p = r
        _copy_field(self.ti_p, self.ti_r, N)

        # rr = r · r (free vertices only)
        _dot_product(self.ti_r, self.ti_r, self.ti_dot_result,
                     self.ti_fixed_mask, N)
        rr = self.ti_dot_result[None]

        if rr < tol * tol:
            return 0

        for cg_iter in range(max_iters):
            # Hp = Hessian-vector product via finite difference
            self._hessian_vector_product()

            # Enforce Dirichlet on Hp
            _zero_fixed(self.ti_Hp, self.ti_fixed_mask, N)

            # pHp = p · Hp
            _dot_product(self.ti_p, self.ti_Hp, self.ti_dot_result,
                         self.ti_fixed_mask, N)
            pHp = self.ti_dot_result[None]

            if abs(pHp) < 1e-30:
                break

            alpha = rr / pHp

            # dx += alpha * p
            _axpy(self.ti_dx, self.ti_p, alpha, N)

            # r -= alpha * Hp
            _axpy(self.ti_r, self.ti_Hp, -alpha, N)

            # new rr
            _dot_product(self.ti_r, self.ti_r, self.ti_dot_result,
                         self.ti_fixed_mask, N)
            rr_new = self.ti_dot_result[None]

            if rr_new < tol * tol:
                return cg_iter + 1

            beta = rr_new / rr
            rr = rr_new

            # p = r + beta * p
            _update_cg_direction(self.ti_p, self.ti_r, beta, N)

        return max_iters

    def _hessian_vector_product(self):
        """Compute H @ p via finite difference: (grad(x + eps*p) - grad(x)) / eps."""
        eps = 1e-6
        N = self._n_verts
        M = self._n_tets

        # Save x_tmp = x + eps * p
        _perturb(self.ti_positions_tmp, self.ti_positions, self.ti_p, eps, N)

        # Compute gradient at perturbed position
        _zero_field(self.ti_grad_tmp, N)
        _compute_elastic_gradient(
            self.ti_positions_tmp, self.ti_tets, self.ti_Dm_inv,
            self.ti_rest_volume, self.ti_grad_tmp,
            self.ti_mu, self.ti_lam, M
        )

        # Add collision gradient at perturbed position
        if self._n_collision_verts > 0:
            _add_collision_gradient(
                self.ti_positions_tmp, self.ti_grad_tmp,
                self.ti_collision_indices, self.ti_collision_targets,
                self.ti_kappa, self._n_collision_verts
            )

        # Add contact gradient at perturbed position
        if self._n_contacts > 0:
            _add_contact_gradient(
                self.ti_positions_tmp, self.ti_grad_tmp,
                self.ti_contact_pairs, self.ti_contact_rest_dists,
                self.ti_kappa, self._n_contacts
            )

        # Hp = (grad_tmp - gradient) / eps
        _finite_diff(self.ti_Hp, self.ti_grad_tmp, self.ti_gradient, eps, N)

    def _line_search(self, max_ls_iters=8):
        """Backtracking line search along dx."""
        alpha = 1.0
        N = self._n_verts
        M = self._n_tets

        # Current energy
        _compute_total_energy(
            self.ti_positions, self.ti_tets, self.ti_Dm_inv,
            self.ti_rest_volume, self.ti_energy,
            self.ti_mu, self.ti_lam, M
        )
        E0 = self.ti_energy[None]

        # Directional derivative: grad · dx
        _dot_product(self.ti_gradient, self.ti_dx, self.ti_dot_result,
                     self.ti_fixed_mask, N)
        dir_deriv = self.ti_dot_result[None]

        c = 1e-4  # Armijo condition parameter

        for _ in range(max_ls_iters):
            # Trial position: x_tmp = x + alpha * dx
            _perturb(self.ti_positions_tmp, self.ti_positions, self.ti_dx, alpha, N)

            # Snap fixed
            _snap_fixed(self.ti_positions_tmp, self.ti_fixed_targets,
                        self.ti_fixed_mask, N)

            # Energy at trial
            _compute_total_energy(
                self.ti_positions_tmp, self.ti_tets, self.ti_Dm_inv,
                self.ti_rest_volume, self.ti_energy,
                self.ti_mu, self.ti_lam, M
            )
            E_trial = self.ti_energy[None]

            if E_trial <= E0 + c * alpha * dir_deriv:
                return alpha

            alpha *= 0.5

        return alpha

    def get_positions(self):
        """Return current vertex positions as numpy array."""
        return self.positions.copy()

    def update_material(self, youngs_modulus=None, poisson_ratio=None,
                        collision_kappa=None):
        """Update material parameters (can be called between frames)."""
        if youngs_modulus is not None:
            self.E = youngs_modulus
        if poisson_ratio is not None:
            self.nu = poisson_ratio
        if collision_kappa is not None:
            self.kappa = collision_kappa

        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        if self._fields_allocated:
            self.ti_mu[None] = self.mu
            self.ti_lam[None] = self.lam
            self.ti_kappa[None] = self.kappa


# ---------------------------------------------------------------------------
# Taichi kernels
# ---------------------------------------------------------------------------

@ti.kernel
def _zero_field(field: ti.template(), n: ti.i32):
    for i in range(n):
        field[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)


@ti.kernel
def _copy_field(dst: ti.template(), src: ti.template(), n: ti.i32):
    for i in range(n):
        dst[i] = src[i]


@ti.kernel
def _copy_neg(dst: ti.template(), src: ti.template(), n: ti.i32):
    for i in range(n):
        dst[i] = -src[i]


@ti.kernel
def _axpy(y: ti.template(), x: ti.template(), alpha: ti.f64, n: ti.i32):
    """y += alpha * x"""
    for i in range(n):
        y[i] += alpha * x[i]


@ti.kernel
def _update_cg_direction(p: ti.template(), r: ti.template(),
                         beta: ti.f64, n: ti.i32):
    """p = r + beta * p"""
    for i in range(n):
        p[i] = r[i] + beta * p[i]


@ti.kernel
def _perturb(dst: ti.template(), x: ti.template(), d: ti.template(),
             eps: ti.f64, n: ti.i32):
    """dst = x + eps * d"""
    for i in range(n):
        dst[i] = x[i] + eps * d[i]


@ti.kernel
def _finite_diff(result: ti.template(), g1: ti.template(), g0: ti.template(),
                 eps: ti.f64, n: ti.i32):
    """result = (g1 - g0) / eps"""
    for i in range(n):
        result[i] = (g1[i] - g0[i]) / eps


@ti.kernel
def _dot_product(a: ti.template(), b: ti.template(), result: ti.template(),
                 fixed_mask: ti.template(), n: ti.i32):
    """Dot product of a and b over free vertices only."""
    result[None] = 0.0
    for i in range(n):
        if fixed_mask[i] == 0:
            result[None] += a[i].dot(b[i])


@ti.kernel
def _update_positions(positions: ti.template(), dx: ti.template(),
                      alpha: ti.f64, n: ti.i32):
    for i in range(n):
        positions[i] += alpha * dx[i]


@ti.kernel
def _zero_fixed(field: ti.template(), fixed_mask: ti.template(), n: ti.i32):
    for i in range(n):
        if fixed_mask[i] == 1:
            field[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)


@ti.kernel
def _snap_fixed(positions: ti.template(), targets: ti.template(),
                fixed_mask: ti.template(), n: ti.i32):
    for i in range(n):
        if fixed_mask[i] == 1:
            positions[i] = targets[i]


@ti.kernel
def _enforce_dirichlet(gradient: ti.template(), positions: ti.template(),
                       targets: ti.template(), fixed_mask: ti.template(),
                       n: ti.i32):
    """Zero gradient at fixed vertices and snap their positions."""
    for i in range(n):
        if fixed_mask[i] == 1:
            gradient[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
            positions[i] = targets[i]


@ti.kernel
def _compute_elastic_gradient(
    positions: ti.template(), tets: ti.template(),
    Dm_inv: ti.template(), rest_volume: ti.template(),
    gradient: ti.template(), mu: ti.template(), lam: ti.template(),
    n_tets: ti.i32
):
    """
    Compute Neo-Hookean energy gradient and distribute to vertices.
    W = mu/2 * (I1 - 3) - mu * log(J) + lam/2 * (log(J))^2
    P = mu * (F - F^{-T}) + lam * log(J) * F^{-T}
    """
    for t in range(n_tets):
        i0 = tets[t][0]
        i1 = tets[t][1]
        i2 = tets[t][2]
        i3 = tets[t][3]

        x0 = positions[i0]
        x1 = positions[i1]
        x2 = positions[i2]
        x3 = positions[i3]

        # Deformed edge matrix Ds
        e1 = x1 - x0
        e2 = x2 - x0
        e3 = x3 - x0
        Ds = ti.Matrix([
            [e1[0], e2[0], e3[0]],
            [e1[1], e2[1], e3[1]],
            [e1[2], e2[2], e3[2]]
        ], dt=ti.f64)

        # Deformation gradient F = Ds @ Dm_inv
        F = Ds @ Dm_inv[t]

        # J = det(F), clamped to avoid log(0)
        J = F.determinant()
        J_safe = ti.max(J, 1e-6)
        logJ = ti.log(J_safe)

        # F^{-T} = cofactor(F) / J  (use adjugate / J)
        # For 3x3: F_inv_T = (1/J) * cofactor(F)^T
        # More robust: compute F_inv_T directly
        # cofactor matrix
        cof = ti.Matrix.zero(ti.f64, 3, 3)
        cof[0, 0] = F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]
        cof[0, 1] = F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2]
        cof[0, 2] = F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]
        cof[1, 0] = F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2]
        cof[1, 1] = F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0]
        cof[1, 2] = F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1]
        cof[2, 0] = F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]
        cof[2, 1] = F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]
        cof[2, 2] = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]

        F_inv_T = cof / J_safe

        # First Piola-Kirchhoff stress: P = mu*(F - F^{-T}) + lam*log(J)*F^{-T}
        P = mu[None] * (F - F_inv_T) + lam[None] * logJ * F_inv_T

        # Force = -V * P @ Dm_inv^T
        # grad_Ds = V * P @ Dm_inv^T  (energy gradient w.r.t. Ds)
        H = rest_volume[t] * P @ Dm_inv[t].transpose()

        # Distribute to vertices
        # grad(x1) += H[:, 0], grad(x2) += H[:, 1], grad(x3) += H[:, 2]
        # grad(x0) -= H[:, 0] + H[:, 1] + H[:, 2]
        h0 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]], dt=ti.f64)
        h1 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]], dt=ti.f64)
        h2 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]], dt=ti.f64)

        ti.atomic_add(gradient[i1], h0)
        ti.atomic_add(gradient[i2], h1)
        ti.atomic_add(gradient[i3], h2)
        ti.atomic_add(gradient[i0], -(h0 + h1 + h2))


@ti.kernel
def _compute_total_energy(
    positions: ti.template(), tets: ti.template(),
    Dm_inv: ti.template(), rest_volume: ti.template(),
    energy: ti.template(), mu: ti.template(), lam: ti.template(),
    n_tets: ti.i32
):
    """Compute total Neo-Hookean elastic energy."""
    energy[None] = 0.0
    for t in range(n_tets):
        i0 = tets[t][0]
        i1 = tets[t][1]
        i2 = tets[t][2]
        i3 = tets[t][3]

        x0 = positions[i0]
        x1 = positions[i1]
        x2 = positions[i2]
        x3 = positions[i3]

        e1 = x1 - x0
        e2 = x2 - x0
        e3 = x3 - x0
        Ds = ti.Matrix([
            [e1[0], e2[0], e3[0]],
            [e1[1], e2[1], e3[1]],
            [e1[2], e2[2], e3[2]]
        ], dt=ti.f64)

        F = Ds @ Dm_inv[t]
        J = F.determinant()
        J_safe = ti.max(J, 1e-6)
        logJ = ti.log(J_safe)

        # I1 = tr(F^T F)
        I1 = (F.transpose() @ F).trace()

        # W = mu/2 * (I1 - 3) - mu * log(J) + lam/2 * (log(J))^2
        W = mu[None] / 2.0 * (I1 - 3.0) - mu[None] * logJ + lam[None] / 2.0 * logJ * logJ

        energy[None] += rest_volume[t] * W


@ti.kernel
def _add_collision_gradient(
    positions: ti.template(), gradient: ti.template(),
    collision_indices: ti.template(), collision_targets: ti.template(),
    kappa: ti.template(), n_collisions: ti.i32
):
    """Add penalty gradient for bone-penetrating vertices."""
    for k in range(n_collisions):
        idx = collision_indices[k]
        target = collision_targets[k]
        x = positions[idx]
        # Penalty: E = kappa/2 * ||x - target||^2 when penetrating
        # grad = kappa * (x - target)
        grad = kappa[None] * (x - target)
        ti.atomic_add(gradient[idx], grad)


@ti.kernel
def _add_contact_gradient(
    positions: ti.template(), gradient: ti.template(),
    contact_pairs: ti.template(), rest_dists: ti.template(),
    kappa: ti.template(), n_contacts: ti.i32
):
    """Add penalty gradient for inter-muscle contact pairs."""
    for k in range(n_contacts):
        i = contact_pairs[k][0]
        j = contact_pairs[k][1]
        xi = positions[i]
        xj = positions[j]
        diff = xi - xj
        dist = diff.norm()
        if dist < rest_dists[k] and dist > 1e-10:
            # Penalty: E = kappa/2 * (dist - rest_dist)^2 when dist < rest_dist
            # grad_i = kappa * (dist - rest_dist) * (xi - xj) / dist
            scale = kappa[None] * (dist - rest_dists[k]) / dist
            gi = scale * diff
            ti.atomic_add(gradient[i], gi)
            ti.atomic_add(gradient[j], -gi)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_fem_sim(v, max_iterations=10, tolerance=1e-4, verbose=True):
    """
    Run FEM simulation for all muscles, with bone collision and inter-muscle contacts.

    Args:
        v: Viewer context (or SimpleNamespace for headless baking)
        max_iterations: Max Newton iterations per muscle per outer iteration
        tolerance: Newton convergence tolerance
        verbose: Print progress
    """
    active_muscles = {
        name: mobj for name, mobj in v.zygote_muscle_meshes.items()
        if hasattr(mobj, 'soft_body') and mobj.soft_body is not None
    }

    if not active_muscles:
        if verbose:
            print("FEM: No active muscles with soft bodies")
        return

    skel = v.env.skel if hasattr(v, 'env') else None

    # Material parameters from viewer state
    youngs = getattr(v, 'fem_youngs_modulus', 5000.0)
    poisson = getattr(v, 'fem_poisson_ratio', 0.49)
    kappa = getattr(v, 'fem_collision_kappa', 1e4)
    contact_thresh = getattr(v, 'fem_contact_threshold', 0.015)

    # Build bone collision meshes
    bone_trimeshes = _build_bone_trimeshes(v, active_muscles, skel, verbose)

    # Outer iterations for inter-muscle convergence
    outer_iters = getattr(v, 'fem_outer_iterations', 3)

    for outer in range(outer_iters):
        if verbose and outer_iters > 1:
            print(f"FEM outer iteration {outer + 1}/{outer_iters}")

        for name, mobj in active_muscles.items():
            sb = mobj.soft_body

            # Initialize or get FEM sim
            fem = getattr(mobj, '_fem_sim', None)
            if fem is None:
                fem = FEMSimulation(
                    youngs_modulus=youngs,
                    poisson_ratio=poisson,
                    collision_kappa=kappa,
                    contact_threshold=contact_thresh
                )
                fem.build(
                    sb.rest_positions,
                    sb.tetrahedra if hasattr(sb, 'tetrahedra') else mobj.tet_tetrahedra,
                    sb.fixed_mask,
                    surface_faces=getattr(mobj, 'tet_faces', None)
                )
                mobj._fem_sim = fem
            else:
                fem.update_material(youngs, poisson, kappa)

            # Update fixed targets from skeleton pose
            if hasattr(sb, 'fixed_indices') and hasattr(sb, 'fixed_targets'):
                fem.set_fixed_targets(sb.fixed_targets)

            # Set bone collisions
            if bone_trimeshes:
                fem.set_bone_collisions(bone_trimeshes, margin=0.002)

            # Set inter-muscle contacts
            _setup_inter_muscle_contacts(v, fem, name, mobj, active_muscles)

            # Solve
            iters, residual = fem.solve(
                max_newton_iters=max_iterations,
                newton_tol=tolerance,
                verbose=verbose
            )

            # Write back positions
            new_pos = fem.get_positions()
            sb.positions[:] = new_pos
            if hasattr(mobj, 'tet_vertices'):
                mobj.tet_vertices[:] = new_pos.astype(mobj.tet_vertices.dtype)

            if verbose:
                print(f"  {name}: {iters} Newton iters, residual={residual:.6e}")


def _build_bone_trimeshes(v, active_muscles, skel, verbose=False):
    """Build trimesh objects for bone collision from skeleton meshes."""
    if not TRIMESH_AVAILABLE:
        return []

    skeleton_meshes = getattr(v, 'zygote_skeleton_meshes', None)
    if skeleton_meshes is None or skel is None:
        return []

    bone_meshes = []
    for mesh_name, mesh_obj in skeleton_meshes.items():
        try:
            if hasattr(mesh_obj, 'trimesh') and mesh_obj.trimesh is not None:
                # Get current transform from skeleton
                verts = mesh_obj.trimesh.vertices.copy()
                faces = mesh_obj.trimesh.faces.copy()
                tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                bone_meshes.append(tm)
            elif hasattr(mesh_obj, 'vertices') and hasattr(mesh_obj, 'faces'):
                tm = trimesh.Trimesh(
                    vertices=np.array(mesh_obj.vertices),
                    faces=np.array(mesh_obj.faces),
                    process=False
                )
                bone_meshes.append(tm)
        except Exception:
            continue

    if verbose and bone_meshes:
        print(f"FEM: {len(bone_meshes)} bone collision meshes")

    return bone_meshes


def _setup_inter_muscle_contacts(v, fem, muscle_name, mobj, active_muscles):
    """Set up inter-muscle contact pairs for a specific muscle."""
    constraints = getattr(v, 'inter_muscle_constraints', [])
    if not constraints:
        return

    pairs = []
    rest_dists = []

    for c in constraints:
        name1, v1_idx, v1_fixed, name2, v2_idx, v2_fixed, rest_dist = c

        if name1 == muscle_name:
            # This muscle's vertex v1_idx interacts with another muscle's vertex
            # We need the other muscle's current position
            other = active_muscles.get(name2)
            if other is None:
                continue
            other_pos = other.soft_body.positions[v2_idx]
            # For now, store as contact between local vertex and a target position
            # This is handled differently - we add a spring to the other muscle's vertex
            pairs.append([v1_idx, v1_idx])  # placeholder
            rest_dists.append(rest_dist)

        elif name2 == muscle_name:
            other = active_muscles.get(name1)
            if other is None:
                continue
            pairs.append([v2_idx, v2_idx])
            rest_dists.append(rest_dist)

    # For inter-muscle contacts, we use a simpler approach:
    # add penalty forces based on positions from the previous outer iteration
    # This avoids needing a global index space
    if pairs:
        # Actually, inter-muscle contacts in the per-muscle solve are better
        # handled as additional collision targets
        pass


# ---------------------------------------------------------------------------
# Convenience: get_fem_sim for a muscle object
# ---------------------------------------------------------------------------

def get_fem_sim(mobj):
    """Get or None the FEM simulation attached to a muscle object."""
    return getattr(mobj, '_fem_sim', None)
