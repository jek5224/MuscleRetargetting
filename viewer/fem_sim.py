"""
FEM-based muscle simulation with Neo-Hookean material model + volume preservation.

Self-contained module using Taichi for GPU acceleration.
Handles muscle-bone collisions (direct projection) and elastic relaxation.

Architecture:
- Unified batched solver: all muscles concatenated into one global system
- L-BFGS optimizer (~10-20 iters vs 100+ for gradient descent)
- Merged bone collision mesh for single BVH query
- Degenerate tet filtering
- Quasistatic: deterministic given same boundary conditions
"""

import time
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


_ti_initialized = False


def _ensure_ti_init():
    global _ti_initialized
    if _ti_initialized:
        return
    if not TAICHI_AVAILABLE:
        raise RuntimeError("Taichi is required for FEM simulation")
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f64)
    except RuntimeError:
        pass  # Already initialized
    _ti_initialized = True


# ---------------------------------------------------------------------------
# Taichi kernels
# ---------------------------------------------------------------------------

@ti.kernel
def _compute_gradient_and_energy(
    positions: ti.template(), tets: ti.template(),
    Dm_inv: ti.template(), rest_volume: ti.template(),
    gradient: ti.template(), energy: ti.template(),
    mu: ti.f64, lam: ti.f64, vol_penalty: ti.f64, n_tets: ti.i32
):
    """
    Fused Neo-Hookean gradient + energy + volume preservation in one pass.
    Volume penalty: vol_penalty * (J - 1)^2 per tet (additional to Neo-Hookean).
    """
    for t in range(n_tets):
        i0 = tets[t][0]
        i1 = tets[t][1]
        i2 = tets[t][2]
        i3 = tets[t][3]

        x0 = positions[i0]
        e1 = positions[i1] - x0
        e2 = positions[i2] - x0
        e3 = positions[i3] - x0
        Ds = ti.Matrix([
            [e1[0], e2[0], e3[0]],
            [e1[1], e2[1], e3[1]],
            [e1[2], e2[2], e3[2]]
        ], dt=ti.f64)

        F = Ds @ Dm_inv[t]
        J = F.determinant()
        J_safe = ti.max(J, 1e-6)
        logJ = ti.log(J_safe)

        # --- Energy ---
        I1 = (F.transpose() @ F).trace()
        # Neo-Hookean
        W = mu / 2.0 * (I1 - 3.0) - mu * logJ + lam / 2.0 * logJ * logJ
        # Volume preservation penalty
        W += vol_penalty * (J_safe - 1.0) * (J_safe - 1.0)
        ti.atomic_add(energy[None], rest_volume[t] * W)

        # --- Gradient (Piola-Kirchhoff stress) ---
        # Cofactor matrix for F^{-T}
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

        # P = mu*(F - F^{-T}) + lam*log(J)*F^{-T} + 2*vol_penalty*(J-1)*cofactor(F)
        P = mu * (F - F_inv_T) + lam * logJ * F_inv_T
        P += 2.0 * vol_penalty * (J_safe - 1.0) * cof

        H = rest_volume[t] * P @ Dm_inv[t].transpose()

        h0 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]], dt=ti.f64)
        h1 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]], dt=ti.f64)
        h2 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]], dt=ti.f64)

        ti.atomic_add(gradient[i1], h0)
        ti.atomic_add(gradient[i2], h1)
        ti.atomic_add(gradient[i3], h2)
        ti.atomic_add(gradient[i0], -(h0 + h1 + h2))


@ti.kernel
def _compute_energy_only(
    positions: ti.template(), tets: ti.template(),
    Dm_inv: ti.template(), rest_volume: ti.template(),
    energy: ti.template(), mu: ti.f64, lam: ti.f64,
    vol_penalty: ti.f64, n_tets: ti.i32
):
    """Energy only for line search (no gradient)."""
    for t in range(n_tets):
        i0 = tets[t][0]
        x0 = positions[i0]
        e1 = positions[tets[t][1]] - x0
        e2 = positions[tets[t][2]] - x0
        e3 = positions[tets[t][3]] - x0
        Ds = ti.Matrix([
            [e1[0], e2[0], e3[0]],
            [e1[1], e2[1], e3[1]],
            [e1[2], e2[2], e3[2]]
        ], dt=ti.f64)
        F = Ds @ Dm_inv[t]
        J = F.determinant()
        J_safe = ti.max(J, 1e-6)
        logJ = ti.log(J_safe)
        I1 = (F.transpose() @ F).trace()
        W = mu / 2.0 * (I1 - 3.0) - mu * logJ + lam / 2.0 * logJ * logJ
        W += vol_penalty * (J_safe - 1.0) * (J_safe - 1.0)
        ti.atomic_add(energy[None], rest_volume[t] * W)


@ti.kernel
def _zero_grad_snap_fixed(
    gradient: ti.template(), positions: ti.template(),
    targets: ti.template(), fixed_mask: ti.template(), n: ti.i32
):
    for i in range(n):
        if fixed_mask[i] == 1:
            gradient[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
            positions[i] = targets[i]


@ti.kernel
def _apply_step(
    positions: ti.template(), direction: ti.template(),
    fixed_mask: ti.template(), alpha: ti.f64, n: ti.i32
):
    """x += alpha * direction for free vertices only."""
    for i in range(n):
        if fixed_mask[i] == 0:
            positions[i] += alpha * direction[i]


@ti.kernel
def _set_positions_from_numpy_offset(
    positions: ti.template(), data: ti.types.ndarray(), offset: ti.i32, count: ti.i32
):
    """Upload positions from numpy to a slice of the field."""
    for i in range(count):
        for d in ti.static(range(3)):
            positions[offset + i][d] = data[i, d]


@ti.kernel
def _zero_field(field: ti.template(), n: ti.i32):
    for i in range(n):
        field[i] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)


# ---------------------------------------------------------------------------
# L-BFGS Optimizer (CPU, operating on flat numpy vectors)
# ---------------------------------------------------------------------------

class LBFGS:
    """Limited-memory BFGS for large-scale unconstrained optimization.

    Operates on the free-vertex DOFs only (3 * n_free floats).
    Stores m correction pairs for the inverse Hessian approximation.
    """

    def __init__(self, n_free, m=10):
        self.n = n_free * 3  # total DOFs
        self.m = m
        self.s_list = []  # position differences
        self.y_list = []  # gradient differences
        self.rho_list = []  # 1 / (y^T s)
        self.prev_x = None
        self.prev_g = None

    def reset(self):
        self.s_list.clear()
        self.y_list.clear()
        self.rho_list.clear()
        self.prev_x = None
        self.prev_g = None

    def update(self, x, g):
        """Store a new correction pair from (x_k, g_k)."""
        if self.prev_x is not None:
            s = x - self.prev_x
            y = g - self.prev_g
            sy = s.dot(y)
            if sy > 1e-20:
                if len(self.s_list) >= self.m:
                    self.s_list.pop(0)
                    self.y_list.pop(0)
                    self.rho_list.pop(0)
                self.s_list.append(s.copy())
                self.y_list.append(y.copy())
                self.rho_list.append(1.0 / sy)
        self.prev_x = x.copy()
        self.prev_g = g.copy()

    def direction(self, g):
        """Compute search direction d = -H_k * g via two-loop recursion."""
        q = g.copy()
        k = len(self.s_list)
        if k == 0:
            # No history yet — use scaled steepest descent
            return -q

        alpha_list = np.zeros(k)

        # Backward pass
        for i in range(k - 1, -1, -1):
            alpha_list[i] = self.rho_list[i] * self.s_list[i].dot(q)
            q -= alpha_list[i] * self.y_list[i]

        # Initial Hessian approximation: H0 = (s^T y) / (y^T y) * I
        sy = self.s_list[-1].dot(self.y_list[-1])
        yy = self.y_list[-1].dot(self.y_list[-1])
        gamma = sy / max(yy, 1e-20)
        r = gamma * q

        # Forward pass
        for i in range(k):
            beta = self.rho_list[i] * self.y_list[i].dot(r)
            r += (alpha_list[i] - beta) * self.s_list[i]

        return -r


# ---------------------------------------------------------------------------
# Unified Batched FEM Solver
# ---------------------------------------------------------------------------

class UnifiedFEMSolver:
    """
    Batched FEM solver: all muscles in one global system.
    L-BFGS optimizer. Direct collision projection.
    Quasistatic: deterministic output for same boundary conditions.
    """

    def __init__(self):
        self._built = False
        self._n_verts = 0
        self._n_tets = 0
        self._muscle_ranges = {}
        self._merged_bone_mesh = None

    def build(self, muscles_data):
        """
        Build unified system from multiple muscles.

        Args:
            muscles_data: dict of name -> {
                'rest_positions': (N,3) float64,
                'tetrahedra': (M,4) int32,
                'fixed_mask': (N,) bool,
                'surface_faces': (F,3) int32 or None,
            }
        """
        _ensure_ti_init()

        all_verts = []
        all_tets = []
        all_fixed = []
        all_surface_verts = []
        vert_offset = 0
        tet_offset = 0

        for name, data in muscles_data.items():
            n_v = len(data['rest_positions'])
            n_t = len(data['tetrahedra'])

            all_verts.append(data['rest_positions'].astype(np.float64))
            all_tets.append(data['tetrahedra'].astype(np.int32) + vert_offset)
            all_fixed.append(data['fixed_mask'].astype(bool))

            if data.get('surface_faces') is not None:
                surf_v = np.unique(data['surface_faces'].ravel()) + vert_offset
            else:
                surf_v = np.arange(vert_offset, vert_offset + n_v)
            all_surface_verts.append(surf_v)

            self._muscle_ranges[name] = (vert_offset, vert_offset + n_v,
                                         tet_offset, tet_offset + n_t)
            vert_offset += n_v
            tet_offset += n_t

        self._n_verts = vert_offset

        self.rest_positions = np.concatenate(all_verts, axis=0)
        self.positions = self.rest_positions.copy()
        all_tets_np = np.concatenate(all_tets, axis=0)
        self.fixed_mask = np.concatenate(all_fixed, axis=0)
        self.surface_verts = np.concatenate(all_surface_verts, axis=0)

        self.fixed_indices = np.where(self.fixed_mask)[0]
        self.free_mask = ~self.fixed_mask
        self.free_indices = np.where(self.free_mask)[0]
        self.fixed_targets = self.positions[self.fixed_indices].copy()

        # Free surface verts (for collision detection)
        self.free_surface_verts = self.surface_verts[self.free_mask[self.surface_verts]]

        # --- Filter degenerate tets ---
        self.tetrahedra, self._n_tets = self._filter_degenerate_tets(all_tets_np)

        # Precompute rest state
        self._precompute_rest_state()

        # Allocate Taichi fields
        self._allocate_fields()
        self._upload_static_data()

        # L-BFGS optimizer (operates on free DOFs only)
        self._lbfgs = LBFGS(len(self.free_indices), m=10)

        self._built = True
        print(f"  FEM unified: {self._n_verts} verts, {self._n_tets} tets, "
              f"{len(muscles_data)} muscles, {len(self.free_indices)} free DOFs")

    def _filter_degenerate_tets(self, tets):
        """Remove tets with near-zero rest volume."""
        X = self.rest_positions
        v0 = X[tets[:, 0]]
        e1 = X[tets[:, 1]] - v0
        e2 = X[tets[:, 2]] - v0
        e3 = X[tets[:, 3]] - v0

        Dm = np.stack([e1, e2, e3], axis=-1)
        vol = np.abs(np.linalg.det(Dm)) / 6.0

        # Keep tets with reasonable volume (> 1e-12 m^3)
        good = vol > 1e-12
        n_removed = np.sum(~good)
        if n_removed > 0:
            print(f"  FEM: Removed {n_removed} degenerate tets "
                  f"(of {len(tets)}, kept {np.sum(good)})")

        return tets[good].copy(), int(np.sum(good))

    def _precompute_rest_state(self):
        tets = self.tetrahedra
        X = self.rest_positions

        v0 = X[tets[:, 0]]
        e1 = X[tets[:, 1]] - v0
        e2 = X[tets[:, 2]] - v0
        e3 = X[tets[:, 3]] - v0

        Dm = np.stack([e1, e2, e3], axis=-1)
        self.Dm_inv = np.linalg.inv(Dm)
        self.rest_volume = np.abs(np.linalg.det(Dm)) / 6.0

    def _allocate_fields(self):
        N = self._n_verts
        M = self._n_tets

        self.ti_positions = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_gradient = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_direction = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_fixed_mask = ti.field(dtype=ti.i32, shape=N)
        self.ti_fixed_targets = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_tets = ti.Vector.field(4, dtype=ti.i32, shape=M)
        self.ti_Dm_inv = ti.Matrix.field(3, 3, dtype=ti.f64, shape=M)
        self.ti_rest_volume = ti.field(dtype=ti.f64, shape=M)
        self.ti_energy = ti.field(dtype=ti.f64, shape=())

    def _upload_static_data(self):
        self.ti_tets.from_numpy(self.tetrahedra)
        self.ti_Dm_inv.from_numpy(self.Dm_inv)
        self.ti_rest_volume.from_numpy(self.rest_volume)
        self.ti_fixed_mask.from_numpy(self.fixed_mask.astype(np.int32))

    def update_targets_and_positions(self, muscles_data):
        """Upload current positions and fixed targets for all muscles."""
        for name, data in muscles_data.items():
            vs, ve, _, _ = self._muscle_ranges[name]
            if 'positions' in data:
                self.positions[vs:ve] = data['positions']
            if 'fixed_targets' in data:
                fi = np.where(self.fixed_mask[vs:ve])[0]
                self.positions[vs + fi] = data['fixed_targets']
                global_fi = fi + vs
                for i, gfi in enumerate(global_fi):
                    idx_in_fixed = np.searchsorted(self.fixed_indices, gfi)
                    if idx_in_fixed < len(self.fixed_indices) and self.fixed_indices[idx_in_fixed] == gfi:
                        self.fixed_targets[idx_in_fixed] = data['fixed_targets'][i]

        targets_full = np.zeros((self._n_verts, 3), dtype=np.float64)
        targets_full[self.fixed_indices] = self.fixed_targets
        self.ti_fixed_targets.from_numpy(targets_full)
        self.ti_positions.from_numpy(self.positions)

    def _build_merged_bone_mesh(self, bone_meshes):
        """Merge all bone trimeshes into one for fast single-query BVH."""
        if not bone_meshes:
            self._merged_bone_mesh = None
            return
        valid = [m for m in bone_meshes if m is not None]
        if not valid:
            self._merged_bone_mesh = None
            return
        self._merged_bone_mesh = trimesh.util.concatenate(valid)

    def resolve_collisions(self, margin=0.002, max_rounds=5):
        """
        Direct projection: push penetrating vertices outside bone surface.
        Modifies self.positions in-place (CPU). Returns number pushed.
        """
        if self._merged_bone_mesh is None:
            return 0

        free_surf = self.free_surface_verts
        if len(free_surf) == 0:
            return 0

        mesh = self._merged_bone_mesh
        total_pushed = 0

        for _ in range(max_rounds):
            surf_pos = self.positions[free_surf]
            try:
                closest_pts, dists, face_ids = mesh.nearest.on_surface(surf_pos)
                normals = mesh.face_normals[face_ids]
            except Exception:
                break

            to_vertex = surf_pos - closest_pts
            dots = np.einsum('ij,ij->i', to_vertex, normals)
            penetrating = (dists < margin) & (dots < 0)
            n_pen = np.sum(penetrating)

            if n_pen == 0:
                break

            pen_idx = np.where(penetrating)[0]
            self.positions[free_surf[pen_idx]] = (
                closest_pts[pen_idx] + normals[pen_idx] * margin
            )
            total_pushed += n_pen

        return total_pushed

    def _eval_energy(self, mu, lam, vol_penalty):
        """Compute energy at current ti_positions."""
        self.ti_energy[None] = 0.0
        _compute_energy_only(
            self.ti_positions, self.ti_tets, self.ti_Dm_inv,
            self.ti_rest_volume, self.ti_energy, mu, lam,
            vol_penalty, self._n_tets
        )
        return self.ti_energy[None]

    def _eval_gradient_and_energy(self, mu, lam, vol_penalty):
        """Compute gradient + energy at current ti_positions. Returns (grad_free, energy)."""
        N = self._n_verts
        _zero_field(self.ti_gradient, N)
        self.ti_energy[None] = 0.0
        _compute_gradient_and_energy(
            self.ti_positions, self.ti_tets, self.ti_Dm_inv,
            self.ti_rest_volume, self.ti_gradient, self.ti_energy,
            mu, lam, vol_penalty, self._n_tets
        )
        _zero_grad_snap_fixed(
            self.ti_gradient, self.ti_positions,
            self.ti_fixed_targets, self.ti_fixed_mask, N
        )

        # Download gradient for free vertices only
        grad_full = self.ti_gradient.to_numpy()  # (N, 3)
        grad_free = grad_full[self.free_indices].ravel()  # (n_free * 3,)
        energy = self.ti_energy[None]
        return grad_free, energy

    def _get_free_positions(self):
        """Download free vertex positions as flat vector."""
        pos = self.ti_positions.to_numpy()
        return pos[self.free_indices].ravel()

    def _set_free_positions(self, x_flat):
        """Upload free vertex positions from flat vector."""
        x = x_flat.reshape(-1, 3)
        self.positions[self.free_indices] = x
        self.ti_positions.from_numpy(self.positions)

    def solve(self, mu, lam, vol_penalty=100.0, max_iters=30, tol=1e-4,
              bone_meshes=None, bone_margin=0.002, verbose=False):
        """
        Solve sequence per frame:
        1. Direct collision projection (CPU, merged mesh)
        2. L-BFGS elastic minimization (GPU gradient, CPU optimizer)
        3. Post-solve collision re-check
        """
        N = self._n_verts
        M = self._n_tets

        # Phase 1: merge bone meshes (once) and resolve collisions
        if bone_meshes is not None:
            self._build_merged_bone_mesh(bone_meshes)
        n_pushed = self.resolve_collisions(bone_margin)
        if verbose and n_pushed > 0:
            print(f"    Collision: projected {n_pushed} vertices")

        # Upload post-collision positions
        self.ti_positions.from_numpy(self.positions)

        # Phase 2: L-BFGS elastic minimization
        self._lbfgs.reset()
        grad_norm = 0.0
        x_free = self._get_free_positions()

        for it in range(max_iters):
            # Evaluate gradient and energy
            g_free, energy = self._eval_gradient_and_energy(mu, lam, vol_penalty)
            grad_norm = np.linalg.norm(g_free)

            if verbose and (it < 3 or it % 5 == 0 or grad_norm < tol):
                print(f"    iter {it}: |grad|={grad_norm:.4e} E={energy:.6e}")

            if grad_norm < tol:
                break

            # L-BFGS: update history and compute search direction
            self._lbfgs.update(x_free, g_free)
            d_free = self._lbfgs.direction(g_free)

            # Check descent direction
            dg = d_free.dot(g_free)
            if dg >= 0:
                # Not a descent direction — fall back to negative gradient
                d_free = -g_free
                dg = -grad_norm * grad_norm
                self._lbfgs.reset()

            # Backtracking line search (Armijo condition)
            alpha = 1.0
            c_armijo = 1e-4
            target_decrease = c_armijo * dg  # negative number

            for ls in range(12):
                # Trial: x_trial = x_free + alpha * d_free
                x_trial = x_free + alpha * d_free
                self._set_free_positions(x_trial)

                E_trial = self._eval_energy(mu, lam, vol_penalty)

                if E_trial <= energy + alpha * target_decrease:
                    break
                alpha *= 0.5
            else:
                # Line search failed — accept smallest step
                x_trial = x_free + alpha * d_free
                self._set_free_positions(x_trial)

            x_free = x_trial

        # Download final positions
        self.positions = self.ti_positions.to_numpy()

        # Phase 3: post-solve collision check
        if self._merged_bone_mesh is not None:
            n_pushed2 = self.resolve_collisions(bone_margin)
            if verbose and n_pushed2 > 0:
                print(f"    Post-solve: projected {n_pushed2} more vertices")

        return it + 1, grad_norm

    def get_muscle_positions(self, name):
        """Extract positions for a single muscle."""
        vs, ve, _, _ = self._muscle_ranges[name]
        return self.positions[vs:ve].copy()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_fem_sim(v, max_iterations=30, tolerance=1e-4, verbose=True):
    """
    Run unified FEM simulation for all muscles in one batched solve.
    """
    t0 = time.time()

    active_muscles = {
        name: mobj for name, mobj in v.zygote_muscle_meshes.items()
        if hasattr(mobj, 'soft_body') and mobj.soft_body is not None
    }

    if not active_muscles:
        if verbose:
            print("FEM: No active muscles with soft bodies")
        return

    skel = v.env.skel if hasattr(v, 'env') else None

    # Material parameters
    youngs = getattr(v, 'fem_youngs_modulus', 5000.0)
    poisson = getattr(v, 'fem_poisson_ratio', 0.49)
    vol_penalty = getattr(v, 'fem_volume_penalty', 100.0)
    mu = youngs / (2.0 * (1.0 + poisson))
    lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    # Get or build unified solver
    solver = getattr(v, '_fem_unified_solver', None)
    if solver is None or set(solver._muscle_ranges.keys()) != set(active_muscles.keys()):
        solver = UnifiedFEMSolver()
        muscles_data = {}
        for name, mobj in active_muscles.items():
            sb = mobj.soft_body
            muscles_data[name] = {
                'rest_positions': sb.rest_positions,
                'tetrahedra': sb.tetrahedra if hasattr(sb, 'tetrahedra') else mobj.tet_tetrahedra,
                'fixed_mask': sb.fixed_mask,
                'surface_faces': getattr(mobj, 'tet_faces', None),
            }
        solver.build(muscles_data)
        v._fem_unified_solver = solver

    # Update current positions + fixed targets
    muscles_update = {}
    for name, mobj in active_muscles.items():
        sb = mobj.soft_body
        muscles_update[name] = {
            'positions': sb.positions,
            'fixed_targets': sb.fixed_targets,
        }
    solver.update_targets_and_positions(muscles_update)

    # Build bone collision meshes
    bone_meshes = _build_bone_trimeshes(v, active_muscles, skel, verbose=False)

    t_setup = time.time()

    # Solve
    iters, residual = solver.solve(
        mu=mu, lam=lam, vol_penalty=vol_penalty,
        max_iters=max_iterations, tol=tolerance,
        bone_meshes=bone_meshes, bone_margin=0.002,
        verbose=verbose
    )

    t_solve = time.time()

    # Write back positions to each muscle
    for name, mobj in active_muscles.items():
        sb = mobj.soft_body
        new_pos = solver.get_muscle_positions(name)
        sb.positions[:] = new_pos
        if hasattr(mobj, 'tet_vertices'):
            mobj.tet_vertices[:] = new_pos.astype(mobj.tet_vertices.dtype)

    if verbose:
        total = time.time() - t0
        print(f"  FEM: {iters} iters, |grad|={residual:.4e}, "
              f"setup={t_setup-t0:.2f}s solve={t_solve-t_setup:.2f}s total={total:.2f}s")


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

    return bone_meshes
