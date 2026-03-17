"""
XPBD Neo-Hookean muscle simulation (Macklin & Mueller 2021).

Self-contained module using Taichi for GPU acceleration.
Quasistatic: no dynamics/inertia, iterates constraints to convergence.

Architecture:
- Unified batched solver: all muscles in one global system
- XPBD constraint projection on GPU with graph coloring for parallel Gauss-Seidel
- Coupled hydrostatic (C_H = det(F)-1) + deviatoric (C_D = ||F||^2-3) per tet
- Bone collision via position-level projection (not energy penalty)
- Merged bone collision mesh for single BVH query
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
# Graph coloring for parallel Gauss-Seidel
# ---------------------------------------------------------------------------

def _greedy_graph_color(n_tets, tetrahedra, n_verts):
    """Greedy graph coloring: tets sharing a vertex get different colors.

    Returns (colors, color_groups) where color_groups[c] = (start, end)
    indices into the sorted-by-color tet order.
    """
    # Build vertex -> tet adjacency
    vert_to_tets = [[] for _ in range(n_verts)]
    for t in range(n_tets):
        for j in range(4):
            vert_to_tets[tetrahedra[t, j]].append(t)

    colors = -np.ones(n_tets, dtype=np.int32)
    for t in range(n_tets):
        # Collect colors of neighboring tets (sharing any vertex)
        neighbor_colors = set()
        for j in range(4):
            v = tetrahedra[t, j]
            for nb in vert_to_tets[v]:
                if colors[nb] >= 0:
                    neighbor_colors.add(colors[nb])
        # Assign smallest available color
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[t] = c

    n_colors = int(colors.max()) + 1
    sorted_order = np.argsort(colors, kind='stable')
    sorted_colors = colors[sorted_order]

    color_groups = []
    start = 0
    for c in range(n_colors):
        end = start
        while end < n_tets and sorted_colors[end] == c:
            end += 1
        color_groups.append((start, end))
        start = end

    return sorted_order, color_groups


# ---------------------------------------------------------------------------
# Taichi XPBD kernels
# ---------------------------------------------------------------------------

@ti.kernel
def _xpbd_project_color_group(
    positions: ti.template(),     # ti.Vector.field(3, f64, N)
    invm: ti.template(),          # ti.field(f64, N) — 0 for fixed
    tets: ti.template(),          # ti.Vector.field(4, i32, M) — color-sorted
    Bm_inv: ti.template(),        # ti.Matrix.field(3,3, f64, M) — rest inv
    rest_volume: ti.template(),   # ti.field(f64, M)
    lambda_H: ti.template(),      # ti.field(f64, M) — hydrostatic multiplier
    lambda_D: ti.template(),      # ti.field(f64, M) — deviatoric multiplier
    alpha_H: ti.template(),       # ti.field(f64, M) — hydrostatic compliance
    alpha_D: ti.template(),       # ti.field(f64, M) — deviatoric compliance
    start: ti.i32, end: ti.i32,
):
    """Project coupled Neo-Hookean constraints for one color group.

    C_H = det(F) - 1       (volume preservation)
    C_D = ||F||^2_F - 3    (shape preservation / shear)

    Coupled 2x2 block solve for delta_lambda_H, delta_lambda_D.
    """
    for k in range(start, end):
        i0 = tets[k][0]
        i1 = tets[k][1]
        i2 = tets[k][2]
        i3 = tets[k][3]

        x0 = positions[i0]; x1 = positions[i1]
        x2 = positions[i2]; x3 = positions[i3]
        w0 = invm[i0]; w1 = invm[i1]
        w2 = invm[i2]; w3 = invm[i3]

        # Deformation gradient: F = Ds @ Bm_inv
        # Ds = [x0-x3, x1-x3, x2-x3] (column vectors)
        Ds = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])
        B = Bm_inv[k]
        F = Ds @ B

        f0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]], dt=ti.f64)
        f1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]], dt=ti.f64)
        f2 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]], dt=ti.f64)

        # --- Hydrostatic: C_H = det(F) - 1 ---
        C_H = F.determinant() - 1.0

        # grad(det(F)) w.r.t. F = cofactor matrix
        # cofactor columns = cross products of F columns
        cof = ti.Matrix.cols([f1.cross(f2), f2.cross(f0), f0.cross(f1)])

        # Chain rule: dC_H/dx = cof @ B^T, distributed to vertices
        # For vertices 0,1,2: columns of (cof @ B^T)
        # Vertex 3 gets negative sum
        CH_H = cof @ B.transpose()
        g_H_0 = ti.Vector([CH_H[0, 0], CH_H[1, 0], CH_H[2, 0]], dt=ti.f64)
        g_H_1 = ti.Vector([CH_H[0, 1], CH_H[1, 1], CH_H[2, 1]], dt=ti.f64)
        g_H_2 = ti.Vector([CH_H[0, 2], CH_H[1, 2], CH_H[2, 2]], dt=ti.f64)
        g_H_3 = -g_H_0 - g_H_1 - g_H_2

        # --- Deviatoric: C_D = ||F||^2_F - 3 ---
        C_D = F.norm_sqr() - 3.0

        # grad(||F||^2) w.r.t. F = 2*F
        # Chain rule: dC_D/dx = 2*F @ B^T
        CD_H = 2.0 * F @ B.transpose()
        g_D_0 = ti.Vector([CD_H[0, 0], CD_H[1, 0], CD_H[2, 0]], dt=ti.f64)
        g_D_1 = ti.Vector([CD_H[0, 1], CD_H[1, 1], CD_H[2, 1]], dt=ti.f64)
        g_D_2 = ti.Vector([CD_H[0, 2], CD_H[1, 2], CD_H[2, 2]], dt=ti.f64)
        g_D_3 = -g_D_0 - g_D_1 - g_D_2

        # Weighted gradient norms
        s_HH = (w0 * g_H_0.norm_sqr() + w1 * g_H_1.norm_sqr() +
                 w2 * g_H_2.norm_sqr() + w3 * g_H_3.norm_sqr())
        s_DD = (w0 * g_D_0.norm_sqr() + w1 * g_D_1.norm_sqr() +
                 w2 * g_D_2.norm_sqr() + w3 * g_D_3.norm_sqr())
        s_HD = (w0 * g_H_0.dot(g_D_0) + w1 * g_H_1.dot(g_D_1) +
                 w2 * g_H_2.dot(g_D_2) + w3 * g_H_3.dot(g_D_3))

        a_H = alpha_H[k]
        a_D = alpha_D[k]

        # Coupled 2x2 system:
        # [s_HH + a_H,  s_HD      ] [dl_H]   [-(C_H + a_H * lam_H)]
        # [s_HD,         s_DD + a_D] [dl_D] = [-(C_D + a_D * lam_D)]
        A00 = s_HH + a_H
        A01 = s_HD
        A11 = s_DD + a_D
        rhs_H = -(C_H + a_H * lambda_H[k])
        rhs_D = -(C_D + a_D * lambda_D[k])

        det = A00 * A11 - A01 * A01
        if ti.abs(det) > 1e-30:
            dl_H = (A11 * rhs_H - A01 * rhs_D) / det
            dl_D = (A00 * rhs_D - A01 * rhs_H) / det
        else:
            # Fallback: solve decoupled
            dl_H = rhs_H / (A00 + 1e-20) if A00 > 1e-20 else 0.0
            dl_D = rhs_D / (A11 + 1e-20) if A11 > 1e-20 else 0.0

        # Apply position corrections
        positions[i0] += w0 * (g_H_0 * dl_H + g_D_0 * dl_D)
        positions[i1] += w1 * (g_H_1 * dl_H + g_D_1 * dl_D)
        positions[i2] += w2 * (g_H_2 * dl_H + g_D_2 * dl_D)
        positions[i3] += w3 * (g_H_3 * dl_H + g_D_3 * dl_D)

        lambda_H[k] += dl_H
        lambda_D[k] += dl_D


@ti.kernel
def _snap_fixed(positions: ti.template(), targets: ti.template(),
                invm: ti.template(), n: ti.i32):
    """Snap fixed vertices (invm=0) to their targets."""
    for i in range(n):
        if invm[i] == 0.0:
            positions[i] = targets[i]


@ti.kernel
def _apply_collision_projection(
    positions: ti.template(),
    invm: ti.template(),
    coll_vert_idx: ti.template(),
    coll_targets: ti.template(),
    n_coll: ti.i32,
):
    """Project penetrating vertices onto bone surface (position-level)."""
    for c in range(n_coll):
        vi = coll_vert_idx[c]
        if invm[vi] > 0.0:
            positions[vi] = coll_targets[c]


@ti.kernel
def _compute_max_displacement(
    positions: ti.template(), rest_positions: ti.template(),
    invm: ti.template(), n: ti.i32,
) -> ti.f64:
    """Max displacement of free vertices from rest (convergence metric)."""
    max_d = 0.0
    for i in range(n):
        if invm[i] > 0.0:
            d = (positions[i] - rest_positions[i]).norm()
            ti.atomic_max(max_d, d)
    return max_d


@ti.kernel
def _compute_position_delta(
    pos_new: ti.template(), pos_old: ti.template(),
    invm: ti.template(), n: ti.i32,
) -> ti.f64:
    """Sum of squared position changes for free vertices."""
    s = 0.0
    for i in range(n):
        if invm[i] > 0.0:
            d = pos_new[i] - pos_old[i]
            ti.atomic_add(s, d.norm_sqr())
    return s


@ti.kernel
def _copy_positions(dst: ti.template(), src: ti.template(), n: ti.i32):
    for i in range(n):
        dst[i] = src[i]


# ---------------------------------------------------------------------------
# Unified Batched XPBD Solver
# ---------------------------------------------------------------------------

class UnifiedFEMSolver:
    """
    Batched quasistatic XPBD solver with Neo-Hookean constraints.
    GPU-parallel Gauss-Seidel via graph coloring.
    Bone collision as position-level projection.
    """

    def __init__(self):
        self._built = False
        self._n_verts = 0
        self._n_tets = 0
        self._muscle_ranges = {}
        self._merged_bone_mesh = None
        self._orig_fixed_mask = None
        self._color_groups = []

    def build(self, muscles_data):
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
        self.tetrahedra = np.concatenate(all_tets, axis=0)
        self._orig_fixed_mask = np.concatenate(all_fixed, axis=0)
        self.fixed_mask = self._orig_fixed_mask.copy()
        self.surface_verts = np.concatenate(all_surface_verts, axis=0)

        self.fixed_indices = np.where(self.fixed_mask)[0]
        self.free_indices = np.where(~self.fixed_mask)[0]
        self.fixed_targets = self.positions[self.fixed_indices].copy()

        # Free surface verts -- collision checks
        self.free_surface_verts = self.surface_verts[~self._orig_fixed_mask[self.surface_verts]]
        free_set = set(self.free_indices.tolist())
        self._free_surf_global = np.array([
            gv for gv in self.free_surface_verts if gv in free_set
        ], dtype=np.int64)

        # Filter degenerate tets
        self._filter_degenerate_tets()
        self._precompute_rest_state()

        # Graph coloring for parallel Gauss-Seidel
        t_color = time.time()
        self._sorted_order, self._color_groups = _greedy_graph_color(
            self._n_tets, self.tetrahedra, self._n_verts)
        # Reorder tetrahedra by color
        self.tetrahedra = self.tetrahedra[self._sorted_order].copy()
        self.Bm_inv = self.Bm_inv[self._sorted_order].copy()
        self.rest_volume = self.rest_volume[self._sorted_order].copy()
        dt_color = time.time() - t_color
        print(f"  Graph coloring: {len(self._color_groups)} colors, {dt_color:.2f}s")

        # Allocate Taichi fields
        self._allocate_fields()
        self._upload_static_data()

        self._built = True
        print(f"  XPBD unified: {self._n_verts} verts, {self._n_tets} tets, "
              f"{len(self._muscle_ranges)} muscles, {len(self.free_indices)} free DOFs")

    def _filter_degenerate_tets(self):
        X = self.rest_positions
        tets = self.tetrahedra
        v0 = X[tets[:, 0]]
        e1 = X[tets[:, 1]] - v0
        e2 = X[tets[:, 2]] - v0
        e3 = X[tets[:, 3]] - v0
        Dm = np.stack([e1, e2, e3], axis=-1)
        vol = np.abs(np.linalg.det(Dm)) / 6.0
        edge_scale = np.mean(np.linalg.norm(e1, axis=1)) ** 3
        good = vol > edge_scale * 1e-10
        n_removed = np.sum(~good)
        if n_removed > 0:
            print(f"  XPBD: Removed {n_removed} degenerate tets (of {len(tets)})")
            self.tetrahedra = tets[good].copy()
        self._n_tets = len(self.tetrahedra)

    def _precompute_rest_state(self):
        """Compute Bm_inv (rest edge matrix inverse) and rest volumes.

        Convention: Ds = [x0-x3, x1-x3, x2-x3] (matching PBD_Taichi).
        """
        tets = self.tetrahedra
        X = self.rest_positions
        x3 = X[tets[:, 3]]
        Bm = np.stack([X[tets[:, 0]] - x3, X[tets[:, 1]] - x3, X[tets[:, 2]] - x3], axis=-1)
        self.Bm_inv = np.linalg.inv(Bm)
        self.rest_volume = np.abs(np.linalg.det(Bm)) / 6.0

    def _allocate_fields(self):
        N = self._n_verts
        M = self._n_tets

        self.ti_positions = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_pos_prev = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_rest_positions = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_targets = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_invm = ti.field(dtype=ti.f64, shape=N)

        self.ti_tets = ti.Vector.field(4, dtype=ti.i32, shape=M)
        self.ti_Bm_inv = ti.Matrix.field(3, 3, dtype=ti.f64, shape=M)
        self.ti_rest_volume = ti.field(dtype=ti.f64, shape=M)
        self.ti_lambda_H = ti.field(dtype=ti.f64, shape=M)
        self.ti_lambda_D = ti.field(dtype=ti.f64, shape=M)
        self.ti_alpha_H = ti.field(dtype=ti.f64, shape=M)
        self.ti_alpha_D = ti.field(dtype=ti.f64, shape=M)

        # Collision fields (allocated on first use)
        self._max_coll = 0
        self.ti_coll_idx = None
        self.ti_coll_targets = None

    def _upload_static_data(self):
        self.ti_tets.from_numpy(self.tetrahedra)
        self.ti_Bm_inv.from_numpy(self.Bm_inv)
        self.ti_rest_volume.from_numpy(self.rest_volume)
        self.ti_rest_positions.from_numpy(self.rest_positions)

        # Inverse mass: uniform 1.0 for free, 0.0 for fixed
        invm = np.ones(self._n_verts, dtype=np.float64)
        invm[self.fixed_indices] = 0.0
        self.ti_invm.from_numpy(invm)

    def update_targets_and_positions(self, muscles_data, use_lbs_init=False):
        """Update fixed targets from skeleton. Free verts keep previous solution."""
        for name, data in muscles_data.items():
            vs, ve, _, _ = self._muscle_ranges[name]
            if use_lbs_init and 'positions' in data:
                self.positions[vs:ve] = data['positions']
            if 'fixed_targets' in data:
                fi = np.where(self._orig_fixed_mask[vs:ve])[0]
                self.positions[vs + fi] = data['fixed_targets']
                global_fi = fi + vs
                for i, gfi in enumerate(global_fi):
                    idx_in_fixed = np.searchsorted(self.fixed_indices, gfi)
                    if idx_in_fixed < len(self.fixed_indices) and self.fixed_indices[idx_in_fixed] == gfi:
                        self.fixed_targets[idx_in_fixed] = data['fixed_targets'][i]

    def _build_merged_bone_mesh(self, bone_meshes):
        if not bone_meshes:
            self._merged_bone_mesh = None
            return
        valid = [m for m in bone_meshes if m is not None]
        if not valid:
            self._merged_bone_mesh = None
            return
        self._merged_bone_mesh = trimesh.util.concatenate(valid)

    def _compute_collision_targets(self, margin=0.002):
        """Compute collision projection targets for penetrating surface vertices.

        Returns (global_vert_indices, target_positions) as numpy arrays.
        """
        if self._merged_bone_mesh is None:
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        free_surf = self._free_surf_global
        if len(free_surf) == 0:
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        surf_pos = self.positions[free_surf]
        try:
            closest_pts, dists, face_ids = self._merged_bone_mesh.nearest.on_surface(surf_pos)
            normals = self._merged_bone_mesh.face_normals[face_ids]
        except Exception:
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        to_vertex = surf_pos - closest_pts
        dots = np.einsum('ij,ij->i', to_vertex, normals)

        # Penetrating = inside bone (dot < 0), excluding rest-pose penetrations
        pen_mask = dots < 0
        if hasattr(self, '_rest_inside') and len(self._rest_inside) == len(dots):
            pen_mask = pen_mask & ~self._rest_inside

        if not np.any(pen_mask):
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        # Target = surface point + margin * outward normal
        targets = closest_pts[pen_mask] + normals[pen_mask] * margin
        global_idx = free_surf[pen_mask].astype(np.int32)

        return global_idx, targets

    def solve(self, mu, lam, vol_penalty=100.0, kappa=1e4, margin=0.002,
              max_lbfgs_iters=100, n_outer=3, bone_meshes=None, verbose=False,
              n_load_steps=0):
        """XPBD quasistatic solve.

        Args:
            mu, lam: Lame parameters
            vol_penalty: additional volume stiffness (added to lam)
            kappa: not used (collision is position-level now)
            margin: collision surface margin
            max_lbfgs_iters: number of XPBD iterations (reusing param name for API compat)
            bone_meshes: list of trimesh objects for collision
            verbose: print detailed output
        """
        if bone_meshes is not None:
            self._build_merged_bone_mesh(bone_meshes)

        # Detect rest-pose penetrations to exclude
        if self._merged_bone_mesh is not None and len(self._free_surf_global) > 0:
            rest_pos = self.rest_positions[self._free_surf_global]
            try:
                cp, d, fi = self._merged_bone_mesh.nearest.on_surface(rest_pos)
                normals = self._merged_bone_mesh.face_normals[fi]
                tv = rest_pos - cp
                dots = np.einsum('ij,ij->i', tv, normals)
                self._rest_inside = dots < 0
            except Exception:
                self._rest_inside = np.zeros(len(self._free_surf_global), dtype=bool)
        else:
            self._rest_inside = np.zeros(0, dtype=bool)

        N = self._n_verts
        M = self._n_tets
        n_iters = max_lbfgs_iters

        # Compute per-tet compliance from material params
        # alpha_H = 1 / ((lam + vol_penalty) * V_e)
        # alpha_D = 1 / (mu * V_e)
        effective_lam = lam + vol_penalty
        alpha_H_np = 1.0 / (effective_lam * self.rest_volume + 1e-30)
        alpha_D_np = 1.0 / (mu * self.rest_volume + 1e-30)
        self.ti_alpha_H.from_numpy(alpha_H_np)
        self.ti_alpha_D.from_numpy(alpha_D_np)

        # Upload current positions and targets
        targets_full = self.positions.copy()
        targets_full[self.fixed_indices] = self.fixed_targets
        self.ti_targets.from_numpy(targets_full)
        self.ti_positions.from_numpy(self.positions)
        _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)

        # Reset Lagrange multipliers
        self.ti_lambda_H.fill(0.0)
        self.ti_lambda_D.fill(0.0)

        # Collision detection (CPU, once before solve)
        coll_idx_np, coll_tgt_np = self._compute_collision_targets(margin)
        n_coll = len(coll_idx_np)
        if n_coll > 0:
            if n_coll > self._max_coll:
                self._max_coll = max(n_coll, 1000)
                self.ti_coll_idx = ti.field(dtype=ti.i32, shape=self._max_coll)
                self.ti_coll_targets = ti.Vector.field(3, dtype=ti.f64, shape=self._max_coll)
            self.ti_coll_idx.from_numpy(coll_idx_np)
            self.ti_coll_targets.from_numpy(coll_tgt_np)

        # Save positions before solve for convergence check
        _copy_positions(self.ti_pos_prev, self.ti_positions, N)

        # XPBD iteration loop
        for it in range(n_iters):
            # Project constraints by color group (parallel Gauss-Seidel)
            for (start, end) in self._color_groups:
                _xpbd_project_color_group(
                    self.ti_positions, self.ti_invm,
                    self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
                    self.ti_lambda_H, self.ti_lambda_D,
                    self.ti_alpha_H, self.ti_alpha_D,
                    start, end,
                )

            # Snap fixed vertices
            _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)

            # Collision projection (every few iterations to amortize CPU cost)
            if n_coll > 0 and it % 5 == 4:
                _apply_collision_projection(
                    self.ti_positions, self.ti_invm,
                    self.ti_coll_idx, self.ti_coll_targets, n_coll,
                )

        # Final collision projection
        if n_coll > 0:
            _apply_collision_projection(
                self.ti_positions, self.ti_invm,
                self.ti_coll_idx, self.ti_coll_targets, n_coll,
            )

        # Download positions
        self.positions = self.ti_positions.to_numpy()

        # Convergence metric: position change
        delta_sq = _compute_position_delta(
            self.ti_positions, self.ti_pos_prev, self.ti_invm, N)
        residual = float(delta_sq ** 0.5)

        if verbose:
            # Count inverted tets
            X = self.positions
            T = self.tetrahedra
            x3 = X[T[:, 3]]
            Ds = np.stack([X[T[:, 0]] - x3, X[T[:, 1]] - x3, X[T[:, 2]] - x3], axis=-1)
            Js = np.linalg.det(Ds @ self.Bm_inv)
            n_inv = int(np.sum(Js <= 0))
            print(f"    final: iters={n_iters} ||dx||={residual:.4e} "
                  f"inv={n_inv}/{M} coll={n_coll} "
                  f"J=[{np.min(Js):.3f},{np.max(Js):.3f}]")

        return n_iters, residual

    def get_muscle_positions(self, name):
        vs, ve, _, _ = self._muscle_ranges[name]
        return self.positions[vs:ve].copy()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_fem_sim(v, max_iterations=100, tolerance=1e-4, verbose=True):
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
    youngs = getattr(v, 'fem_youngs_modulus', 5000.0)
    poisson = getattr(v, 'fem_poisson_ratio', 0.49)
    vol_penalty = getattr(v, 'fem_volume_penalty', 100.0)
    kappa = getattr(v, 'fem_collision_kappa', 1e4)
    mu = youngs / (2.0 * (1.0 + poisson))
    lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

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

    # Update each muscle's positions and fixed targets from current skeleton pose
    skeleton_meshes = getattr(v, 'zygote_skeleton_meshes', None)
    for name, mobj in active_muscles.items():
        if hasattr(mobj, '_update_tet_positions_from_skeleton') and skel is not None:
            mobj._update_tet_positions_from_skeleton(skel)
        if hasattr(mobj, '_update_fixed_targets_from_skeleton') and skel is not None:
            mobj._update_fixed_targets_from_skeleton(skeleton_meshes, skel)

    muscles_update = {}
    max_ft_disp = 0.0
    for name, mobj in active_muscles.items():
        sb = mobj.soft_body
        muscles_update[name] = {
            'positions': sb.positions,
            'fixed_targets': sb.fixed_targets,
        }
        if sb.fixed_targets is not None and sb.initial_fixed_targets is not None:
            d = np.max(np.linalg.norm(sb.fixed_targets - sb.initial_fixed_targets, axis=1))
            if d > max_ft_disp:
                max_ft_disp = d
    if verbose and max_ft_disp > 0:
        print(f"  Max fixed target displacement from rest: {max_ft_disp:.6f}m")
    solver.update_targets_and_positions(muscles_update)

    bone_meshes = _build_bone_trimeshes(v, active_muscles, skel, verbose=False)
    t_setup = time.time()

    fevals, residual = solver.solve(
        mu=mu, lam=lam, vol_penalty=vol_penalty, kappa=kappa,
        max_lbfgs_iters=max_iterations,
        bone_meshes=bone_meshes, verbose=verbose
    )

    t_solve = time.time()
    for name, mobj in active_muscles.items():
        sb = mobj.soft_body
        new_pos = solver.get_muscle_positions(name)
        sb.positions[:] = new_pos
        if hasattr(mobj, 'tet_vertices'):
            mobj.tet_vertices[:] = new_pos.astype(mobj.tet_vertices.dtype)

    total = time.time() - t0
    if verbose:
        print(f"  XPBD: {fevals} iters, ||dx||={residual:.4e}, "
              f"setup={t_setup-t0:.2f}s solve={t_solve-t_setup:.2f}s total={total:.2f}s")
    else:
        print(f"  XPBD: {fevals}it ||dx||={residual:.2e} {total:.1f}s")


def _build_bone_trimeshes(v, active_muscles, skel, verbose=False):
    """Build bone trimeshes transformed to current skeleton pose (world coords)."""
    if not TRIMESH_AVAILABLE:
        return []
    skeleton_meshes = getattr(v, 'zygote_skeleton_meshes', None)
    if skeleton_meshes is None or skel is None:
        return []
    bone_meshes = []
    for mesh_name, mesh_obj in skeleton_meshes.items():
        try:
            if not (hasattr(mesh_obj, 'trimesh') and mesh_obj.trimesh is not None):
                continue
            body_node = None
            try:
                body_node = skel.getBodyNode(mesh_name)
            except Exception:
                pass
            if body_node is None:
                for variant in [mesh_name.replace('_L', '_l').replace('_R', '_r'),
                                mesh_name.replace('_l', '_L').replace('_r', '_R')]:
                    try:
                        body_node = skel.getBodyNode(variant)
                        if body_node is not None:
                            break
                    except Exception:
                        continue
            if body_node is None:
                continue
            wt = body_node.getWorldTransform()
            R = wt.rotation()
            t = wt.translation()
            verts = mesh_obj.trimesh.vertices.copy()
            transformed_verts = (R @ verts.T).T + t
            tm = trimesh.Trimesh(vertices=transformed_verts,
                                 faces=mesh_obj.trimesh.faces.copy(), process=False)
            bone_meshes.append(tm)
        except Exception:
            continue
    return bone_meshes
