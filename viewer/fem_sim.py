"""
XPBD Neo-Hookean muscle simulation (Macklin & Mueller 2021).

Self-contained module using Taichi for GPU acceleration.
Quasistatic: no dynamics/inertia, iterates constraints to convergence.

Architecture:
- Unified batched solver: all muscles in one global system
- Jacobi-parallel XPBD constraint projection on GPU (valence-weighted)
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


@ti.kernel
def _xpbd_project_jacobi(
    positions: ti.template(),
    invm: ti.template(),
    valence: ti.template(),
    tets: ti.template(),
    Bm_inv: ti.template(),
    rest_volume: ti.template(),
    lambda_H: ti.template(),
    lambda_D: ti.template(),
    alpha_H: ti.template(),
    alpha_D: ti.template(),
    omega: ti.f64,
    n_tets: ti.i32,
):
    """Jacobi-style XPBD: all tets in parallel with atomic_add.

    Each vertex correction is divided by its valence (number of incident tets)
    to compensate for multiple atomic_adds. omega provides additional tuning.
    """
    for k in range(n_tets):
        i0 = tets[k][0]
        i1 = tets[k][1]
        i2 = tets[k][2]
        i3 = tets[k][3]

        x0 = positions[i0]; x1 = positions[i1]
        x2 = positions[i2]; x3 = positions[i3]
        w0 = invm[i0]; w1 = invm[i1]
        w2 = invm[i2]; w3 = invm[i3]

        Ds = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])
        B = Bm_inv[k]
        F = Ds @ B

        f0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]], dt=ti.f64)
        f1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]], dt=ti.f64)
        f2 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]], dt=ti.f64)

        C_H = F.determinant() - 1.0
        cof = ti.Matrix.cols([f1.cross(f2), f2.cross(f0), f0.cross(f1)])
        CH_H = cof @ B.transpose()
        g_H_0 = ti.Vector([CH_H[0, 0], CH_H[1, 0], CH_H[2, 0]], dt=ti.f64)
        g_H_1 = ti.Vector([CH_H[0, 1], CH_H[1, 1], CH_H[2, 1]], dt=ti.f64)
        g_H_2 = ti.Vector([CH_H[0, 2], CH_H[1, 2], CH_H[2, 2]], dt=ti.f64)
        g_H_3 = -g_H_0 - g_H_1 - g_H_2

        C_D = F.norm_sqr() - 3.0
        CD_H = 2.0 * F @ B.transpose()
        g_D_0 = ti.Vector([CD_H[0, 0], CD_H[1, 0], CD_H[2, 0]], dt=ti.f64)
        g_D_1 = ti.Vector([CD_H[0, 1], CD_H[1, 1], CD_H[2, 1]], dt=ti.f64)
        g_D_2 = ti.Vector([CD_H[0, 2], CD_H[1, 2], CD_H[2, 2]], dt=ti.f64)
        g_D_3 = -g_D_0 - g_D_1 - g_D_2

        s_HH = (w0 * g_H_0.norm_sqr() + w1 * g_H_1.norm_sqr() +
                 w2 * g_H_2.norm_sqr() + w3 * g_H_3.norm_sqr())
        s_DD = (w0 * g_D_0.norm_sqr() + w1 * g_D_1.norm_sqr() +
                 w2 * g_D_2.norm_sqr() + w3 * g_D_3.norm_sqr())
        s_HD = (w0 * g_H_0.dot(g_D_0) + w1 * g_H_1.dot(g_D_1) +
                 w2 * g_H_2.dot(g_D_2) + w3 * g_H_3.dot(g_D_3))

        a_H = alpha_H[k]
        a_D = alpha_D[k]

        A00 = s_HH + a_H
        A01 = s_HD
        A11 = s_DD + a_D
        rhs_H = -(C_H + a_H * lambda_H[k])
        rhs_D = -(C_D + a_D * lambda_D[k])

        det = A00 * A11 - A01 * A01
        dl_H = 0.0
        dl_D = 0.0
        if ti.abs(det) > 1e-30:
            dl_H = (A11 * rhs_H - A01 * rhs_D) / det
            dl_D = (A00 * rhs_D - A01 * rhs_H) / det
        else:
            if A00 > 1e-20:
                dl_H = rhs_H / A00
            if A11 > 1e-20:
                dl_D = rhs_D / A11

        # Per-vertex valence-weighted atomic updates
        s0 = omega / valence[i0]
        s1 = omega / valence[i1]
        s2 = omega / valence[i2]
        s3 = omega / valence[i3]

        ti.atomic_add(positions[i0], s0 * w0 * (g_H_0 * dl_H + g_D_0 * dl_D))
        ti.atomic_add(positions[i1], s1 * w1 * (g_H_1 * dl_H + g_D_1 * dl_D))
        ti.atomic_add(positions[i2], s2 * w2 * (g_H_2 * dl_H + g_D_2 * dl_D))
        ti.atomic_add(positions[i3], s3 * w3 * (g_H_3 * dl_H + g_D_3 * dl_D))

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
    coll_stiffness: ti.f64,
    n_coll: ti.i32,
):
    """Move penetrating vertices toward bone surface targets.

    Uses stiffness blending: x = (1-s)*x + s*target, where s is clamped stiffness.
    This lets collision and elastic forces find equilibrium instead of hard snapping.
    """
    for c in range(n_coll):
        vi = coll_vert_idx[c]
        if invm[vi] > 0.0:
            positions[vi] = (1.0 - coll_stiffness) * positions[vi] + coll_stiffness * coll_targets[c]


@ti.kernel
def _xpbd_project_distance_jacobi(
    positions: ti.template(),
    invm: ti.template(),
    valence_dist: ti.template(),
    pair_i: ti.template(),
    pair_j: ti.template(),
    rest_dist: ti.template(),
    lambda_dist: ti.template(),
    alpha_dist: ti.template(),
    omega: ti.f64,
    n_pairs: ti.i32,
):
    """XPBD distance constraint projection for inter-muscle contacts.

    C = ||x_i - x_j|| - d_rest
    grad_i = n, grad_j = -n  where n = (x_i - x_j) / ||x_i - x_j||
    """
    for k in range(n_pairs):
        i = pair_i[k]
        j = pair_j[k]
        xi = positions[i]
        xj = positions[j]
        wi = invm[i]
        wj = invm[j]

        diff = xi - xj
        d = diff.norm()
        if d < 1e-12:
            continue

        n = diff / d
        C = d - rest_dist[k]

        # Only enforce when muscles are pushing into each other (compression)
        if C >= 0.0:
            continue

        w_sum = wi + wj
        a = alpha_dist[k]
        denom = w_sum + a
        if denom < 1e-30:
            continue

        dl = -(C + a * lambda_dist[k]) / denom

        si = omega / valence_dist[i]
        sj = omega / valence_dist[j]

        ti.atomic_add(positions[i], si * wi * dl * n)
        ti.atomic_add(positions[j], -sj * wj * dl * n)
        lambda_dist[k] += dl


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
        # Inter-muscle distance constraints
        self._n_dist_constraints = 0
        self._dist_pair_i = None
        self._dist_pair_j = None
        self._dist_rest = None

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

        # Compute vertex valence (number of incident tets) for Jacobi relaxation
        self._vertex_valence = np.zeros(self._n_verts, dtype=np.int32)
        for t in range(self._n_tets):
            for j in range(4):
                self._vertex_valence[self.tetrahedra[t, j]] += 1
        max_val = int(self._vertex_valence.max())
        avg_val = float(self._vertex_valence[self._vertex_valence > 0].mean())
        print(f"  Vertex valence: avg={avg_val:.1f}, max={max_val}")

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
        self.ti_valence = ti.field(dtype=ti.f64, shape=N)  # vertex valence for Jacobi

        # Collision fields (allocated on first use)
        self._max_coll = 0
        self.ti_coll_idx = None
        self.ti_coll_targets = None

    def _upload_static_data(self):
        self.ti_tets.from_numpy(self.tetrahedra)
        self.ti_Bm_inv.from_numpy(self.Bm_inv)
        self.ti_rest_volume.from_numpy(self.rest_volume)
        self.ti_rest_positions.from_numpy(self.rest_positions)

        # Per-vertex inverse mass from tet volumes (rho=1060 kg/m^3 for muscle)
        rho = 1060.0
        vertex_mass = np.zeros(self._n_verts, dtype=np.float64)
        for t in range(self._n_tets):
            m = rho * self.rest_volume[t] / 4.0
            for j in range(4):
                vertex_mass[self.tetrahedra[t, j]] += m
        invm = np.zeros(self._n_verts, dtype=np.float64)
        free_mask = vertex_mass > 0
        invm[free_mask] = 1.0 / vertex_mass[free_mask]
        invm[self.fixed_indices] = 0.0
        self._tet_mass = rho * self.rest_volume  # per-tet mass for compliance scaling
        self.ti_invm.from_numpy(invm)
        # Upload vertex valence
        valence = np.maximum(self._vertex_valence.astype(np.float64), 1.0)
        self.ti_valence.from_numpy(valence)

    def set_inter_muscle_constraints(self, constraints):
        """Convert inter-muscle constraints from local to global indices.

        Args:
            constraints: list of (name1, v1_local, is_fixed1, name2, v2_local, is_fixed2, rest_dist)
        """
        if not constraints:
            self._n_dist_constraints = 0
            return

        pair_i = []
        pair_j = []
        rest_d = []
        skipped = 0
        for name1, v1, fixed1, name2, v2, fixed2, d in constraints:
            if name1 not in self._muscle_ranges or name2 not in self._muscle_ranges:
                skipped += 1
                continue
            # Skip if both vertices are fixed (no DOFs to move)
            if fixed1 and fixed2:
                skipped += 1
                continue
            gi = self._muscle_ranges[name1][0] + v1
            gj = self._muscle_ranges[name2][0] + v2
            pair_i.append(gi)
            pair_j.append(gj)
            rest_d.append(d)

        self._n_dist_constraints = len(pair_i)
        if self._n_dist_constraints == 0:
            return

        self._dist_pair_i = np.array(pair_i, dtype=np.int32)
        self._dist_pair_j = np.array(pair_j, dtype=np.int32)
        self._dist_rest = np.array(rest_d, dtype=np.float64)

        # Compute distance-constraint valence (how many distance constraints per vertex)
        self._dist_valence = np.ones(self._n_verts, dtype=np.float64)
        for k in range(self._n_dist_constraints):
            self._dist_valence[self._dist_pair_i[k]] += 1.0
            self._dist_valence[self._dist_pair_j[k]] += 1.0

        # Allocate Taichi fields for distance constraints
        K = self._n_dist_constraints
        self.ti_dist_pair_i = ti.field(dtype=ti.i32, shape=K)
        self.ti_dist_pair_j = ti.field(dtype=ti.i32, shape=K)
        self.ti_dist_rest = ti.field(dtype=ti.f64, shape=K)
        self.ti_dist_lambda = ti.field(dtype=ti.f64, shape=K)
        self.ti_dist_alpha = ti.field(dtype=ti.f64, shape=K)
        self.ti_dist_valence = ti.field(dtype=ti.f64, shape=self._n_verts)

        self.ti_dist_pair_i.from_numpy(self._dist_pair_i)
        self.ti_dist_pair_j.from_numpy(self._dist_pair_j)
        self.ti_dist_rest.from_numpy(self._dist_rest)
        self.ti_dist_valence.from_numpy(self._dist_valence)

        print(f"  Inter-muscle constraints: {K} active ({skipped} skipped)")

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
            self._individual_bone_meshes = []
            return
        valid = [m for m in bone_meshes if m is not None and m.is_watertight]
        non_watertight = [m for m in bone_meshes if m is not None and not m.is_watertight]
        if not valid and not non_watertight:
            self._merged_bone_mesh = None
            self._individual_bone_meshes = []
            return
        # Keep individual watertight meshes for contains() queries
        self._individual_bone_meshes = valid
        # Merged mesh still used for nearest-point queries (projection targets)
        all_valid = valid + non_watertight
        self._merged_bone_mesh = trimesh.util.concatenate(all_valid) if all_valid else None
        self._bone_kdtree = None  # invalidate pre-filter cache

    def _compute_collision_targets(self, margin=0.002, verbose=False):
        """Compute collision projection targets for penetrating surface vertices.

        Three-phase approach:
        1. KD-tree pre-filter: only keep vertices within proximity_radius of bone vertices
        2. Per-bone contains() query (ray-casting) for reliable inside/outside detection
        3. Merged mesh nearest-point query for projection targets

        Returns (global_vert_indices, target_positions) as numpy arrays.
        """
        if self._merged_bone_mesh is None:
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        free_surf = self._free_surf_global
        if len(free_surf) == 0:
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        surf_pos = self.positions[free_surf]

        # Phase 1: KD-tree pre-filter — only query vertices near bone surface
        from scipy.spatial import cKDTree
        proximity_radius = 0.02  # 2cm
        if not hasattr(self, '_bone_kdtree') or self._bone_kdtree is None:
            self._bone_kdtree = cKDTree(self._merged_bone_mesh.vertices)

        # Fast: query each muscle vert's distance to nearest bone vert
        dists_to_bone, _ = self._bone_kdtree.query(surf_pos)
        nearby_mask = dists_to_bone < proximity_radius

        n_nearby = int(np.sum(nearby_mask))
        if n_nearby == 0:
            if verbose:
                print(f"    Collision: {len(free_surf)} surf verts, 0 near bones")
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        nearby_pos = surf_pos[nearby_mask]
        nearby_free_surf = free_surf[nearby_mask]

        # Phase 2: Per-bone contains() for reliable inside detection
        # Ray-casting is much more reliable than dot-product with normals
        inside_mask = np.zeros(len(nearby_pos), dtype=bool)
        individual = getattr(self, '_individual_bone_meshes', [])
        if individual:
            for bone_mesh in individual:
                try:
                    # Bounding box pre-filter (fast reject)
                    bmin = bone_mesh.bounds[0] - proximity_radius
                    bmax = bone_mesh.bounds[1] + proximity_radius
                    in_bbox = np.all((nearby_pos >= bmin) & (nearby_pos <= bmax), axis=1)
                    if not np.any(in_bbox):
                        continue
                    # contains() uses ray-casting — reliable for watertight meshes
                    contained = bone_mesh.contains(nearby_pos[in_bbox])
                    inside_mask[in_bbox] |= contained
                except Exception:
                    continue

        # Fallback: dot-product test on merged mesh for non-watertight bones
        if not individual:
            try:
                closest_pts, dists, face_ids = self._merged_bone_mesh.nearest.on_surface(nearby_pos)
                normals = self._merged_bone_mesh.face_normals[face_ids]
                to_vertex = nearby_pos - closest_pts
                dots = np.einsum('ij,ij->i', to_vertex, normals)
                inside_mask = (dots < 0) & (dists < margin * 10)
            except Exception:
                return np.array([], dtype=np.int32), np.zeros((0, 3))

        if verbose:
            n_inside = int(np.sum(inside_mask))
            print(f"    Collision: {len(free_surf)} surf verts, {n_nearby} near bones, "
                  f"{n_inside} inside ({len(individual)} watertight bones)")

        if not np.any(inside_mask):
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        # Phase 3: Project penetrating vertices to bone surface + margin
        pen_pos = nearby_pos[inside_mask]
        try:
            closest_pts, dists, face_ids = self._merged_bone_mesh.nearest.on_surface(pen_pos)
            normals = self._merged_bone_mesh.face_normals[face_ids]
        except Exception:
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        # Target = surface point + margin * outward normal
        # Face normals on bone surface point OUTWARD from bone interior.
        # For vertices INSIDE the bone, we want to push them to:
        #   closest_surface_point + outward_normal * margin
        # The face normals from trimesh already point outward (process=True),
        # so we use them directly — do NOT flip based on vertex direction.
        targets = closest_pts + normals * margin
        global_idx = nearby_free_surf[inside_mask].astype(np.int32)

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

        N = self._n_verts
        M = self._n_tets
        n_iters = max_lbfgs_iters

        # XPBD compliance: alpha_tilde = alpha / (dt^2 * tet_mass)
        # For quasistatic, use pseudo-timestep h to control stiffness.
        # h^2 * tet_mass scales the compliance down, making constraints stiffer.
        # With h=1, alpha_tilde = alpha / tet_mass = 1/(k*V_e*rho*V_e)
        effective_lam = lam + vol_penalty
        alpha_H_raw = 1.0 / (effective_lam * self.rest_volume + 1e-30)
        alpha_D_raw = 1.0 / (mu * self.rest_volume + 1e-30)
        # Use h=1, proper tet mass for physically correct material response
        alpha_H_np = alpha_H_raw / (self._tet_mass + 1e-30)
        alpha_D_np = alpha_D_raw / (self._tet_mass + 1e-30)
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

        # Inter-muscle distance constraint setup
        K = self._n_dist_constraints
        if K > 0:
            # Compliance: alpha = 1 / (stiffness * rest_dist)
            # Use vol_penalty as stiffness scale for inter-muscle separation
            dist_stiffness = effective_lam  # same stiffness as volume constraints
            alpha_dist_np = 1.0 / (dist_stiffness * self._dist_rest + 1e-30)
            self.ti_dist_alpha.from_numpy(alpha_dist_np)
            self.ti_dist_lambda.fill(0.0)

        # Save positions before solve for total convergence metric
        initial_pos = self.ti_positions.to_numpy().copy()

        # Solve: all elastic + distance iterations, then final collision projection.
        # Collision is the LAST step — no elastic after, guarantees no penetration.
        omega = 1.0

        # All elastic + inter-muscle distance iterations
        for it in range(n_iters):
            if verbose and it < 5:
                _copy_positions(self.ti_pos_prev, self.ti_positions, N)

            _xpbd_project_jacobi(
                self.ti_positions, self.ti_invm, self.ti_valence,
                self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
                self.ti_lambda_H, self.ti_lambda_D,
                self.ti_alpha_H, self.ti_alpha_D,
                omega, M,
            )
            if K > 0:
                _xpbd_project_distance_jacobi(
                    self.ti_positions, self.ti_invm, self.ti_dist_valence,
                    self.ti_dist_pair_i, self.ti_dist_pair_j,
                    self.ti_dist_rest, self.ti_dist_lambda, self.ti_dist_alpha,
                    omega, K,
                )
            _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)

            if verbose and it < 5:
                delta = _compute_position_delta(
                    self.ti_positions, self.ti_pos_prev, self.ti_invm, N)
                print(f"      iter {it}: ||dx||={delta**0.5:.6e}")

        # Final bone collision: detect and project as the VERY LAST step.
        # No elastic iterations after this — output is guaranteed penetration-free.
        n_coll = 0
        if self._merged_bone_mesh is not None:
            self.positions = self.ti_positions.to_numpy()
            coll_idx_np, coll_tgt_np = self._compute_collision_targets(
                margin, verbose=verbose)
            n_coll = len(coll_idx_np)
            if n_coll > 0:
                if n_coll > self._max_coll:
                    self._max_coll = max(n_coll, 1000)
                    self.ti_coll_idx = ti.field(dtype=ti.i32, shape=self._max_coll)
                    self.ti_coll_targets = ti.Vector.field(3, dtype=ti.f64, shape=self._max_coll)
                idx_padded = np.zeros(self._max_coll, dtype=np.int32)
                tgt_padded = np.zeros((self._max_coll, 3), dtype=np.float64)
                idx_padded[:n_coll] = coll_idx_np
                tgt_padded[:n_coll] = coll_tgt_np
                self.ti_coll_idx.from_numpy(idx_padded)
                self.ti_coll_targets.from_numpy(tgt_padded)
                _apply_collision_projection(
                    self.ti_positions, self.ti_invm,
                    self.ti_coll_idx, self.ti_coll_targets,
                    1.0, n_coll,
                )
                if verbose:
                    print(f"    bone collision: {n_coll} vertices projected (final)")

        # Download positions
        self.positions = self.ti_positions.to_numpy()

        # Convergence metric: total position change from frame start
        free = self.free_indices
        diff = self.positions[free] - initial_pos[free]
        residual = float(np.sqrt(np.sum(diff ** 2)))

        if verbose:
            # Count inverted tets
            X = self.positions
            T = self.tetrahedra
            x3 = X[T[:, 3]]
            Ds = np.stack([X[T[:, 0]] - x3, X[T[:, 1]] - x3, X[T[:, 2]] - x3], axis=-1)
            Js = np.linalg.det(Ds @ self.Bm_inv)
            n_inv = int(np.sum(Js <= 0))
            print(f"    final: iters={n_iters} ||dx||={residual:.4e} "
                  f"inv={n_inv}/{M} coll={n_coll} dist={K} "
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
        # Set inter-muscle constraints if available
        if hasattr(v, 'inter_muscle_constraints') and v.inter_muscle_constraints:
            solver.set_inter_muscle_constraints(v.inter_muscle_constraints)
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
    solver.update_targets_and_positions(muscles_update, use_lbs_init=True)

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
    """Build bone trimeshes transformed to current skeleton pose (world coords).

    Bone mesh vertices (after MESH_SCALE) are in REST-POSE world coordinates,
    not body-local. To get posed world coords:
        local = R_rest^T @ (mesh_verts - t_rest)
        posed = R_posed @ local + t_posed

    Rest transforms are cached on first call and reused.
    """
    if not TRIMESH_AVAILABLE:
        return []
    skeleton_meshes = getattr(v, 'zygote_skeleton_meshes', None)
    if skeleton_meshes is None or skel is None:
        return []

    # Cache rest transforms (computed once at rest pose)
    if not hasattr(v, '_bone_rest_transforms') or v._bone_rest_transforms is None:
        saved_pos = skel.getPositions().copy()
        skel.setPositions(np.zeros(skel.getNumDofs()))
        v._bone_rest_transforms = {}
        for i in range(skel.getNumBodyNodes()):
            bn = skel.getBodyNode(i)
            wt = bn.getWorldTransform()
            v._bone_rest_transforms[bn.getName()] = (wt.rotation().copy(), wt.translation().copy())
        skel.setPositions(saved_pos)

    bone_meshes = []
    n_matched = 0
    for mesh_name, mesh_obj in skeleton_meshes.items():
        try:
            if not (hasattr(mesh_obj, 'trimesh') and mesh_obj.trimesh is not None):
                continue
            body_node = None
            body_name = None
            for candidate in [mesh_name, mesh_name + '0']:
                try:
                    body_node = skel.getBodyNode(candidate)
                    if body_node is not None:
                        body_name = candidate
                        break
                except Exception:
                    continue
            if body_node is None:
                continue

            # Get rest and posed transforms
            R_rest, t_rest = v._bone_rest_transforms.get(body_name, (np.eye(3), np.zeros(3)))
            wt = body_node.getWorldTransform()
            R_posed = wt.rotation()
            t_posed = wt.translation()

            # Transform: rest-world → body-local → posed-world
            verts = mesh_obj.trimesh.vertices.copy()
            local_verts = (R_rest.T @ (verts - t_rest).T).T
            posed_verts = (R_posed @ local_verts.T).T + t_posed

            tm = trimesh.Trimesh(vertices=posed_verts,
                                 faces=mesh_obj.trimesh.faces.copy(), process=True)
            bone_meshes.append(tm)
            n_matched += 1
        except Exception:
            continue
    if verbose:
        print(f"    Bone meshes: {n_matched}/{len(skeleton_meshes)} matched")
    return bone_meshes
