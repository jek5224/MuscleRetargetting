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

import os
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


def _triangle_intersection_pairs(mesh_a, mesh_b, eps=1e-10):
    """Return exact crossing face pairs after an R-tree broad phase.

    Vertex-containment tests miss the common case where two triangle interiors
    cross but all six vertices remain outside the opposite closed surface.
    """
    tri_a = np.asarray(mesh_a.triangles, dtype=np.float64)
    tri_b = np.asarray(mesh_b.triangles, dtype=np.float64)
    if not len(tri_a) or not len(tri_b):
        return np.zeros((0, 2), dtype=np.int32)

    candidate_a, candidate_b = [], []
    tree_b = mesh_b.triangles_tree
    bounds_a = np.column_stack((tri_a.min(axis=1), tri_a.max(axis=1)))
    for ia, bounds in enumerate(bounds_a):
        hits = list(tree_b.intersection(bounds))
        if hits:
            candidate_a.extend([ia] * len(hits))
            candidate_b.extend(hits)
    if not candidate_a:
        return np.zeros((0, 2), dtype=np.int32)
    ia = np.asarray(candidate_a, dtype=np.int32)
    ib = np.asarray(candidate_b, dtype=np.int32)
    ta, tb = tri_a[ia], tri_b[ib]

    def segments_hit_triangles(start, end, target):
        direction = end - start
        edge1 = target[:, 1] - target[:, 0]
        edge2 = target[:, 2] - target[:, 0]
        h = np.cross(direction, edge2)
        det = np.einsum('ij,ij->i', edge1, h)
        valid = np.abs(det) > eps
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        s = start - target[:, 0]
        u = inv_det * np.einsum('ij,ij->i', s, h)
        q = np.cross(s, edge1)
        v = inv_det * np.einsum('ij,ij->i', direction, q)
        t = inv_det * np.einsum('ij,ij->i', edge2, q)
        return (valid & (u >= -eps) & (v >= -eps)
                & (u + v <= 1.0 + eps)
                & (t > eps) & (t < 1.0 - eps))

    crossing = np.zeros(len(ia), dtype=bool)
    for e0, e1 in ((0, 1), (1, 2), (2, 0)):
        crossing |= segments_hit_triangles(ta[:, e0], ta[:, e1], tb)
        crossing |= segments_hit_triangles(tb[:, e0], tb[:, e1], ta)
    return np.column_stack((ia[crossing], ib[crossing])).astype(np.int32)


def _dilated_crossing_face_masks(mesh_a, mesh_b, rings=2):
    """Rest-pose interface masks used to distinguish sliding from new tangles."""
    pairs = _triangle_intersection_pairs(mesh_a, mesh_b)
    mask_a = np.zeros(len(mesh_a.faces), dtype=bool)
    mask_b = np.zeros(len(mesh_b.faces), dtype=bool)
    if len(pairs):
        mask_a[pairs[:, 0]] = True
        mask_b[pairs[:, 1]] = True
    for mesh, mask in ((mesh_a, mask_a), (mesh_b, mask_b)):
        adjacency = np.asarray(mesh.face_adjacency, dtype=np.int32)
        for _ in range(rings):
            if not len(adjacency):
                break
            grow = mask[adjacency[:, 0]] | mask[adjacency[:, 1]]
            if not np.any(grow):
                break
            mask[adjacency[grow].ravel()] = True
    return mask_a, mask_b, len(pairs)


def _ensure_ti_init():
    global _ti_initialized
    if _ti_initialized:
        return
    if not TAICHI_AVAILABLE:
        raise RuntimeError("Taichi is required for FEM simulation")
    try:
        import os
        arch_name = os.environ.get('MUSCLE_TAICHI_ARCH', 'cuda').lower()
        arch = ti.cpu if arch_name == 'cpu' else ti.gpu
        kwargs = dict(arch=arch, default_fp=ti.f64,
                      offline_cache=(arch_name != 'cpu'))
        if arch_name == 'cpu':
            kwargs['offline_cache_file_path'] = '/tmp/muscle_taichi_cache'
        ti.init(**kwargs)
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
def _xpbd_project_gauss_seidel(
    positions: ti.template(),
    invm: ti.template(),
    color_tets: ti.template(),
    tets: ti.template(),
    Bm_inv: ti.template(),
    rest_volume: ti.template(),
    lambda_H: ti.template(),
    lambda_D: ti.template(),
    alpha_H: ti.template(),
    alpha_D: ti.template(),
    target_J: ti.template(),
    n_color_tets: ti.i32,
):
    """Gauss-Seidel XPBD: one color's tets in parallel, direct updates.

    Within a color, no two tets share vertices, so corrections are independent.
    No atomic_add needed, no valence division — exact corrections.
    """
    for ck in range(n_color_tets):
        k = color_tets[ck]
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

        # Volume barrier: for inverted tets (J < 0), make hydrostatic
        # constraint 10x stiffer to actively push J back positive.
        J_val = F.determinant()
        if J_val < 0.0:
            a_H = a_H * 0.1  # 10x stiffer

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

        # Compute correction vectors
        dx0 = w0 * (g_H_0 * dl_H + g_D_0 * dl_D)
        dx1 = w1 * (g_H_1 * dl_H + g_D_1 * dl_D)
        dx2 = w2 * (g_H_2 * dl_H + g_D_2 * dl_D)
        dx3 = w3 * (g_H_3 * dl_H + g_D_3 * dl_D)

        # Inversion-safe line search: check if correction would make J < threshold.
        # If so, scale correction back to keep J >= threshold.
        J_threshold = 0.01
        # Tentative new positions
        x0_new = x0 + dx0; x1_new = x1 + dx1
        x2_new = x2 + dx2; x3_new = x3 + dx3
        Ds_new = ti.Matrix.cols([x0_new - x3_new, x1_new - x3_new, x2_new - x3_new])
        J_new = (Ds_new @ B).determinant()

        t = 1.0  # scale factor for correction
        if J_new < J_threshold and J_val > J_threshold:
            # J would cross threshold. Binary search for safe scale.
            t_lo = 0.0
            t_hi = 1.0
            for _ in range(6):  # 6 bisection steps → 1/64 precision
                t_mid = (t_lo + t_hi) * 0.5
                xm0 = x0 + t_mid * dx0; xm1 = x1 + t_mid * dx1
                xm2 = x2 + t_mid * dx2; xm3 = x3 + t_mid * dx3
                Ds_mid = ti.Matrix.cols([xm0 - xm3, xm1 - xm3, xm2 - xm3])
                J_mid = (Ds_mid @ B).determinant()
                if J_mid >= J_threshold:
                    t_lo = t_mid
                else:
                    t_hi = t_mid
            t = t_lo * 0.9  # safety margin
        elif J_val <= J_threshold:
            # Already inverted — allow full correction to try recovery.
            # The GS coloring ensures no conflicts with neighbors.
            t = 1.0

        # Apply scaled correction
        positions[i0] += t * dx0
        positions[i1] += t * dx1
        positions[i2] += t * dx2
        positions[i3] += t * dx3

        lambda_H[k] += t * dl_H
        lambda_D[k] += t * dl_D


@ti.kernel
def _recover_inversions(
    positions: ti.template(),
    invm: ti.template(),
    tets: ti.template(),
    Bm_inv: ti.template(),
    n_tets: ti.i32,
):
    """Fix inverted tets by moving free vertices toward tet centroid.

    For each tet with J < 0.01, move each free vertex toward the tet's
    centroid. This geometrically "uncrushs" the tet. Stronger correction
    for more severely inverted tets.
    """
    for k in range(n_tets):
        i0 = tets[k][0]; i1 = tets[k][1]
        i2 = tets[k][2]; i3 = tets[k][3]
        x0 = positions[i0]; x1 = positions[i1]
        x2 = positions[i2]; x3 = positions[i3]

        Ds = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])
        F = Ds @ Bm_inv[k]
        J = F.determinant()

        if J < 0.01:
            centroid = (x0 + x1 + x2 + x3) * 0.25
            # Correction strength: stronger for more inverted
            strength = ti.min(0.3, 0.1 * (0.01 - J))
            if strength < 0.001:
                continue
            if invm[i0] > 0.0:
                positions[i0] += strength * (centroid - x0)
            if invm[i1] > 0.0:
                positions[i1] += strength * (centroid - x1)
            if invm[i2] > 0.0:
                positions[i2] += strength * (centroid - x2)
            if invm[i3] > 0.0:
                positions[i3] += strength * (centroid - x3)


@ti.kernel
def _svd_recover_inversions_color(
    positions: ti.template(),
    invm: ti.template(),
    color_tets: ti.template(),
    tets: ti.template(),
    Bm_inv: ti.template(),
    rest_volume: ti.template(),
    blend: ti.f64,
    n_color_tets: ti.i32,
):
    """SVD-based inversion recovery for one color's tets.

    For each inverted tet (J < 0.01):
    1. Decompose F = U Σ V^T
    2. Clamp negative singular values to ε (fix reflection)
    3. Reconstruct target deformation gradient F_fixed
    4. Compute target vertex positions from F_fixed
    5. Blend current positions toward target (blend factor)

    Same-color tets share no vertices → parallel-safe.
    """
    for ck in range(n_color_tets):
        k = color_tets[ck]
        i0 = tets[k][0]; i1 = tets[k][1]
        i2 = tets[k][2]; i3 = tets[k][3]
        x0 = positions[i0]; x1 = positions[i1]
        x2 = positions[i2]; x3 = positions[i3]

        Ds = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])
        B = Bm_inv[k]
        F = Ds @ B
        J = F.determinant()

        if J >= 0.01:
            continue

        # SVD: F = U @ diag(sig) @ V^T
        U, sig, V = ti.svd(F, ti.f64)

        # Clamp singular values: flip negative (inverted) to positive
        eps_sv = 0.01
        sig_fixed = ti.Matrix.zero(ti.f64, 3, 3)
        sig_fixed[0, 0] = ti.max(sig[0, 0], eps_sv)
        sig_fixed[1, 1] = ti.max(sig[1, 1], eps_sv)
        sig_fixed[2, 2] = ti.max(sig[2, 2], eps_sv)

        # Ensure positive determinant (proper rotation, not reflection)
        det_UV = (U @ V.transpose()).determinant()
        if det_UV < 0.0:
            # Flip smallest singular value to fix reflection
            if sig_fixed[0, 0] <= sig_fixed[1, 1] and sig_fixed[0, 0] <= sig_fixed[2, 2]:
                sig_fixed[0, 0] = -sig_fixed[0, 0]
            elif sig_fixed[1, 1] <= sig_fixed[2, 2]:
                sig_fixed[1, 1] = -sig_fixed[1, 1]
            else:
                sig_fixed[2, 2] = -sig_fixed[2, 2]

        # Reconstruct fixed F and target Ds
        F_fixed = U @ sig_fixed @ V.transpose()

        # Ds_target = F_fixed @ Bm  where Bm = Bm_inv^{-1}
        # Since Ds = [x0-x3, x1-x3, x2-x3] and F = Ds @ Bm_inv,
        # Ds_target = F_fixed @ Bm_inv^{-1}
        # But we can compute target edge vectors directly:
        # Ds_target[:,j] = F_fixed @ Bm^{-1}[:,j] ... that's circular.
        # Instead: Ds_target = F_fixed @ Bm where Bm = inv(Bm_inv)
        Bm = B.inverse()
        Ds_target = F_fixed @ Bm

        # Target positions: x0_t = x3 + Ds_target[:,0], etc.
        t0 = x3 + ti.Vector([Ds_target[0, 0], Ds_target[1, 0], Ds_target[2, 0]])
        t1 = x3 + ti.Vector([Ds_target[0, 1], Ds_target[1, 1], Ds_target[2, 1]])
        t2 = x3 + ti.Vector([Ds_target[0, 2], Ds_target[1, 2], Ds_target[2, 2]])
        # x3 stays (it's the reference vertex)

        # Blend toward target
        if invm[i0] > 0.0:
            positions[i0] += blend * (t0 - x0)
        if invm[i1] > 0.0:
            positions[i1] += blend * (t1 - x1)
        if invm[i2] > 0.0:
            positions[i2] += blend * (t2 - x2)
        # Don't move i3 — it's the reference for Ds convention


@ti.kernel
def _recover_inversions_color(
    positions: ti.template(),
    invm: ti.template(),
    color_tets: ti.template(),
    tets: ti.template(),
    Bm_inv: ti.template(),
    n_color_tets: ti.i32,
):
    """Fix inverted tets in one color by moving free vertices toward centroid.

    Same-color tets share no vertices, so parallel is safe.
    """
    for ck in range(n_color_tets):
        k = color_tets[ck]
        i0 = tets[k][0]; i1 = tets[k][1]
        i2 = tets[k][2]; i3 = tets[k][3]
        x0 = positions[i0]; x1 = positions[i1]
        x2 = positions[i2]; x3 = positions[i3]

        Ds = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])
        F = Ds @ Bm_inv[k]
        J = F.determinant()

        if J < 0.01:
            centroid = (x0 + x1 + x2 + x3) * 0.25
            strength = ti.min(0.3, 0.05 * ti.max(0.01 - J, 0.0))
            if strength > 0.001:
                if invm[i0] > 0.0:
                    positions[i0] += strength * (centroid - x0)
                if invm[i1] > 0.0:
                    positions[i1] += strength * (centroid - x1)
                if invm[i2] > 0.0:
                    positions[i2] += strength * (centroid - x2)
                if invm[i3] > 0.0:
                    positions[i3] += strength * (centroid - x3)


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
def _freeze_collision_vertices(
    invm: ti.template(),
    targets: ti.template(),
    positions: ti.template(),
    coll_vert_idx: ti.template(),
    coll_targets: ti.template(),
    n_coll: ti.i32,
):
    """Freeze colliding vertices: set invm=0 and snap to collision target.

    After mid-solve collision projection, this prevents subsequent VBD
    elastic iterations from pushing vertices back into bone.
    """
    for c in range(n_coll):
        vi = coll_vert_idx[c]
        if invm[vi] > 0.0:
            invm[vi] = 0.0
            targets[vi] = coll_targets[c]
            positions[vi] = coll_targets[c]


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

        # Dead zone: skip micro-corrections that cause jaggedness
        tolerance = 0.002  # 2mm
        if ti.abs(C) < tolerance:
            continue

        w_sum = wi + wj
        # Asymmetric stiffness: stiff compression, softer separation
        a = alpha_dist[k]
        if C > 0.0:
            a = alpha_dist[k] * 10.0  # 10x weaker for separation
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
def _xpbd_project_distance_jacobi_buffered(
    dist_correction: ti.template(),
    invm: ti.template(),
    valence_dist: ti.template(),
    pair_i: ti.template(),
    pair_j: ti.template(),
    rest_dist: ti.template(),
    lambda_dist: ti.template(),
    alpha_dist: ti.template(),
    positions: ti.template(),
    omega: ti.f64,
    n_pairs: ti.i32,
):
    """Like _xpbd_project_distance_jacobi but writes to a correction buffer."""
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

        # Dead zone: skip micro-corrections that cause jaggedness
        tolerance = 0.002  # 2mm
        if ti.abs(C) < tolerance:
            continue

        w_sum = wi + wj
        # Asymmetric stiffness: stiff compression, softer separation
        a = alpha_dist[k]
        if C > 0.0:
            a = alpha_dist[k] * 10.0  # 10x weaker for separation
        denom = w_sum + a
        if denom < 1e-30:
            continue

        dl = -(C + a * lambda_dist[k]) / denom

        si = omega / valence_dist[i]
        sj = omega / valence_dist[j]

        ti.atomic_add(dist_correction[i], si * wi * dl * n)
        ti.atomic_add(dist_correction[j], -sj * wj * dl * n)
        lambda_dist[k] += dl


@ti.kernel
def _apply_clamped_corrections(
    positions: ti.template(),
    dist_correction: ti.template(),
    invm: ti.template(),
    max_disp: ti.f64,
    n: ti.i32,
):
    """Apply buffered distance corrections with per-vertex displacement clamping."""
    for i in range(n):
        if invm[i] > 0.0:
            dx = dist_correction[i]
            d = dx.norm()
            if d > max_disp:
                dx = dx * (max_disp / d)
            positions[i] += dx
            dist_correction[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def _laplacian_smooth_surface(
    positions: ti.template(),
    smoothed: ti.template(),
    invm: ti.template(),
    surf_verts: ti.template(),
    adj_data: ti.template(),
    adj_offset: ti.template(),
    factor: ti.f64,
    n_surf: ti.i32,
):
    """Laplacian smooth for surface vertices only."""
    for s in range(n_surf):
        vi = surf_verts[s]
        if invm[vi] == 0.0:
            smoothed[vi] = positions[vi]
            continue
        start = adj_offset[s]
        end = adj_offset[s + 1]
        avg = ti.Vector([0.0, 0.0, 0.0])
        count = 0
        for k in range(start, end):
            avg += positions[adj_data[k]]
            count += 1
        if count > 0:
            avg /= float(count)
            smoothed[vi] = positions[vi] + factor * (avg - positions[vi])
        else:
            smoothed[vi] = positions[vi]


@ti.kernel
def _copy_smoothed_to_positions(
    positions: ti.template(),
    smoothed: ti.template(),
    surf_verts: ti.template(),
    invm: ti.template(),
    n_surf: ti.i32,
):
    """Copy smoothed values back to positions for surface vertices only."""
    for s in range(n_surf):
        vi = surf_verts[s]
        if invm[vi] > 0.0:
            positions[vi] = smoothed[vi]


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
        self._has_previous_solution = False
        # Inter-muscle distance constraints
        self._n_dist_constraints = 0
        self._dist_pair_i = None
        self._dist_pair_j = None
        self._dist_rest = None
        self._compliant_attachments = False
        self._attachment_indices = np.zeros(0, dtype=np.int32)
        self._attachment_targets = np.zeros((0, 3), dtype=np.float64)
        self._attachment_kappa = 0.0
        # Lazily populated per muscle pair. Raw source surfaces overlap in the
        # saved rest anatomy, so collision handling must reject only crossings
        # outside those rest interface patches.
        self._rest_crossing_masks = {}

    def build(self, muscles_data):
        _ensure_ti_init()

        all_verts = []
        all_tets = []
        all_fixed = []
        all_surface_verts = []
        all_surface_faces = []  # global-indexed surface faces per muscle
        vert_offset = 0
        tet_offset = 0

        for name, data in muscles_data.items():
            n_v = len(data['rest_positions'])
            n_t = len(data['tetrahedra'])

            all_verts.append(data['rest_positions'].astype(np.float64))
            all_tets.append(data['tetrahedra'].astype(np.int32) + vert_offset)
            all_fixed.append(data['fixed_mask'].astype(bool))

            if data.get('surface_faces') is not None:
                surf_faces = data['surface_faces'].astype(np.int32) + vert_offset
                surf_v = np.unique(surf_faces.ravel())
                all_surface_faces.append(surf_faces)
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

        # Tet graph coloring for Gauss-Seidel XPBD
        self._build_tet_coloring()

        # Build surface adjacency for Laplacian smoothing (per-muscle, no cross-muscle edges)
        self._build_surface_adjacency(all_surface_faces)

        # Store per-muscle surface faces for muscle-muscle collision
        self._muscle_surface_faces = {}
        sf_idx = 0
        for name, data in muscles_data.items():
            if data.get('surface_faces') is not None:
                self._muscle_surface_faces[name] = all_surface_faces[sf_idx]
                sf_idx += 1

        # Discover reciprocal rest-pose anatomical interfaces. Store only the
        # source patch membership; closest triangles are deliberately NOT
        # stored, because they must change as the two surfaces slide.
        self._sliding_interfaces = []
        if TRIMESH_AVAILABLE:
            from scipy.spatial import cKDTree
            directed = {}
            names = list(self._muscle_surface_faces)
            for ia, name_a in enumerate(names):
                vs_a, ve_a, _, _ = self._muscle_ranges[name_a]
                surf_a = np.unique(self._muscle_surface_faces[name_a])
                pa = self.rest_positions[surf_a]
                for name_b in names[ia + 1:]:
                    vs_b, ve_b, _, _ = self._muscle_ranges[name_b]
                    surf_b = np.unique(self._muscle_surface_faces[name_b])
                    pb = self.rest_positions[surf_b]
                    da, _ = cKDTree(pb).query(pa)
                    db, _ = cKDTree(pa).query(pb)
                    keep_a = surf_a[da < 0.006]
                    keep_b = surf_b[db < 0.006]
                    # Reciprocal substantial patches reject incidental point
                    # contacts while retaining broad fascial interfaces.
                    if len(keep_a) >= 12 and len(keep_b) >= 12:
                        directed[(name_a, name_b)] = keep_a
                        directed[(name_b, name_a)] = keep_b
            for (src, tgt), global_vertices in sorted(directed.items()):
                self._sliding_interfaces.append(
                    (src, tgt, np.asarray(global_vertices, dtype=np.int32)))
            print(f"  PN sliding interfaces: {len(self._sliding_interfaces)} "
                  "directed reciprocal patches")

        # Tag collision regions: vertices near attachments excluded from mm collision.
        # 0=attachment (fixed), 1=near-attachment (bone-only), 2=belly (full collision)
        self._build_collision_regions(all_surface_faces)

        # Allocate Taichi fields
        self._allocate_fields()
        self._allocate_surface_smooth_fields()
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
        # Use an element-local scale.  A single global average admitted very
        # thin contour-band slivers (for example a Rectus Femoris tet with
        # 1.4e-11 m^3 rest volume) that become singular under an otherwise
        # rigid attachment motion and block the entire unified solve.
        q = X[tets]
        edge_lengths = np.stack([
            np.linalg.norm(q[:, i] - q[:, j], axis=1)
            for i, j in ((0, 1), (0, 2), (0, 3),
                         (1, 2), (1, 3), (2, 3))], axis=1)
        local_scale = np.maximum(np.max(edge_lengths, axis=1) ** 3, 1e-30)
        good = (vol / local_scale) > 1e-5
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

    def _build_tet_coloring(self):
        """Color tets so no two tets in the same color share a vertex.

        Enables Gauss-Seidel XPBD: within a color, all tet corrections
        are independent, so parallel execution is exact (no atomic_add).
        """
        from collections import defaultdict

        M = self._n_tets
        tets = self.tetrahedra

        # Build tet adjacency: two tets are adjacent if they share any vertex
        vert_to_tets = defaultdict(list)
        for t in range(M):
            for j in range(4):
                vert_to_tets[tets[t, j]].append(t)

        tet_adj = defaultdict(set)
        for v_tets in vert_to_tets.values():
            for i in range(len(v_tets)):
                for j in range(i + 1, len(v_tets)):
                    tet_adj[v_tets[i]].add(v_tets[j])
                    tet_adj[v_tets[j]].add(v_tets[i])

        # Greedy coloring
        color = np.full(M, -1, dtype=np.int32)
        for t in range(M):
            used = set()
            for nb in tet_adj[t]:
                if color[nb] >= 0:
                    used.add(color[nb])
            c = 0
            while c in used:
                c += 1
            color[t] = c

        n_colors = int(color.max()) + 1
        self._tet_n_colors = n_colors
        self._tet_color_arrays = []
        for c in range(n_colors):
            tets_c = np.where(color == c)[0].astype(np.int32)
            self._tet_color_arrays.append(tets_c)

        print(f"  Tet graph coloring: {n_colors} colors")

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
        self.ti_target_J = ti.field(dtype=ti.f64, shape=M)  # per-tet volume target

        # Every tet preserves volume, including attachment-adjacent elements.
        # The previous 0.1/0.3/0.5 targets deliberately collapsed cap tets and
        # caused the visible attachment instability this solver is meant to
        # avoid. Hard boundary conditions and incompressibility must coexist;
        # an incompatible pose should report residual, not delete volume.
        target_J_np = np.ones(M, dtype=np.float64)
        self.ti_target_J.from_numpy(target_J_np)
        print("  Volume target: J=1 for all tets (including attachments)")

        # Per-color tet arrays for Gauss-Seidel
        max_color_tets = max(len(ca) for ca in self._tet_color_arrays)
        self.ti_color_tets = ti.field(dtype=ti.i32, shape=max_color_tets)
        self._max_color_tets = max_color_tets

        # Buffered distance corrections for clamping
        self.ti_dist_correction = ti.Vector.field(3, dtype=ti.f64, shape=N)

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

    def _build_surface_adjacency(self, all_surface_faces):
        """Build CSR adjacency for surface vertices (per-muscle, no cross-muscle edges)."""
        from collections import defaultdict

        if not all_surface_faces:
            self._n_surf_verts = 0
            self._surf_verts_np = np.array([], dtype=np.int32)
            self._surf_adj_data_np = np.array([], dtype=np.int32)
            self._surf_adj_offset_np = np.array([0], dtype=np.int32)
            return

        # Build adjacency per vertex from surface faces
        adj = defaultdict(set)
        for faces in all_surface_faces:
            for f in faces:
                for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
                    adj[a].add(b)
                    adj[b].add(a)

        # Only keep free surface vertices
        free_set = set(self.free_indices.tolist())
        surf_verts = sorted([v for v in adj.keys() if v in free_set])

        # Build CSR
        adj_data = []
        adj_offset = [0]
        for v in surf_verts:
            neighbors = sorted(adj[v])
            adj_data.extend(neighbors)
            adj_offset.append(len(adj_data))

        self._n_surf_verts = len(surf_verts)
        self._surf_verts_np = np.array(surf_verts, dtype=np.int32)
        self._surf_adj_data_np = np.array(adj_data, dtype=np.int32)
        self._surf_adj_offset_np = np.array(adj_offset, dtype=np.int32)
        print(f"  Surface adjacency: {self._n_surf_verts} free surface verts, "
              f"{len(adj_data)} adjacency entries")

    def _build_collision_regions(self, all_surface_faces):
        """Tag vertices by collision region using BFS from fixed vertices.

        Region 0: fixed (attachment) — no collision
        Region 1: within 2 edge-rings of fixed — bone-only collision
        Region 2: belly — full muscle-muscle + bone collision
        """
        from collections import defaultdict, deque

        self._collision_region = np.full(self._n_verts, 2, dtype=np.int32)
        self._collision_region[self.fixed_indices] = 0

        # Build adjacency from tet connectivity (not just surface)
        adj = defaultdict(set)
        for t in range(self._n_tets):
            verts = self.tetrahedra[t]
            for a in range(4):
                for b in range(a + 1, 4):
                    adj[verts[a]].add(verts[b])
                    adj[verts[b]].add(verts[a])

        # BFS 2 rings from fixed vertices
        q = deque()
        for fi in self.fixed_indices:
            q.append((fi, 0))
        visited = set(self.fixed_indices.tolist())

        while q:
            v, depth = q.popleft()
            if depth >= 2:
                continue
            for nb in adj[v]:
                if nb not in visited:
                    visited.add(nb)
                    self._collision_region[nb] = 1
                    q.append((nb, depth + 1))

        n_r0 = int(np.sum(self._collision_region == 0))
        n_r1 = int(np.sum(self._collision_region == 1))
        n_r2 = int(np.sum(self._collision_region == 2))
        print(f"  Collision regions: {n_r0} fixed, {n_r1} near-attach, {n_r2} belly")

    def _allocate_surface_smooth_fields(self):
        """Allocate Taichi fields for surface Laplacian smoothing."""
        if self._n_surf_verts == 0:
            return
        N = self._n_verts
        S = self._n_surf_verts
        A = len(self._surf_adj_data_np)

        self.ti_smoothed = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_surf_verts = ti.field(dtype=ti.i32, shape=S)
        self.ti_surf_adj_data = ti.field(dtype=ti.i32, shape=max(A, 1))
        self.ti_surf_adj_offset = ti.field(dtype=ti.i32, shape=S + 1)

        self.ti_surf_verts.from_numpy(self._surf_verts_np)
        if A > 0:
            self.ti_surf_adj_data.from_numpy(self._surf_adj_data_np)
        self.ti_surf_adj_offset.from_numpy(self._surf_adj_offset_np)

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
        """Update targets and positions from skeleton.

        First call: uses LBS positions as initial guess.
        Subsequent calls: keeps previous solution, only updates fixed targets.
        """
        for name, data in muscles_data.items():
            vs, ve, _, _ = self._muscle_ranges[name]
            if not self._has_previous_solution:
                if 'positions' in data:
                    self.positions[vs:ve] = data['positions']
            if 'fixed_targets' in data:
                fi = np.where(self._orig_fixed_mask[vs:ve])[0]
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

        # Phase 2: Per-bone contains() and per-containing-bone projection.
        # Never detect on individual bones and then project on the merged mesh:
        # its nearest triangle can belong to a different adjacent bone.
        individual = getattr(self, '_individual_bone_meshes', [])
        if individual:
            target_by_global = {}
            for bone_mesh in individual:
                try:
                    # Bounding box pre-filter (fast reject)
                    bmin = bone_mesh.bounds[0] - proximity_radius
                    bmax = bone_mesh.bounds[1] + proximity_radius
                    in_bbox = np.all((nearby_pos >= bmin) & (nearby_pos <= bmax), axis=1)
                    if not np.any(in_bbox):
                        continue
                    # contains() uses ray-casting — reliable for watertight meshes
                    bbox_indices = np.where(in_bbox)[0]
                    contained = bone_mesh.contains(nearby_pos[bbox_indices])
                    local_indices = bbox_indices[contained]
                    if not len(local_indices):
                        continue
                    if verbose:
                        label = bone_mesh.metadata.get(
                            'body_name', bone_mesh.metadata.get('bone_name', '?'))
                        print(f"      inside {label}: {len(local_indices)}")
                    closest, distances, face_ids = (
                        bone_mesh.nearest.on_surface(nearby_pos[local_indices]))
                    escape = closest - nearby_pos[local_indices]
                    escape /= np.maximum(
                        np.linalg.norm(escape, axis=1, keepdims=True), 1e-12)
                    # closest - contained_point is guaranteed to point toward
                    # the exterior; imported face winding is not guaranteed.
                    targets = closest + escape * margin
                    for local_i, target, distance in zip(
                            local_indices, targets, distances):
                        global_i = int(nearby_free_surf[local_i])
                        old = target_by_global.get(global_i)
                        # A point inside overlapping bone volumes must escape
                        # the enclosing volume, not merely the closest nested
                        # surface. Prefer the deepest containing-bone escape.
                        if old is None or float(distance) > old[0]:
                            target_by_global[global_i] = (
                                float(distance), np.asarray(target))
                except Exception:
                    continue
            if verbose:
                print(f"    Collision: {len(free_surf)} surf verts, "
                      f"{n_nearby} near bones, {len(target_by_global)} inside "
                      f"({len(individual)} watertight bones)")
            if not target_by_global:
                return np.array([], dtype=np.int32), np.zeros((0, 3))
            ordered = sorted(target_by_global)
            return (np.asarray(ordered, dtype=np.int32),
                    np.stack([target_by_global[i][1] for i in ordered]))

        # Fallback: dot-product test on merged mesh for non-watertight bones
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
                  f"{n_inside} inside (merged non-watertight fallback)")

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

    def _compute_muscle_collision_targets(self, margin=0.001, verbose=False):
        """Detect muscle-muscle surface penetrations and compute projection targets.

        For each neighboring muscle pair, check if surface vertices of one
        muscle penetrate the other. Project penetrating vertices to the
        other muscle's surface + margin.

        Returns (global_vert_indices, target_positions) as numpy arrays.
        """
        if not TRIMESH_AVAILABLE or not hasattr(self, '_muscle_surface_faces'):
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        if not self._muscle_surface_faces:
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        from scipy.spatial import cKDTree

        # Determine neighbor pairs from distance constraints
        if not hasattr(self, '_muscle_neighbor_pairs'):
            neighbor_set = set()
            names = list(self._muscle_ranges.keys())
            # Build muscle index lookup: global vertex -> muscle name
            vert_to_muscle = {}
            for name, (vs, ve, _, _) in self._muscle_ranges.items():
                for v in range(vs, ve):
                    vert_to_muscle[v] = name
            # From distance constraint pairs, find which muscles are connected
            dist_pair_i = getattr(self, '_dist_pair_i', None)
            dist_pair_j = getattr(self, '_dist_pair_j', None)
            n_dist = int(getattr(self, '_n_dist_constraints', 0))
            if dist_pair_i is not None and dist_pair_j is not None:
                for k in range(n_dist):
                    m1 = vert_to_muscle.get(dist_pair_i[k])
                    m2 = vert_to_muscle.get(dist_pair_j[k])
                    if m1 and m2 and m1 != m2:
                        pair = tuple(sorted([m1, m2]))
                        neighbor_set.add(pair)
            # Sliding contact must not depend on legacy spring/bond
            # constraints. Anatomical-contact baking deliberately disables
            # those pins, so consider every muscle pair and let the cheap AABB
            # test below reject spatially separated pairs. For a 24-muscle
            # upper leg this is only 276 conservative candidate pairs.
            for ia, name_a in enumerate(names):
                for name_b in names[ia + 1:]:
                    neighbor_set.add(tuple(sorted((name_a, name_b))))
            self._muscle_neighbor_pairs = sorted(neighbor_set)
            if verbose:
                print(f"    Muscle neighbor pairs: {len(self._muscle_neighbor_pairs)}")

        all_idx = []
        all_tgt = []
        self._last_triangle_crossings = 0
        self._last_triangle_crossings_raw = 0

        for name_a, name_b in self._muscle_neighbor_pairs:
            if name_a not in self._muscle_surface_faces or name_b not in self._muscle_surface_faces:
                continue

            # Get ranges
            vs_a, ve_a, _, _ = self._muscle_ranges[name_a]
            vs_b, ve_b, _, _ = self._muscle_ranges[name_b]

            pos_a = self.positions[vs_a:ve_a]
            pos_b = self.positions[vs_b:ve_b]

            # AABB overlap check (fast reject)
            min_a, max_a = pos_a.min(axis=0), pos_a.max(axis=0)
            min_b, max_b = pos_b.min(axis=0), pos_b.max(axis=0)
            if np.any(min_a > max_b + margin) or np.any(min_b > max_a + margin):
                continue

            # Build trimeshes with current positions
            faces_a = self._muscle_surface_faces[name_a] - vs_a
            faces_b = self._muscle_surface_faces[name_b] - vs_b

            try:
                mesh_a = trimesh.Trimesh(vertices=pos_a, faces=faces_a, process=False)
                mesh_b = trimesh.Trimesh(vertices=pos_b, faces=faces_b, process=False)
                # Collision classification and escape directions require
                # consistently outward normals. ``process=False`` preserves
                # vertex indexing but does not orient the face winding.
                mesh_a.fix_normals(multibody=True)
                mesh_b.fix_normals(multibody=True)
            except Exception:
                continue

            # Check A's surface verts inside B, and B's surface verts inside A.
            # Uses signed-distance (nearest-point + normal dot product).
            for src_name, src_mesh, tgt_mesh, src_vs in [
                (name_a, mesh_a, mesh_b, vs_a),
                (name_b, mesh_b, mesh_a, vs_b),
            ]:
                src_surf_local = np.unique(self._muscle_surface_faces[src_name] - src_vs)
                src_surf_pos = src_mesh.vertices[src_surf_local]

                # KD-tree pre-filter: only check vertices near the target mesh
                tgt_tree = cKDTree(tgt_mesh.vertices)
                dists_kd, _ = tgt_tree.query(src_surf_pos)
                nearby = dists_kd < margin * 5
                if not np.any(nearby):
                    continue

                nearby_pos = src_surf_pos[nearby]
                nearby_local = src_surf_local[nearby]

                # Signed-distance: nearest surface point + normal dot product
                try:
                    closest_pts, dists_surf, face_ids = tgt_mesh.nearest.on_surface(nearby_pos)
                    normals = tgt_mesh.face_normals[face_ids]
                except Exception:
                    continue

                # Inside = vertex is on the inward side of the nearest,
                # consistently oriented face.  Restrict this to the narrow
                # contact band; full ray-cast containment is prohibitively
                # expensive in the repeated active-set loop.
                to_vertex = nearby_pos - closest_pts
                dots = np.einsum('ij,ij->i', to_vertex, normals)
                inside = ((dots < 0) & (dists_surf < margin * 3))

                if not np.any(inside):
                    continue

                targets = closest_pts[inside] + normals[inside] * margin
                global_idx = (nearby_local[inside] + src_vs).astype(np.int32)

                # Only project belly vertices (region 2) — skip fixed + near-attachment
                coll_region = getattr(self, '_collision_region', None)
                if coll_region is not None:
                    valid_mask = coll_region[global_idx] == 2
                else:
                    valid_mask = ~self.fixed_mask[global_idx]
                if np.any(valid_mask):
                    all_idx.append(global_idx[valid_mask])
                    all_tgt.append(targets[valid_mask])

            # Exact narrow phase for face crossings. This catches edge-face
            # intersections even when every vertex is outside the other mesh.
            try:
                crossing_pairs = _triangle_intersection_pairs(mesh_a, mesh_b)
            except Exception:
                crossing_pairs = np.zeros((0, 2), dtype=np.int32)
            self._last_triangle_crossings_raw += len(crossing_pairs)

            # The input rest surfaces themselves contain intentional/baked-in
            # overlaps. Permit crossings inside a two-face-ring rest interface
            # patch (so it can slide), but reject crossings that migrate beyond
            # that patch or appear between previously disjoint surfaces.
            if not hasattr(self, '_rest_crossing_masks'):
                self._rest_crossing_masks = {}
            pair_key = (name_a, name_b)
            if pair_key not in self._rest_crossing_masks:
                rest_a = self.rest_positions[vs_a:ve_a]
                rest_b = self.rest_positions[vs_b:ve_b]
                try:
                    rest_mesh_a = trimesh.Trimesh(
                        vertices=rest_a, faces=faces_a, process=False)
                    rest_mesh_b = trimesh.Trimesh(
                        vertices=rest_b, faces=faces_b, process=False)
                    allow_a, allow_b, rest_count = \
                        _dilated_crossing_face_masks(rest_mesh_a, rest_mesh_b)
                except Exception:
                    allow_a = np.zeros(len(faces_a), dtype=bool)
                    allow_b = np.zeros(len(faces_b), dtype=bool)
                    rest_count = 0
                self._rest_crossing_masks[pair_key] = (
                    allow_a, allow_b, rest_count)
            allow_a, allow_b, _ = self._rest_crossing_masks[pair_key]
            if len(crossing_pairs):
                allowed = (allow_a[crossing_pairs[:, 0]]
                           & allow_b[crossing_pairs[:, 1]])
                crossing_pairs = crossing_pairs[~allowed]
            self._last_triangle_crossings += len(crossing_pairs)
            if len(crossing_pairs):
                rest_a = self.rest_positions[vs_a:ve_a]
                rest_b = self.rest_positions[vs_b:ve_b]
                for face_a, face_b in crossing_pairs:
                    local_a = faces_a[int(face_a)]
                    local_b = faces_b[int(face_b)]
                    # Preserve the local side ordering of the supplied anatomy.
                    # Whole-muscle centroids give the wrong direction for long,
                    # wrapping muscles such as sartorius and gracilis.
                    tri_axis = (rest_a[local_a].mean(axis=0)
                                - rest_b[local_b].mean(axis=0))
                    if np.linalg.norm(tri_axis) < 1e-8:
                        tri_axis = (pos_a[local_a].mean(axis=0)
                                    - pos_b[local_b].mean(axis=0))
                    tri_axis /= max(float(np.linalg.norm(tri_axis)), 1e-12)
                    for local_ids, offset, direction in (
                            (local_a, vs_a, tri_axis),
                            (local_b, vs_b, -tri_axis)):
                        global_ids = local_ids.astype(np.int32) + offset
                        valid = (~self.fixed_mask[global_ids])
                        coll_region = getattr(self, '_collision_region', None)
                        if coll_region is not None:
                            valid &= coll_region[global_ids] == 2
                        if np.any(valid):
                            ids = global_ids[valid]
                            all_idx.append(ids)
                            all_tgt.append(
                                self.positions[ids] + direction * margin)

        if not all_idx:
            return np.array([], dtype=np.int32), np.zeros((0, 3))

        return np.concatenate(all_idx), np.concatenate(all_tgt)

    def solve(self, mu, lam, vol_penalty=100.0, kappa=1e4, margin=0.002,
              max_lbfgs_iters=100, n_outer=3, bone_meshes=None, verbose=False,
              n_load_steps=0, bone_mesh_callback=None):
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

        # Warm-start: keep Lagrange multipliers from previous frame for faster
        # convergence. Reset only on first frame or when solver is rebuilt.
        if not self._has_previous_solution:
            self.ti_lambda_H.fill(0.0)
            self.ti_lambda_D.fill(0.0)

        # Inter-muscle distance constraint setup
        K = self._n_dist_constraints
        if K > 0:
            dist_stiffness = effective_lam
            alpha_dist_np = 1.0 / (dist_stiffness * self._dist_rest + 1e-30)
            self.ti_dist_alpha.from_numpy(alpha_dist_np)
            if not self._has_previous_solution:
                self.ti_dist_lambda.fill(0.0)

        # Save positions before solve for total convergence metric
        initial_pos = self.ti_positions.to_numpy().copy()

        omega = 1.7

        # Pre-upload per-color tet arrays (padded to max size)
        color_tet_arrays = []
        for c in range(self._tet_n_colors):
            ca = self._tet_color_arrays[c]
            padded = np.zeros(self._max_color_tets, dtype=np.int32)
            padded[:len(ca)] = ca
            color_tet_arrays.append((padded, len(ca)))

        # Helper: run n XPBD iterations using Gauss-Seidel coloring
        def _run_xpbd_iters(n, verbose_iters=False):
            for it in range(n):
                if verbose_iters and it < 3:
                    _copy_positions(self.ti_pos_prev, self.ti_positions, N)

                # Gauss-Seidel: process one color at a time
                for c in range(self._tet_n_colors):
                    padded, n_ct = color_tet_arrays[c]
                    if n_ct == 0:
                        continue
                    self.ti_color_tets.from_numpy(padded)
                    _xpbd_project_gauss_seidel(
                        self.ti_positions, self.ti_invm,
                        self.ti_color_tets, self.ti_tets,
                        self.ti_Bm_inv, self.ti_rest_volume,
                        self.ti_lambda_H, self.ti_lambda_D,
                        self.ti_alpha_H, self.ti_alpha_D,
                        self.ti_target_J,
                        n_ct,
                    )

                if K > 0:
                    self.ti_dist_correction.fill(0.0)
                    _xpbd_project_distance_jacobi_buffered(
                        self.ti_dist_correction, self.ti_invm, self.ti_dist_valence,
                        self.ti_dist_pair_i, self.ti_dist_pair_j,
                        self.ti_dist_rest, self.ti_dist_lambda, self.ti_dist_alpha,
                        self.ti_positions, omega, K,
                    )
                    _apply_clamped_corrections(
                        self.ti_positions, self.ti_dist_correction, self.ti_invm,
                        0.001, N,
                    )
                _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)

                if verbose_iters and it < 3:
                    delta = _compute_position_delta(
                        self.ti_positions, self.ti_pos_prev, self.ti_invm, N)
                    print(f"      iter {it}: ||dx||={delta**0.5:.6e}")

        # Helper: detect and project collisions, return merged targets
        def _detect_and_project_collisions(verbose_coll=False):
            all_ci = []
            all_ct = []
            if hasattr(self, '_muscle_surface_faces') and self._muscle_surface_faces:
                self.positions = self.ti_positions.to_numpy()
                mi, mt = self._compute_muscle_collision_targets(
                    margin=margin, verbose=verbose_coll)
                if len(mi) > 0:
                    all_ci.append(mi); all_ct.append(mt)
            if self._merged_bone_mesh is not None:
                self.positions = self.ti_positions.to_numpy()
                bi, bt = self._compute_collision_targets(
                    margin, verbose=verbose_coll)
                if len(bi) > 0:
                    all_ci.append(bi); all_ct.append(bt)
            if not all_ci:
                return np.array([], dtype=np.int32), np.zeros((0, 3))
            return np.concatenate(all_ci), np.concatenate(all_ct)

        def _upload_and_project(idx, tgt, stiffness=1.0):
            nc = len(idx)
            if nc == 0:
                return
            if nc > self._max_coll:
                self._max_coll = max(nc, 1000)
                self.ti_coll_idx = ti.field(dtype=ti.i32, shape=self._max_coll)
                self.ti_coll_targets = ti.Vector.field(3, dtype=ti.f64, shape=self._max_coll)
            ip = np.zeros(self._max_coll, dtype=np.int32)
            tp = np.zeros((self._max_coll, 3), dtype=np.float64)
            ip[:nc] = idx; tp[:nc] = tgt
            self.ti_coll_idx.from_numpy(ip)
            self.ti_coll_targets.from_numpy(tp)
            _apply_collision_projection(
                self.ti_positions, self.ti_invm,
                self.ti_coll_idx, self.ti_coll_targets, stiffness, nc)
            return nc

        # First frame: Jacobi pre-solve to smooth LBS inversions, then GS.
        # Jacobi handles far-from-equilibrium better (no GS oscillation).
        if not self._has_previous_solution:
            self.ti_lambda_H.fill(0.0)
            self.ti_lambda_D.fill(0.0)
            for pre_it in range(5):
                _xpbd_project_jacobi(
                    self.ti_positions, self.ti_invm, self.ti_valence,
                    self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
                    self.ti_lambda_H, self.ti_lambda_D,
                    self.ti_alpha_H, self.ti_alpha_D,
                    omega, M,
                )
                _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)
            self.ti_lambda_H.fill(0.0)
            self.ti_lambda_D.fill(0.0)

        # GS elastic iterations
        _run_xpbd_iters(n_iters, verbose_iters=verbose)

        # Taubin smoothing: smooth surface jaggedness without shrinkage
        if self._n_surf_verts > 0:
            S = self._n_surf_verts
            lam_smooth = 0.3
            mu_smooth = -0.31
            for _ in range(2):
                _laplacian_smooth_surface(
                    self.ti_positions, self.ti_smoothed, self.ti_invm,
                    self.ti_surf_verts, self.ti_surf_adj_data,
                    self.ti_surf_adj_offset, lam_smooth, S)
                _copy_smoothed_to_positions(
                    self.ti_positions, self.ti_smoothed,
                    self.ti_surf_verts, self.ti_invm, S)
                _laplacian_smooth_surface(
                    self.ti_positions, self.ti_smoothed, self.ti_invm,
                    self.ti_surf_verts, self.ti_surf_adj_data,
                    self.ti_surf_adj_offset, mu_smooth, S)
                _copy_smoothed_to_positions(
                    self.ti_positions, self.ti_smoothed,
                    self.ti_surf_verts, self.ti_invm, S)
            _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)

        # Final collision: detect and project + relax
        final_idx, final_tgt = _detect_and_project_collisions(verbose_coll=verbose)
        n_coll_total = len(final_idx)
        if n_coll_total > 0:
            _upload_and_project(final_idx, final_tgt, stiffness=1.0)
            for _ in range(15):
                _run_xpbd_iters(1)
                _upload_and_project(final_idx, final_tgt, stiffness=0.5)
            if verbose:
                print(f"    final collision: {n_coll_total} vertices projected + 15 relax")

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
                  f"inv={n_inv}/{M} dist={K} "
                  f"J=[{np.min(Js):.3f},{np.max(Js):.3f}]")
            # Per-muscle breakdown (top offenders)
            muscle_inv = []
            for mname, (vs, ve, ts, te) in self._muscle_ranges.items():
                Jm = Js[ts:te]
                ni = int(np.sum(Jm <= 0))
                if ni > 0:
                    muscle_inv.append((ni, te - ts, mname, float(Jm.min())))
            muscle_inv.sort(reverse=True)
            for ni, nt, mname, jmin in muscle_inv[:10]:
                print(f"      {mname}: {ni}/{nt} ({100*ni/nt:.0f}%) Jmin={jmin:.0f}")

        self._has_previous_solution = True
        return n_iters, residual

    def get_muscle_positions(self, name):
        vs, ve, _, _ = self._muscle_ranges[name]
        return self.positions[vs:ve].copy()


# ---------------------------------------------------------------------------
# VBD Solver (Vertex Block Descent, SIGGRAPH 2024)
# ---------------------------------------------------------------------------

@ti.kernel
def _vbd_solve_color(
    positions: ti.template(),
    x_tilde: ti.template(),
    Bm_inv: ti.template(),
    rest_volume: ti.template(),
    vert_tet_data: ti.template(),
    vert_tet_offset: ti.template(),
    vert_local_idx: ti.template(),
    tets: ti.template(),
    invm: ti.template(),
    color_verts: ti.template(),
    # Distance constraint CSR for per-vertex accumulation
    dist_pair_other: ti.template(),     # the OTHER vertex in each constraint
    dist_pair_rest: ti.template(),      # rest distance
    dist_vert_offset: ti.template(),    # CSR offsets per vertex
    dist_stiffness: ti.f64,
    n_dist: ti.i32,
    # Material parameters
    mu: ti.f64,
    lam: ti.f64,
    eps: ti.f64,
    mass_over_h2: ti.f64,
    n_color_verts: ti.i32,
):
    """VBD per-vertex Newton step for one graph color.

    For each vertex in the color (in parallel), accumulate gradient g (3-vec)
    and Hessian H (3x3) from incident tets using Stable Neo-Hookean energy
    plus inertia term m/h² * ||x - x̃||², then solve x_i -= H^{-1} g.
    """
    for ci in range(n_color_verts):
        vi = color_verts[ci]
        if invm[vi] == 0.0:
            continue

        # Inertia term: E_inertia = m/(2h²) * ||x - x̃||²
        # g_inertia = m/h² * (x - x̃),  H_inertia = m/h² * I
        m_h2 = mass_over_h2 / (invm[vi] + 1e-30)  # m/h² where m = 1/invm
        g = m_h2 * (positions[vi] - x_tilde[vi])
        H = m_h2 * ti.Matrix.identity(ti.f64, 3)

        start = vert_tet_offset[vi]
        end = vert_tet_offset[vi + 1]

        for k in range(start, end):
            tet_idx = vert_tet_data[k]
            local_idx = vert_local_idx[k]

            i0 = tets[tet_idx][0]
            i1 = tets[tet_idx][1]
            i2 = tets[tet_idx][2]
            i3 = tets[tet_idx][3]

            x0 = positions[i0]
            x1 = positions[i1]
            x2 = positions[i2]
            x3 = positions[i3]

            # Deformation gradient: Ds = [x0-x3, x1-x3, x2-x3], F = Ds @ Bm_inv
            Ds = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])
            B = Bm_inv[tet_idx]
            F = Ds @ B
            V0 = rest_volume[tet_idx]

            # First Piola-Kirchhoff stress (Stable Neo-Hookean, Smith 2018):
            # Ψ = μ/2(||F||² - 3) + λ/2(J - 1)²
            # P = μF + λ(J - 1)·cof(F)
            f0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]], dt=ti.f64)
            f1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]], dt=ti.f64)
            f2 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]], dt=ti.f64)

            J = F.determinant()
            cof = ti.Matrix.cols([f1.cross(f2), f2.cross(f0), f0.cross(f1)])

            # Get the "b" vector (Bm_inv contribution for this vertex)
            b = ti.Vector([0.0, 0.0, 0.0])
            if local_idx == 0:
                b = ti.Vector([B[0, 0], B[0, 1], B[0, 2]], dt=ti.f64)
            elif local_idx == 1:
                b = ti.Vector([B[1, 0], B[1, 1], B[1, 2]], dt=ti.f64)
            elif local_idx == 2:
                b = ti.Vector([B[2, 0], B[2, 1], B[2, 2]], dt=ti.f64)
            else:  # local_idx == 3
                b = -(ti.Vector([B[0, 0], B[0, 1], B[0, 2]], dt=ti.f64)
                      + ti.Vector([B[1, 0], B[1, 1], B[1, 2]], dt=ti.f64)
                      + ti.Vector([B[2, 0], B[2, 1], B[2, 2]], dt=ti.f64))

            bb = b.dot(b)

            gi = ti.Vector([0.0, 0.0, 0.0])
            Hi = ti.Matrix.zero(ti.f64, 3, 3)

            # Blend between Neo-Hookean (well-conditioned tets) and ARAP fallback
            # (distorted tets). ARAP uses full stiffness mu+lam.
            # Smooth blending avoids oscillation at the threshold.
            w_nh = 1.0  # 1.0 = full Neo-Hookean, 0.0 = full ARAP
            if J < 0.5:
                w_nh = ti.max(0.0, (J - 0.1) / 0.4)  # ramp 0.1→0.5
            elif J > 3.0:
                w_nh = ti.max(0.0, (6.0 - J) / 3.0)  # ramp 3.0→6.0
            if F.norm_sqr() > 30.0:
                w_f = ti.max(0.0, (50.0 - F.norm_sqr()) / 20.0)
                w_nh = ti.min(w_nh, w_f)

            # ARAP contribution (always computed for blending)
            k_arap = mu + lam
            P_arap = k_arap * (F - ti.Matrix.identity(ti.f64, 3))
            PBt_arap = P_arap @ B.transpose()
            gi_arap = ti.Vector([0.0, 0.0, 0.0])
            if local_idx == 0:
                gi_arap = ti.Vector([PBt_arap[0, 0], PBt_arap[1, 0], PBt_arap[2, 0]], dt=ti.f64)
            elif local_idx == 1:
                gi_arap = ti.Vector([PBt_arap[0, 1], PBt_arap[1, 1], PBt_arap[2, 1]], dt=ti.f64)
            elif local_idx == 2:
                gi_arap = ti.Vector([PBt_arap[0, 2], PBt_arap[1, 2], PBt_arap[2, 2]], dt=ti.f64)
            else:
                gi_arap = -(ti.Vector([PBt_arap[0, 0], PBt_arap[1, 0], PBt_arap[2, 0]], dt=ti.f64)
                            + ti.Vector([PBt_arap[0, 1], PBt_arap[1, 1], PBt_arap[2, 1]], dt=ti.f64)
                            + ti.Vector([PBt_arap[0, 2], PBt_arap[1, 2], PBt_arap[2, 2]], dt=ti.f64))
            Hi_arap = k_arap * bb * ti.Matrix.identity(ti.f64, 3)

            if w_nh < 1e-6:
                # Fully ARAP
                gi = gi_arap
                Hi = Hi_arap
            else:
                # Neo-Hookean gradient
                P = mu * F + lam * (J - 1.0) * cof
                PBt = P @ B.transpose()
                gi_nh = ti.Vector([0.0, 0.0, 0.0])
                if local_idx == 0:
                    gi_nh = ti.Vector([PBt[0, 0], PBt[1, 0], PBt[2, 0]], dt=ti.f64)
                elif local_idx == 1:
                    gi_nh = ti.Vector([PBt[0, 1], PBt[1, 1], PBt[2, 1]], dt=ti.f64)
                elif local_idx == 2:
                    gi_nh = ti.Vector([PBt[0, 2], PBt[1, 2], PBt[2, 2]], dt=ti.f64)
                else:
                    gi_nh = -(ti.Vector([PBt[0, 0], PBt[1, 0], PBt[2, 0]], dt=ti.f64)
                              + ti.Vector([PBt[0, 1], PBt[1, 1], PBt[2, 1]], dt=ti.f64)
                              + ti.Vector([PBt[0, 2], PBt[1, 2], PBt[2, 2]], dt=ti.f64))

                # Neo-Hookean Hessian: mu*(b·b)*I + lam*(cof_col ⊗ cof_col)
                Hi_nh = mu * bb * ti.Matrix.identity(ti.f64, 3)
                cof_Bt = cof @ B.transpose()
                ci_vec = ti.Vector([0.0, 0.0, 0.0])
                if local_idx == 0:
                    ci_vec = ti.Vector([cof_Bt[0, 0], cof_Bt[1, 0], cof_Bt[2, 0]], dt=ti.f64)
                elif local_idx == 1:
                    ci_vec = ti.Vector([cof_Bt[0, 1], cof_Bt[1, 1], cof_Bt[2, 1]], dt=ti.f64)
                elif local_idx == 2:
                    ci_vec = ti.Vector([cof_Bt[0, 2], cof_Bt[1, 2], cof_Bt[2, 2]], dt=ti.f64)
                else:
                    ci_vec = -(ti.Vector([cof_Bt[0, 0], cof_Bt[1, 0], cof_Bt[2, 0]], dt=ti.f64)
                               + ti.Vector([cof_Bt[0, 1], cof_Bt[1, 1], cof_Bt[2, 1]], dt=ti.f64)
                               + ti.Vector([cof_Bt[0, 2], cof_Bt[1, 2], cof_Bt[2, 2]], dt=ti.f64))
                Hi_nh += lam * ci_vec.outer_product(ci_vec)

                # Blend Neo-Hookean and ARAP
                gi = w_nh * gi_nh + (1.0 - w_nh) * gi_arap
                Hi = w_nh * Hi_nh + (1.0 - w_nh) * Hi_arap

            g += V0 * gi
            H += V0 * Hi

        # Accumulate distance constraint gradient/Hessian for this vertex
        if n_dist > 0:
            d_start = dist_vert_offset[vi]
            d_end = dist_vert_offset[vi + 1]
            for dk in range(d_start, d_end):
                vj = dist_pair_other[dk]
                d0 = dist_pair_rest[dk]

                diff = positions[vi] - positions[vj]
                d = diff.norm()
                if d < 1e-12:
                    continue

                C = d - d0
                # Dead zone
                if ti.abs(C) < 0.002:
                    continue

                # Asymmetric stiffness
                kk = dist_stiffness
                if C > 0.0:
                    kk = dist_stiffness * 0.1

                n_dir = diff / d
                g += kk * C * n_dir

                # Hessian
                nn = n_dir.outer_product(n_dir)
                ratio = d0 / d
                if ratio > 1.0:
                    ratio = 1.0
                H += kk * nn + kk * (1.0 - ratio) * (ti.Matrix.identity(ti.f64, 3) - nn)

        # Regularization: ensure H is positive-definite
        # Use diagonal-proportional regularization for better conditioning
        diag_avg = (H[0, 0] + H[1, 1] + H[2, 2]) / 3.0
        reg = eps
        if diag_avg > eps:
            reg = diag_avg * 1e-3  # 0.1% of diagonal average
        H += reg * ti.Matrix.identity(ti.f64, 3)

        # Solve 3x3: dx = -H^{-1} g  (analytical inverse)
        det = H.determinant()
        if ti.abs(det) > 1e-30:
            H_inv = H.inverse()
            dx = -H_inv @ g
            # Clamp step size to prevent divergence on ill-conditioned tets
            dx_norm = dx.norm()
            max_step = 0.001  # 1mm max displacement per iteration
            if dx_norm > max_step:
                dx = dx * (max_step / dx_norm)
            positions[vi] += dx


@ti.kernel
def _vbd_project_distance(
    positions: ti.template(),
    invm: ti.template(),
    pair_i: ti.template(),
    pair_j: ti.template(),
    rest_dist: ti.template(),
    dist_stiffness: ti.f64,
    eps: ti.f64,
    n_pairs: ti.i32,
):
    """Distance constraint energy gradient+Hessian added as per-vertex Newton step.

    For each distance constraint pair, computes:
      E = k/2 * (||xi - xj|| - d0)^2
      g_i = k * (d - d0) * n
      H_i = k * (n⊗n) + k*(1 - d0/d)*(I - n⊗n)  [when d > d0*0.01]
    Then applies Newton step to both endpoints.
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

        d0 = rest_dist[k]
        C = d - d0

        # Dead zone: skip micro-corrections
        if ti.abs(C) < 0.002:
            continue

        # Asymmetric stiffness: stiff compression, softer separation
        kk = dist_stiffness
        if C > 0.0:
            kk = dist_stiffness * 0.1  # 10x weaker for separation

        n_dir = diff / d
        g_val = kk * C * n_dir

        # Hessian: k*(n⊗n) + k*(1 - d0/d)*(I - n⊗n)
        nn = n_dir.outer_product(n_dir)
        ratio = d0 / d
        if ratio > 1.0:
            ratio = 1.0  # clamp for stability when compressed
        H_val = kk * nn + kk * (1.0 - ratio) * (ti.Matrix.identity(ti.f64, 3) - nn)
        H_val += eps * ti.Matrix.identity(ti.f64, 3)

        det_H = H_val.determinant()
        if ti.abs(det_H) < 1e-30:
            continue
        H_inv = H_val.inverse()

        # Apply Newton step to both vertices (weighted by inverse mass)
        if wi > 0.0:
            dx_i = -H_inv @ g_val
            # Clamp displacement to 1mm per step
            dx_norm = dx_i.norm()
            if dx_norm > 0.001:
                dx_i = dx_i * (0.001 / dx_norm)
            positions[i] += dx_i

        if wj > 0.0:
            dx_j = H_inv @ g_val  # opposite gradient for j
            dx_norm = dx_j.norm()
            if dx_norm > 0.001:
                dx_j = dx_j * (0.001 / dx_norm)
            positions[j] += dx_j


class VBDSolver:
    """Vertex Block Descent solver (SIGGRAPH 2024).

    Minimizes per-vertex local elastic energy using 3x3 Newton steps.
    Graph coloring enables parallel updates without conflicts.
    Guaranteed energy descent every iteration (unconditionally stable).
    Same interface as UnifiedFEMSolver: build(), solve(), get_muscle_positions().
    """

    def __init__(self):
        self._built = False
        self._n_verts = 0
        self._n_tets = 0
        self._muscle_ranges = {}
        self._merged_bone_mesh = None
        self._orig_fixed_mask = None
        self._n_dist_constraints = 0
        self._dist_pair_i = None
        self._dist_pair_j = None
        self._dist_rest = None
        self._has_previous_solution = False

    def _compute_sliding_cohesion_targets(self, max_gap=0.0015,
                                          verbose=False):
        """Dynamic closest-surface, normal-only cohesive targets.

        Patch membership comes from rest anatomy, but the closest triangle is
        recomputed at every call. Moving a source vertex only toward that
        closest point changes the normal gap and leaves tangential sliding
        unconstrained.
        """
        if not getattr(self, '_sliding_interfaces', None):
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        all_idx, all_target = [], []
        mesh_cache = {}
        for src_name, tgt_name, src_global in self._sliding_interfaces:
            vs_t, ve_t, _, _ = self._muscle_ranges[tgt_name]
            if tgt_name not in mesh_cache:
                faces = self._muscle_surface_faces[tgt_name] - vs_t
                mesh_cache[tgt_name] = trimesh.Trimesh(
                    vertices=self.positions[vs_t:ve_t], faces=faces,
                    process=False)
            mesh = mesh_cache[tgt_name]
            movable = (~self.fixed_mask[src_global]
                       & (self._collision_region[src_global] == 2))
            src = src_global[movable]
            if not len(src):
                continue
            try:
                closest, distance, _ = mesh.nearest.on_surface(
                    self.positions[src])
            except Exception:
                continue
            separated = distance > max_gap
            if not np.any(separated):
                continue
            p = self.positions[src[separated]]
            cp = closest[separated]
            direction = p - cp
            direction /= np.maximum(
                np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
            # Retain a thin physical interface rather than welding surfaces.
            target = cp + max_gap * direction
            all_idx.append(src[separated].astype(np.int32))
            all_target.append(target)
        if not all_idx:
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        idx = np.concatenate(all_idx)
        target = np.concatenate(all_target)
        if verbose:
            print(f"    Sliding cohesion: {len(idx)} separated patch vertices")
        return idx, target

    def build(self, muscles_data):
        _ensure_ti_init()

        all_verts = []
        all_tets = []
        all_fixed = []
        all_surface_verts = []
        all_surface_faces = []
        vert_offset = 0
        tet_offset = 0

        for name, data in muscles_data.items():
            n_v = len(data['rest_positions'])
            n_t = len(data['tetrahedra'])

            all_verts.append(data['rest_positions'].astype(np.float64))
            all_tets.append(data['tetrahedra'].astype(np.int32) + vert_offset)
            all_fixed.append(data['fixed_mask'].astype(bool))

            if data.get('surface_faces') is not None:
                surf_faces = data['surface_faces'].astype(np.int32) + vert_offset
                surf_v = np.unique(surf_faces.ravel())
                all_surface_faces.append(surf_faces)
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

        # Free surface verts for collision
        self.free_surface_verts = self.surface_verts[~self._orig_fixed_mask[self.surface_verts]]
        free_set = set(self.free_indices.tolist())
        self._free_surf_global = np.array([
            gv for gv in self.free_surface_verts if gv in free_set
        ], dtype=np.int64)

        # Filter degenerate tets and precompute rest state
        self._filter_degenerate_tets()
        self._precompute_rest_state()

        # Build graph coloring and vertex-tet incidence
        self._build_graph_coloring()
        self._build_vertex_tet_incidence()

        # Build surface adjacency for smoothing
        self._build_surface_adjacency(all_surface_faces)

        # Store per-muscle surface faces for muscle-muscle collision
        self._muscle_surface_faces = {}
        sf_idx = 0
        for name, data in muscles_data.items():
            if data.get('surface_faces') is not None:
                self._muscle_surface_faces[name] = all_surface_faces[sf_idx]
                sf_idx += 1

        # Allocate Taichi fields
        self._allocate_fields()
        self._allocate_surface_smooth_fields()
        self._upload_static_data()

        self._built = True
        print(f"  VBD unified: {self._n_verts} verts, {self._n_tets} tets, "
              f"{len(self._muscle_ranges)} muscles, {len(self.free_indices)} free DOFs, "
              f"{self._n_colors} colors")

    def _filter_degenerate_tets(self):
        X = self.rest_positions
        tets = self.tetrahedra
        v0 = X[tets[:, 0]]
        e1 = X[tets[:, 1]] - v0
        e2 = X[tets[:, 2]] - v0
        e3 = X[tets[:, 3]] - v0
        Dm = np.stack([e1, e2, e3], axis=-1)
        vol = np.abs(np.linalg.det(Dm)) / 6.0
        q = X[tets]
        edge_lengths = np.stack([
            np.linalg.norm(q[:, i] - q[:, j], axis=1)
            for i, j in ((0, 1), (0, 2), (0, 3),
                         (1, 2), (1, 3), (2, 3))], axis=1)
        local_scale = np.maximum(np.max(edge_lengths, axis=1) ** 3, 1e-30)
        good = (vol / local_scale) > 1e-5
        n_removed = np.sum(~good)
        if n_removed > 0:
            print(f"  VBD: Removed {n_removed} degenerate tets (of {len(tets)})")
            self.tetrahedra = tets[good].copy()
        self._n_tets = len(self.tetrahedra)

    def _precompute_rest_state(self):
        tets = self.tetrahedra
        X = self.rest_positions
        x3 = X[tets[:, 3]]
        Bm = np.stack([X[tets[:, 0]] - x3, X[tets[:, 1]] - x3, X[tets[:, 2]] - x3], axis=-1)
        self.Bm_inv = np.linalg.inv(Bm)
        self.rest_volume = np.abs(np.linalg.det(Bm)) / 6.0

    def _build_graph_coloring(self):
        """Greedy vertex coloring from tet adjacency.

        Two vertices are adjacent if they share a tet.
        Assigns colors so no two adjacent vertices have the same color.
        """
        N = self._n_verts
        M = self._n_tets
        tets = self.tetrahedra

        # Build adjacency lists
        from collections import defaultdict
        adj = defaultdict(set)
        for t in range(M):
            verts = [tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]]
            for a in range(4):
                for b in range(a + 1, 4):
                    adj[verts[a]].add(verts[b])
                    adj[verts[b]].add(verts[a])

        # Greedy coloring
        color = np.full(N, -1, dtype=np.int32)
        for v in range(N):
            if v not in adj and self.fixed_mask[v]:
                continue  # skip isolated fixed verts
            used = set()
            for nb in adj[v]:
                if color[nb] >= 0:
                    used.add(color[nb])
            c = 0
            while c in used:
                c += 1
            color[v] = c

        n_colors = int(color.max()) + 1
        self._n_colors = n_colors

        # Build per-color vertex lists (only free vertices need solving)
        self._color_verts = []
        for c in range(n_colors):
            verts_c = np.where(color == c)[0].astype(np.int32)
            self._color_verts.append(verts_c)

        total_colored = sum(len(cv) for cv in self._color_verts)
        print(f"  VBD graph coloring: {n_colors} colors, {total_colored} vertices colored")

    def _build_vertex_tet_incidence(self):
        """Build CSR mapping each vertex to its incident tets + local index."""
        N = self._n_verts
        M = self._n_tets
        tets = self.tetrahedra

        # Count incident tets per vertex
        counts = np.zeros(N, dtype=np.int32)
        for t in range(M):
            for j in range(4):
                counts[tets[t, j]] += 1

        # Build offsets (CSR format)
        offsets = np.zeros(N + 1, dtype=np.int32)
        for i in range(N):
            offsets[i + 1] = offsets[i] + counts[i]

        total = int(offsets[N])
        tet_data = np.zeros(total, dtype=np.int32)
        local_idx = np.zeros(total, dtype=np.int32)

        # Fill data
        fill_pos = offsets[:-1].copy()
        for t in range(M):
            for j in range(4):
                v = tets[t, j]
                pos = fill_pos[v]
                tet_data[pos] = t
                local_idx[pos] = j
                fill_pos[v] += 1

        self._vert_tet_data = tet_data
        self._vert_tet_offset = offsets
        self._vert_local_idx = local_idx

        avg_incident = float(counts[counts > 0].mean())
        max_incident = int(counts.max())
        print(f"  VBD vertex-tet incidence: avg={avg_incident:.1f}, max={max_incident}")

    def _build_surface_adjacency(self, all_surface_faces):
        """Build CSR adjacency for surface vertices (per-muscle, no cross-muscle edges)."""
        from collections import defaultdict

        if not all_surface_faces:
            self._n_surf_verts = 0
            self._surf_verts_np = np.array([], dtype=np.int32)
            self._surf_adj_data_np = np.array([], dtype=np.int32)
            self._surf_adj_offset_np = np.array([0], dtype=np.int32)
            return

        adj = defaultdict(set)
        for faces in all_surface_faces:
            for f in faces:
                for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
                    adj[a].add(b)
                    adj[b].add(a)

        free_set = set(self.free_indices.tolist())
        surf_verts = sorted([v for v in adj.keys() if v in free_set])

        adj_data = []
        adj_offset = [0]
        for v in surf_verts:
            neighbors = sorted(adj[v])
            adj_data.extend(neighbors)
            adj_offset.append(len(adj_data))

        self._n_surf_verts = len(surf_verts)
        self._surf_verts_np = np.array(surf_verts, dtype=np.int32)
        self._surf_adj_data_np = np.array(adj_data, dtype=np.int32)
        self._surf_adj_offset_np = np.array(adj_offset, dtype=np.int32)
        print(f"  Surface adjacency: {self._n_surf_verts} free surface verts, "
              f"{len(adj_data)} adjacency entries")

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
        self.ti_x_tilde = ti.Vector.field(3, dtype=ti.f64, shape=N)  # inertia target

        # Vertex-tet incidence (CSR)
        total_inc = len(self._vert_tet_data)
        self.ti_vert_tet_data = ti.field(dtype=ti.i32, shape=max(total_inc, 1))
        self.ti_vert_tet_offset = ti.field(dtype=ti.i32, shape=N + 1)
        self.ti_vert_local_idx = ti.field(dtype=ti.i32, shape=max(total_inc, 1))

        # Per-color vertex lists
        self._max_color_size = max(len(cv) for cv in self._color_verts) if self._color_verts else 1
        self.ti_color_verts = ti.field(dtype=ti.i32, shape=self._max_color_size)

        # Distance constraint CSR (empty by default, populated by set_inter_muscle_constraints)
        self.ti_dist_pair_other = ti.field(dtype=ti.i32, shape=1)
        self.ti_dist_pair_rest_csr = ti.field(dtype=ti.f64, shape=1)
        self.ti_dist_vert_offset = ti.field(dtype=ti.i32, shape=N + 1)
        self.ti_dist_vert_offset.fill(0)

        # Collision fields
        self._max_coll = 0
        self.ti_coll_idx = None
        self.ti_coll_targets = None

    def _allocate_surface_smooth_fields(self):
        if self._n_surf_verts == 0:
            return
        N = self._n_verts
        S = self._n_surf_verts
        A = len(self._surf_adj_data_np)

        self.ti_smoothed = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_surf_verts = ti.field(dtype=ti.i32, shape=S)
        self.ti_surf_adj_data = ti.field(dtype=ti.i32, shape=max(A, 1))
        self.ti_surf_adj_offset = ti.field(dtype=ti.i32, shape=S + 1)

        self.ti_surf_verts.from_numpy(self._surf_verts_np)
        if A > 0:
            self.ti_surf_adj_data.from_numpy(self._surf_adj_data_np)
        self.ti_surf_adj_offset.from_numpy(self._surf_adj_offset_np)

    def _upload_static_data(self):
        self.ti_tets.from_numpy(self.tetrahedra)
        self.ti_Bm_inv.from_numpy(self.Bm_inv)
        self.ti_rest_volume.from_numpy(self.rest_volume)
        self.ti_rest_positions.from_numpy(self.rest_positions)

        # Vertex-tet incidence
        if len(self._vert_tet_data) > 0:
            self.ti_vert_tet_data.from_numpy(self._vert_tet_data)
            self.ti_vert_local_idx.from_numpy(self._vert_local_idx)
        self.ti_vert_tet_offset.from_numpy(self._vert_tet_offset)

        # Per-vertex inverse mass (same as XPBD solver)
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
        self.ti_invm.from_numpy(invm)

    def set_inter_muscle_constraints(self, constraints):
        """Convert inter-muscle constraints from local to global indices."""
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

        K = self._n_dist_constraints
        N = self._n_verts

        # Build vertex-to-distance-constraint CSR incidence
        # Each constraint (i,j,d0) contributes to BOTH vertex i and vertex j
        # For vertex vi, store (other_vertex, rest_dist) pairs
        from collections import defaultdict
        dist_adj = defaultdict(list)  # vertex -> [(other_vert, rest_dist), ...]
        for k in range(K):
            dist_adj[self._dist_pair_i[k]].append((self._dist_pair_j[k], self._dist_rest[k]))
            dist_adj[self._dist_pair_j[k]].append((self._dist_pair_i[k], self._dist_rest[k]))

        # Build CSR arrays
        dist_vert_offset = np.zeros(N + 1, dtype=np.int32)
        for v in range(N):
            dist_vert_offset[v + 1] = dist_vert_offset[v] + len(dist_adj[v])

        total_entries = int(dist_vert_offset[N])
        dist_pair_other = np.zeros(max(total_entries, 1), dtype=np.int32)
        dist_pair_rest_csr = np.zeros(max(total_entries, 1), dtype=np.float64)

        pos = 0
        for v in range(N):
            for other, rd in dist_adj[v]:
                dist_pair_other[pos] = other
                dist_pair_rest_csr[pos] = rd
                pos += 1

        # Allocate Taichi fields
        self.ti_dist_pair_i = ti.field(dtype=ti.i32, shape=K)
        self.ti_dist_pair_j = ti.field(dtype=ti.i32, shape=K)
        self.ti_dist_rest = ti.field(dtype=ti.f64, shape=K)
        self.ti_dist_pair_other = ti.field(dtype=ti.i32, shape=max(total_entries, 1))
        self.ti_dist_pair_rest_csr = ti.field(dtype=ti.f64, shape=max(total_entries, 1))
        self.ti_dist_vert_offset = ti.field(dtype=ti.i32, shape=N + 1)

        self.ti_dist_pair_i.from_numpy(self._dist_pair_i)
        self.ti_dist_pair_j.from_numpy(self._dist_pair_j)
        self.ti_dist_rest.from_numpy(self._dist_rest)
        if total_entries > 0:
            self.ti_dist_pair_other.from_numpy(dist_pair_other)
            self.ti_dist_pair_rest_csr.from_numpy(dist_pair_rest_csr)
        self.ti_dist_vert_offset.from_numpy(dist_vert_offset)

        print(f"  Inter-muscle constraints: {K} active ({skipped} skipped)")

    def update_targets_and_positions(self, muscles_data, use_lbs_init=False):
        """Update targets and positions from skeleton.

        First frame: uses LBS positions for ALL vertices (free + fixed) as initial guess.
        Subsequent frames: keeps previous solution for free vertices, only updates fixed targets.
        """
        for name, data in muscles_data.items():
            vs, ve, _, _ = self._muscle_ranges[name]

            if not self._has_previous_solution:
                # First frame: use LBS positions for everything (better starting point)
                if 'positions' in data:
                    self.positions[vs:ve] = data['positions']
            # Always update fixed targets
            if 'fixed_targets' in data:
                fi = np.where(self._orig_fixed_mask[vs:ve])[0]
                if not self._has_previous_solution:
                    # First frame: also snap fixed verts to exact targets
                    self.positions[vs + fi] = data['fixed_targets']
                global_fi = fi + vs
                for i, gfi in enumerate(global_fi):
                    idx_in_fixed = np.searchsorted(self.fixed_indices, gfi)
                    if idx_in_fixed < len(self.fixed_indices) and self.fixed_indices[idx_in_fixed] == gfi:
                        self.fixed_targets[idx_in_fixed] = data['fixed_targets'][i]

    # Reuse collision methods from UnifiedFEMSolver (same logic)
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
        self._individual_bone_meshes = valid
        all_valid = valid + non_watertight
        self._merged_bone_mesh = trimesh.util.concatenate(all_valid) if all_valid else None
        self._bone_kdtree = None

    def _compute_collision_targets(self, margin=0.002, verbose=False):
        if self._merged_bone_mesh is None:
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        free_surf = self._free_surf_global
        if len(free_surf) == 0:
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        surf_pos = self.positions[free_surf]
        from scipy.spatial import cKDTree
        proximity_radius = 0.02
        if not hasattr(self, '_bone_kdtree') or self._bone_kdtree is None:
            self._bone_kdtree = cKDTree(self._merged_bone_mesh.vertices)
        dists_to_bone, _ = self._bone_kdtree.query(surf_pos)
        nearby_mask = dists_to_bone < proximity_radius
        if not np.any(nearby_mask):
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        nearby_pos = surf_pos[nearby_mask]
        nearby_free_surf = free_surf[nearby_mask]
        inside_mask = np.zeros(len(nearby_pos), dtype=bool)
        individual = getattr(self, '_individual_bone_meshes', [])
        if individual:
            for bone_mesh in individual:
                try:
                    bmin = bone_mesh.bounds[0] - proximity_radius
                    bmax = bone_mesh.bounds[1] + proximity_radius
                    in_bbox = np.all((nearby_pos >= bmin) & (nearby_pos <= bmax), axis=1)
                    if not np.any(in_bbox):
                        continue
                    contained = bone_mesh.contains(nearby_pos[in_bbox])
                    inside_mask[in_bbox] |= contained
                except Exception:
                    continue
        if not individual:
            try:
                closest_pts, dists, face_ids = self._merged_bone_mesh.nearest.on_surface(nearby_pos)
                normals = self._merged_bone_mesh.face_normals[face_ids]
                to_vertex = nearby_pos - closest_pts
                dots = np.einsum('ij,ij->i', to_vertex, normals)
                inside_mask = (dots < 0) & (dists < margin * 10)
            except Exception:
                return np.array([], dtype=np.int32), np.zeros((0, 3))
        if not np.any(inside_mask):
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        pen_pos = nearby_pos[inside_mask]
        try:
            closest_pts, dists, face_ids = self._merged_bone_mesh.nearest.on_surface(pen_pos)
            normals = self._merged_bone_mesh.face_normals[face_ids]
        except Exception:
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        targets = closest_pts + normals * margin
        global_idx = nearby_free_surf[inside_mask].astype(np.int32)
        return global_idx, targets

    def _compute_muscle_collision_targets(self, margin=0.001, verbose=False):
        if not TRIMESH_AVAILABLE or not hasattr(self, '_muscle_surface_faces'):
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        if not self._muscle_surface_faces:
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        from scipy.spatial import cKDTree
        if not hasattr(self, '_muscle_neighbor_pairs'):
            neighbor_set = set()
            vert_to_muscle = {}
            for name, (vs, ve, _, _) in self._muscle_ranges.items():
                for v in range(vs, ve):
                    vert_to_muscle[v] = name
            if self._dist_pair_i is not None:
                for k in range(self._n_dist_constraints):
                    m1 = vert_to_muscle.get(self._dist_pair_i[k])
                    m2 = vert_to_muscle.get(self._dist_pair_j[k])
                    if m1 and m2 and m1 != m2:
                        pair = tuple(sorted([m1, m2]))
                        neighbor_set.add(pair)
            self._muscle_neighbor_pairs = list(neighbor_set)
        all_idx = []
        all_tgt = []
        for name_a, name_b in self._muscle_neighbor_pairs:
            if name_a not in self._muscle_surface_faces or name_b not in self._muscle_surface_faces:
                continue
            vs_a, ve_a, _, _ = self._muscle_ranges[name_a]
            vs_b, ve_b, _, _ = self._muscle_ranges[name_b]
            pos_a = self.positions[vs_a:ve_a]
            pos_b = self.positions[vs_b:ve_b]
            min_a, max_a = pos_a.min(axis=0), pos_a.max(axis=0)
            min_b, max_b = pos_b.min(axis=0), pos_b.max(axis=0)
            if np.any(min_a > max_b + margin) or np.any(min_b > max_a + margin):
                continue
            faces_a = self._muscle_surface_faces[name_a] - vs_a
            faces_b = self._muscle_surface_faces[name_b] - vs_b
            try:
                mesh_a = trimesh.Trimesh(vertices=pos_a, faces=faces_a, process=False)
                mesh_b = trimesh.Trimesh(vertices=pos_b, faces=faces_b, process=False)
                # Note: process=False keeps original vertex indexing.
                # Normals may be inconsistent but signed-distance still works
                # well enough — most false positives are filtered by the
                # distance threshold (dists_surf < margin * 3).
            except Exception:
                continue
            for src_name, src_mesh, tgt_mesh, src_vs in [
                (name_a, mesh_a, mesh_b, vs_a),
                (name_b, mesh_b, mesh_a, vs_b),
            ]:
                src_surf_local = np.unique(self._muscle_surface_faces[src_name] - src_vs)
                src_surf_pos = src_mesh.vertices[src_surf_local]
                tgt_tree = cKDTree(tgt_mesh.vertices)
                dists_kd, _ = tgt_tree.query(src_surf_pos)
                nearby = dists_kd < margin * 5
                if not np.any(nearby):
                    continue
                nearby_pos = src_surf_pos[nearby]
                nearby_local = src_surf_local[nearby]
                try:
                    closest_pts, dists_surf, face_ids = tgt_mesh.nearest.on_surface(nearby_pos)
                    normals = tgt_mesh.face_normals[face_ids]
                except Exception:
                    continue
                to_vertex = nearby_pos - closest_pts
                dots = np.einsum('ij,ij->i', to_vertex, normals)
                inside = (dots < 0) & (dists_surf < margin * 3)
                if not np.any(inside):
                    continue
                targets = closest_pts[inside] + normals[inside] * margin
                global_idx = (nearby_local[inside] + src_vs).astype(np.int32)
                coll_region = getattr(self, '_collision_region', None)
                if coll_region is not None:
                    valid_mask = coll_region[global_idx] == 2
                else:
                    valid_mask = ~self.fixed_mask[global_idx]
                if np.any(valid_mask):
                    all_idx.append(global_idx[valid_mask])
                    all_tgt.append(targets[valid_mask])
        if not all_idx:
            return np.array([], dtype=np.int32), np.zeros((0, 3))
        return np.concatenate(all_idx), np.concatenate(all_tgt)

    def solve(self, mu, lam, vol_penalty=100.0, kappa=1e4, margin=0.002,
              max_lbfgs_iters=100, n_outer=3, bone_meshes=None, verbose=False,
              n_load_steps=0, bone_mesh_callback=None):
        """VBD quasistatic solve with internal load stepping and collision.

        Internal load stepping interpolates fixed targets from rest to final
        over N steps, running full iterations per step. This keeps the mesh
        near equilibrium throughout, preventing inversion accumulation.

        bone_mesh_callback(alpha) rebuilds bone collision meshes at
        interpolated skeleton poses during load stepping.
        """
        if bone_meshes is not None:
            self._build_merged_bone_mesh(bone_meshes)

        N = self._n_verts
        M = self._n_tets
        n_iters = max_lbfgs_iters
        K = self._n_dist_constraints

        effective_lam = lam + vol_penalty
        effective_mu = mu

        # Hessian regularization epsilon (scales with material stiffness)
        eps = max(effective_mu, effective_lam) * 1e-4

        # Distance constraint stiffness
        dist_stiffness = effective_lam if K > 0 else 0.0

        # Pre-upload all color vertex arrays (padded to max size)
        color_arrays = []
        for c in range(self._n_colors):
            cv = self._color_verts[c]
            padded = np.zeros(self._max_color_size, dtype=np.int32)
            padded[:len(cv)] = cv
            color_arrays.append((padded, len(cv)))

        has_bone_coll = self._merged_bone_mesh is not None
        has_mm_coll = hasattr(self, '_muscle_surface_faces') and bool(self._muscle_surface_faces)
        n_coll = 0
        n_mm_coll = 0

        # Internal load stepping: interpolate fixed targets from rest to final
        # over N steps, with full iterations per step. This keeps vertices near
        # equilibrium throughout, preventing inversion accumulation.
        if n_load_steps <= 0:
            n_load_steps = 10 if not self._has_previous_solution else 1

        rest_fixed_targets = self.rest_positions[self.fixed_indices].copy()
        final_fixed_targets = self.fixed_targets.copy()

        # h=1.0: inertia regularization prevents divergence from large Newton steps.
        h = 1.0
        mass_over_h2 = 1.0 / (h * h)

        # Upload positions
        self.ti_positions.from_numpy(self.positions)

        # Save positions before solve for convergence metric
        initial_pos = self.positions.copy()

        if verbose:
            X0 = self.ti_positions.to_numpy()
            T = self.tetrahedra
            x3 = X0[T[:, 3]]
            Ds0 = np.stack([X0[T[:, 0]] - x3, X0[T[:, 1]] - x3, X0[T[:, 2]] - x3], axis=-1)
            Js0 = np.linalg.det(Ds0 @ self.Bm_inv)
            n_inv0 = int(np.sum(Js0 <= 0))
            print(f"    initial: inv={n_inv0}/{M} J=[{np.min(Js0):.3f},{np.max(Js0):.3f}]")

        global_it = 0
        for load_step in range(n_load_steps):
            alpha = (load_step + 1) / n_load_steps

            # Interpolate fixed targets: rest → final
            interp_targets = rest_fixed_targets + alpha * (final_fixed_targets - rest_fixed_targets)
            self.fixed_targets[:] = interp_targets

            targets_full = self.ti_positions.to_numpy()
            targets_full[self.fixed_indices] = interp_targets
            self.ti_targets.from_numpy(targets_full)
            _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)

            # Re-anchor inertia at start of each load step
            _copy_positions(self.ti_x_tilde, self.ti_positions, N)

            for it in range(n_iters):
                for c in range(self._n_colors):
                    padded, n_cv = color_arrays[c]
                    if n_cv == 0:
                        continue
                    self.ti_color_verts.from_numpy(padded)
                    _vbd_solve_color(
                        self.ti_positions,
                        self.ti_x_tilde,
                        self.ti_Bm_inv,
                        self.ti_rest_volume,
                        self.ti_vert_tet_data,
                        self.ti_vert_tet_offset,
                        self.ti_vert_local_idx,
                        self.ti_tets,
                        self.ti_invm,
                        self.ti_color_verts,
                        self.ti_dist_pair_other,
                        self.ti_dist_pair_rest_csr,
                        self.ti_dist_vert_offset,
                        dist_stiffness,
                        K,
                        effective_mu,
                        effective_lam,
                        eps,
                        mass_over_h2,
                        n_cv,
                    )
                _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)
                global_it += 1

            # Collision at each load step — keeps muscles on correct side
            # Rebuild bone meshes at interpolated pose if callback provided
            if bone_mesh_callback is not None:
                step_bone_meshes = bone_mesh_callback(alpha)
                if step_bone_meshes is not None:
                    self._build_merged_bone_mesh(step_bone_meshes)

            self.positions = self.ti_positions.to_numpy()
            if has_bone_coll and self._merged_bone_mesh is not None:
                coll_idx_np, coll_tgt_np = self._compute_collision_targets(
                    margin, verbose=False)
                if len(coll_idx_np) > 0:
                    n_coll = len(coll_idx_np)
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

            if has_mm_coll:
                self.positions = self.ti_positions.to_numpy()
                mm_idx, mm_tgt = self._compute_muscle_collision_targets(
                    margin=margin, verbose=False)
                if len(mm_idx) > 0:
                    n_mm_coll = len(mm_idx)
                    if n_mm_coll > self._max_coll:
                        self._max_coll = max(n_mm_coll, 1000)
                        self.ti_coll_idx = ti.field(dtype=ti.i32, shape=self._max_coll)
                        self.ti_coll_targets = ti.Vector.field(3, dtype=ti.f64, shape=self._max_coll)
                    idx_padded = np.zeros(self._max_coll, dtype=np.int32)
                    tgt_padded = np.zeros((self._max_coll, 3), dtype=np.float64)
                    idx_padded[:n_mm_coll] = mm_idx
                    tgt_padded[:n_mm_coll] = mm_tgt
                    self.ti_coll_idx.from_numpy(idx_padded)
                    self.ti_coll_targets.from_numpy(tgt_padded)
                    _apply_collision_projection(
                        self.ti_positions, self.ti_invm,
                        self.ti_coll_idx, self.ti_coll_targets,
                        1.0, n_mm_coll,
                    )

        # Restore final fixed targets
        self.fixed_targets[:] = final_fixed_targets

        # Taubin smoothing
        if self._n_surf_verts > 0:
            S = self._n_surf_verts
            lam_smooth = 0.3
            mu_smooth = -0.31
            for _ in range(2):
                _laplacian_smooth_surface(
                    self.ti_positions, self.ti_smoothed, self.ti_invm,
                    self.ti_surf_verts, self.ti_surf_adj_data,
                    self.ti_surf_adj_offset, lam_smooth, S)
                _copy_smoothed_to_positions(
                    self.ti_positions, self.ti_smoothed,
                    self.ti_surf_verts, self.ti_invm, S)
                _laplacian_smooth_surface(
                    self.ti_positions, self.ti_smoothed, self.ti_invm,
                    self.ti_surf_verts, self.ti_surf_adj_data,
                    self.ti_surf_adj_offset, mu_smooth, S)
                _copy_smoothed_to_positions(
                    self.ti_positions, self.ti_smoothed,
                    self.ti_surf_verts, self.ti_invm, S)
            _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)

        # Final collision pass after smoothing (guarantees clean output)
        if has_mm_coll:
            self.positions = self.ti_positions.to_numpy()
            mm_idx, mm_tgt = self._compute_muscle_collision_targets(
                margin=margin, verbose=verbose)
            n_mm_coll = len(mm_idx)
            if n_mm_coll > 0:
                if n_mm_coll > self._max_coll:
                    self._max_coll = max(n_mm_coll, 1000)
                    self.ti_coll_idx = ti.field(dtype=ti.i32, shape=self._max_coll)
                    self.ti_coll_targets = ti.Vector.field(3, dtype=ti.f64, shape=self._max_coll)
                idx_padded = np.zeros(self._max_coll, dtype=np.int32)
                tgt_padded = np.zeros((self._max_coll, 3), dtype=np.float64)
                idx_padded[:n_mm_coll] = mm_idx
                tgt_padded[:n_mm_coll] = mm_tgt
                self.ti_coll_idx.from_numpy(idx_padded)
                self.ti_coll_targets.from_numpy(tgt_padded)
                _apply_collision_projection(
                    self.ti_positions, self.ti_invm,
                    self.ti_coll_idx, self.ti_coll_targets,
                    1.0, n_mm_coll,
                )

        # Bone collision (last — guarantees output is penetration-free)
        if has_bone_coll:
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

        # Download positions
        self.positions = self.ti_positions.to_numpy()

        # Convergence metric
        free = self.free_indices
        diff = self.positions[free] - initial_pos[free]
        residual = float(np.sqrt(np.sum(diff ** 2)))

        if verbose:
            X = self.positions
            T = self.tetrahedra
            x3 = X[T[:, 3]]
            Ds = np.stack([X[T[:, 0]] - x3, X[T[:, 1]] - x3, X[T[:, 2]] - x3], axis=-1)
            Js = np.linalg.det(Ds @ self.Bm_inv)
            n_inv = int(np.sum(Js <= 0))
            print(f"    final: iters={n_iters} ||dx||={residual:.4e} "
                  f"inv={n_inv}/{M} dist={K} "
                  f"J=[{np.min(Js):.3f},{np.max(Js):.3f}]")

        self._has_previous_solution = True
        return n_iters, residual

    def get_muscle_positions(self, name):
        vs, ve, _, _ = self._muscle_ranges[name]
        return self.positions[vs:ve].copy()


# ---------------------------------------------------------------------------
# Projected Newton Solver (CG + inversion-safe line search)
# ---------------------------------------------------------------------------

@ti.kernel
def _pn_compute_energy(
    positions: ti.template(), invm: ti.template(),
    tets: ti.template(), Bm_inv: ti.template(), rest_volume: ti.template(),
    mu: ti.f64, lam: ti.f64, n_tets: ti.i32,
) -> ti.f64:
    """Compute total Stable Neo-Hookean energy."""
    E = 0.0
    for k in range(n_tets):
        i0 = tets[k][0]; i1 = tets[k][1]; i2 = tets[k][2]; i3 = tets[k][3]
        Ds = ti.Matrix.cols([positions[i0] - positions[i3],
                              positions[i1] - positions[i3],
                              positions[i2] - positions[i3]])
        F = Ds @ Bm_inv[k]
        I1 = F.norm_sqr()
        J = F.determinant()
        V0 = rest_volume[k]
        ti.atomic_add(E, V0 * (mu * 0.5 * (I1 - 3.0) + lam * 0.5 * (J - 1.0) * (J - 1.0)))
    return E


@ti.kernel
def _pn_compute_gradient(
    positions: ti.template(), gradient: ti.template(), invm: ti.template(),
    tets: ti.template(), Bm_inv: ti.template(), rest_volume: ti.template(),
    mu: ti.f64, lam: ti.f64, n_tets: ti.i32,
):
    """Compute global gradient of Stable Neo-Hookean energy. g = ∂E/∂x."""
    for k in range(n_tets):
        i0 = tets[k][0]; i1 = tets[k][1]; i2 = tets[k][2]; i3 = tets[k][3]
        Ds = ti.Matrix.cols([positions[i0] - positions[i3],
                              positions[i1] - positions[i3],
                              positions[i2] - positions[i3]])
        B = Bm_inv[k]
        F = Ds @ B
        V0 = rest_volume[k]
        f0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
        f1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
        f2 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])
        J = F.determinant()
        cof = ti.Matrix.cols([f1.cross(f2), f2.cross(f0), f0.cross(f1)])
        # P = mu*F + lam*(J-1)*cof
        P = mu * F + lam * (J - 1.0) * cof
        # gradient per vertex = V0 * P @ B^T, extracted per column
        PBt = P @ B.transpose()
        g0 = V0 * ti.Vector([PBt[0, 0], PBt[1, 0], PBt[2, 0]])
        g1 = V0 * ti.Vector([PBt[0, 1], PBt[1, 1], PBt[2, 1]])
        g2 = V0 * ti.Vector([PBt[0, 2], PBt[1, 2], PBt[2, 2]])
        g3 = -(g0 + g1 + g2)
        ti.atomic_add(gradient[i0], g0)
        ti.atomic_add(gradient[i1], g1)
        ti.atomic_add(gradient[i2], g2)
        ti.atomic_add(gradient[i3], g3)


@ti.kernel
def _pn_hessian_vec(
    positions: ti.template(), p_vec: ti.template(), result: ti.template(),
    invm: ti.template(),
    tets: ti.template(), Bm_inv: ti.template(), rest_volume: ti.template(),
    mu: ti.f64, lam: ti.f64, n_tets: ti.i32,
):
    """Matrix-free Hessian-vector product: result = H @ p_vec.

    Uses Gauss-Newton approximation of the Hessian (drops ∂cof/∂F term).
    For each tet: dF = dDs @ Bm_inv, dP_GN = mu*dF + lam*(cof:dF)*cof.
    """
    for k in range(n_tets):
        i0 = tets[k][0]; i1 = tets[k][1]; i2 = tets[k][2]; i3 = tets[k][3]
        B = Bm_inv[k]
        V0 = rest_volume[k]

        # Current F for cofactor
        Ds = ti.Matrix.cols([positions[i0] - positions[i3],
                              positions[i1] - positions[i3],
                              positions[i2] - positions[i3]])
        F = Ds @ B
        f0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
        f1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
        f2 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]])
        cof = ti.Matrix.cols([f1.cross(f2), f2.cross(f0), f0.cross(f1)])

        # Search direction for this tet's vertices
        p0 = p_vec[i0]; p1 = p_vec[i1]; p2 = p_vec[i2]; p3 = p_vec[i3]
        dDs = ti.Matrix.cols([p0 - p3, p1 - p3, p2 - p3])
        dF = dDs @ B

        # dP_GN = mu*dF + lam*(cof:dF)*cof  (Gauss-Newton)
        cof_dF = 0.0  # cof : dF (Frobenius inner product)
        for ii in range(3):
            for jj in range(3):
                cof_dF += cof[ii, jj] * dF[ii, jj]
        dP = mu * dF + lam * cof_dF * cof

        # H*p per vertex = V0 * dP @ B^T
        dPBt = dP @ B.transpose()
        h0 = V0 * ti.Vector([dPBt[0, 0], dPBt[1, 0], dPBt[2, 0]])
        h1 = V0 * ti.Vector([dPBt[0, 1], dPBt[1, 1], dPBt[2, 1]])
        h2 = V0 * ti.Vector([dPBt[0, 2], dPBt[1, 2], dPBt[2, 2]])
        h3 = -(h0 + h1 + h2)
        ti.atomic_add(result[i0], h0)
        ti.atomic_add(result[i1], h1)
        ti.atomic_add(result[i2], h2)
        ti.atomic_add(result[i3], h3)


@ti.kernel
def _pn_max_step_size(
    positions: ti.template(), direction: ti.template(),
    tets: ti.template(), Bm_inv: ti.template(), n_tets: ti.i32,
) -> ti.f64:
    """Shape-safe step filter: keep every valid tet above a finite J floor.

    J(α) = det(Ds(x + α*p) @ Bm_inv) is cubic in α.
    Merely keeping J positive permits elements to become arbitrarily flat and
    leaves no feasible step for the next moving-attachment increment.  Preserve
    five percent of rest volume; the outer continuation separately rejects any
    boundary move that cannot meet this condition.
    """
    alpha_max = 1.0
    # Keep a buffer above the outer continuation's 0.05 acceptance floor so
    # the next moving-boundary increment has nonzero deformation room.
    j_floor = 0.10
    for k in range(n_tets):
        i0 = tets[k][0]; i1 = tets[k][1]; i2 = tets[k][2]; i3 = tets[k][3]
        B = Bm_inv[k]
        x0 = positions[i0]; x1 = positions[i1]; x2 = positions[i2]; x3 = positions[i3]
        p0 = direction[i0]; p1 = direction[i1]; p2 = direction[i2]; p3 = direction[i3]

        # Current J (should be > 0)
        Ds0 = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])
        J0 = (Ds0 @ B).determinant()
        # J at α: binary search for zero crossing
        a_lo = 0.0
        a_hi = alpha_max
        dDs = ti.Matrix.cols([p0 - p3, p1 - p3, p2 - p3])
        Ds1 = Ds0 + a_hi * dDs
        J1 = (Ds1 @ B).determinant()

        # Above the desired floor, do not cross it. If an incoming boundary
        # step is already below the floor, do not let Newton make it flatter;
        # it may still take directions that recover volume.
        target_j = j_floor if J0 > j_floor else J0 * 0.999
        if J1 >= target_j:
            continue  # full step is safe for this tet

        # Binary search for zero crossing
        for _ in range(10):
            a_mid = (a_lo + a_hi) * 0.5
            Ds_mid = Ds0 + a_mid * dDs
            J_mid = (Ds_mid @ B).determinant()
            if J_mid >= target_j:
                a_lo = a_mid
            else:
                a_hi = a_mid

        safe_alpha = a_lo * 0.9
        ti.atomic_min(alpha_max, safe_alpha)
    return alpha_max


@ti.kernel
def _pn_dot(a: ti.template(), b: ti.template(), invm: ti.template(), n: ti.i32) -> ti.f64:
    """Dot product of two vectors (free DOFs only)."""
    s = 0.0
    for i in range(n):
        if invm[i] > 0.0:
            ti.atomic_add(s, a[i].dot(b[i]))
    return s


@ti.kernel
def _pn_axpy(y: ti.template(), a: ti.f64, x: ti.template(), invm: ti.template(), n: ti.i32):
    """y += a * x for free DOFs."""
    for i in range(n):
        if invm[i] > 0.0:
            y[i] += a * x[i]


@ti.kernel
def _pn_zero(v: ti.template(), n: ti.i32):
    for i in range(n):
        v[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def _pn_copy(dst: ti.template(), src: ti.template(), n: ti.i32):
    for i in range(n):
        dst[i] = src[i]


@ti.kernel
def _pn_scale(v: ti.template(), s: ti.f64, invm: ti.template(), n: ti.i32):
    for i in range(n):
        if invm[i] > 0.0:
            v[i] *= s


@ti.kernel
def _pn_zero_fixed(v: ti.template(), invm: ti.template(), n: ti.i32):
    """Zero out fixed DOFs."""
    for i in range(n):
        if invm[i] == 0.0:
            v[i] = ti.Vector([0.0, 0.0, 0.0])


class ProjectedNewtonSolver:
    """Projected Newton solver with CG and inversion-safe line search.

    Guarantees:
    - Energy decreases every Newton step (Armijo line search)
    - J > 0 for all tets at every step (inversion filter)
    - Quadratic convergence near solution

    Same interface as UnifiedFEMSolver: build(), solve(), get_muscle_positions().
    """

    def __init__(self):
        self._built = False
        self._n_verts = 0
        self._n_tets = 0
        self._muscle_ranges = {}
        self._merged_bone_mesh = None
        self._orig_fixed_mask = None
        self._has_previous_solution = False
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
        all_surface_faces = []
        vert_offset = 0
        tet_offset = 0

        for name, data in muscles_data.items():
            n_v = len(data['rest_positions'])
            n_t = len(data['tetrahedra'])
            all_verts.append(data['rest_positions'].astype(np.float64))
            all_tets.append(data['tetrahedra'].astype(np.int32) + vert_offset)
            all_fixed.append(data['fixed_mask'].astype(bool))
            if data.get('surface_faces') is not None:
                surf_faces = data['surface_faces'].astype(np.int32) + vert_offset
                surf_v = np.unique(surf_faces.ravel())
                all_surface_faces.append(surf_faces)
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
        # Fine tetrahedralizations put several very small elements next to an
        # attachment cap.  Exact Dirichlet motion of a single cap vertex can
        # then make a positive-J continuation step mathematically impossible,
        # even though a real tendon insertion is compliant over a finite area.
        # In compliant mode the same anatomical cap vertices remain attachment
        # vertices, but are coupled to the bone with positional energy instead
        # of being removed from the solve.
        self._compliant_attachments = (
            os.environ.get('MUSCLE_PN_COMPLIANT_ATTACH', '0') == '1')
        self.fixed_mask = self._orig_fixed_mask.copy()
        if self._compliant_attachments:
            self.fixed_mask[:] = False
        self.surface_verts = np.concatenate(all_surface_verts, axis=0)
        self.fixed_indices = np.where(self.fixed_mask)[0]
        self.free_indices = np.where(~self.fixed_mask)[0]
        self.fixed_targets = self.positions[self.fixed_indices].copy()
        self._attachment_indices = np.where(self._orig_fixed_mask)[0].astype(
            np.int32)
        self._attachment_targets = self.positions[
            self._attachment_indices].copy()
        self._attachment_kappa = float(os.environ.get(
            'MUSCLE_PN_ATTACHMENT_KAPPA', '25000'))

        # Free surface verts for collision
        self.free_surface_verts = self.surface_verts[~self._orig_fixed_mask[self.surface_verts]]
        free_set = set(self.free_indices.tolist())
        self._free_surf_global = np.array([
            gv for gv in self.free_surface_verts if gv in free_set
        ], dtype=np.int64)

        # Filter degenerate tets
        X = self.rest_positions
        tets = self.tetrahedra
        v0 = X[tets[:, 0]]
        e1 = X[tets[:, 1]] - v0; e2 = X[tets[:, 2]] - v0; e3 = X[tets[:, 3]] - v0
        Dm = np.stack([e1, e2, e3], axis=-1)
        vol = np.abs(np.linalg.det(Dm)) / 6.0
        q = X[tets]
        edge_lengths = np.stack([
            np.linalg.norm(q[:, i] - q[:, j], axis=1)
            for i, j in ((0, 1), (0, 2), (0, 3),
                         (1, 2), (1, 3), (2, 3))], axis=1)
        local_scale = np.maximum(np.max(edge_lengths, axis=1) ** 3, 1e-30)
        good = (vol / local_scale) > 1e-5
        n_removed = np.sum(~good)
        if n_removed > 0:
            print(f"  PN: Removed {n_removed} degenerate tets (of {len(tets)})")
            self.tetrahedra = tets[good].copy()
        self._n_tets = len(self.tetrahedra)

        # Precompute rest state
        tets = self.tetrahedra
        X = self.rest_positions
        x3 = X[tets[:, 3]]
        Bm = np.stack([X[tets[:, 0]] - x3, X[tets[:, 1]] - x3, X[tets[:, 2]] - x3], axis=-1)
        self.Bm_inv = np.linalg.inv(Bm)
        self.rest_volume = np.abs(np.linalg.det(Bm)) / 6.0

        # Collision infrastructure
        self._muscle_surface_faces = {}
        sf_idx = 0
        for name, data in muscles_data.items():
            if data.get('surface_faces') is not None:
                self._muscle_surface_faces[name] = all_surface_faces[sf_idx]
                sf_idx += 1

        # PN uses reciprocal rest-pose patches only to decide anatomical
        # neighbors. The target triangle is recomputed later, so no tangential
        # material coordinates are fixed.
        self._sliding_interfaces = []
        if TRIMESH_AVAILABLE:
            from scipy.spatial import cKDTree
            names = list(self._muscle_surface_faces)
            for ia, name_a in enumerate(names):
                surf_a = np.unique(self._muscle_surface_faces[name_a])
                pa = self.rest_positions[surf_a]
                for name_b in names[ia + 1:]:
                    surf_b = np.unique(self._muscle_surface_faces[name_b])
                    pb = self.rest_positions[surf_b]
                    da, _ = cKDTree(pb).query(pa)
                    db, _ = cKDTree(pa).query(pb)
                    keep_a = surf_a[da < 0.006]
                    keep_b = surf_b[db < 0.006]
                    if len(keep_a) >= 12 and len(keep_b) >= 12:
                        self._sliding_interfaces.append(
                            (name_a, name_b, keep_a.astype(np.int32)))
                        self._sliding_interfaces.append(
                            (name_b, name_a, keep_b.astype(np.int32)))
            print(f"  PN sliding interfaces: {len(self._sliding_interfaces)} "
                  "directed reciprocal patches")

        # Collision regions
        from collections import defaultdict, deque
        self._collision_region = np.full(self._n_verts, 2, dtype=np.int32)
        self._collision_region[self.fixed_indices] = 0
        adj = defaultdict(set)
        for t in range(self._n_tets):
            verts = self.tetrahedra[t]
            for a in range(4):
                for b in range(a + 1, 4):
                    adj[verts[a]].add(verts[b])
                    adj[verts[b]].add(verts[a])
        q = deque()
        for fi in self.fixed_indices:
            q.append((fi, 0))
        visited = set(self.fixed_indices.tolist())
        while q:
            v, depth = q.popleft()
            if depth >= 2:
                continue
            for nb in adj[v]:
                if nb not in visited:
                    visited.add(nb)
                    self._collision_region[nb] = 1
                    q.append((nb, depth + 1))

        # Allocate Taichi fields
        N = self._n_verts
        M = self._n_tets
        self.ti_positions = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_targets = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_invm = ti.field(dtype=ti.f64, shape=N)
        self.ti_tets = ti.Vector.field(4, dtype=ti.i32, shape=M)
        self.ti_Bm_inv = ti.Matrix.field(3, 3, dtype=ti.f64, shape=M)
        self.ti_rest_volume = ti.field(dtype=ti.f64, shape=M)
        # CG vectors
        self.ti_gradient = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_direction = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_residual = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_hv = ti.Vector.field(3, dtype=ti.f64, shape=N)
        self.ti_pos_backup = ti.Vector.field(3, dtype=ti.f64, shape=N)
        # Collision
        self._max_coll = 0
        self.ti_coll_idx = None
        self.ti_coll_targets = None

        # Upload static data
        self.ti_tets.from_numpy(self.tetrahedra)
        self.ti_Bm_inv.from_numpy(self.Bm_inv)
        self.ti_rest_volume.from_numpy(self.rest_volume)

        # Inverse mass
        rho = 1060.0
        vertex_mass = np.zeros(N, dtype=np.float64)
        for t in range(M):
            m = rho * self.rest_volume[t] / 4.0
            for j in range(4):
                vertex_mass[self.tetrahedra[t, j]] += m
        invm = np.zeros(N, dtype=np.float64)
        free_mask = vertex_mass > 0
        invm[free_mask] = 1.0 / vertex_mass[free_mask]
        invm[self.fixed_indices] = 0.0
        self.ti_invm.from_numpy(invm)

        self._built = True
        print(f"  PN unified: {N} verts, {M} tets, "
              f"{len(self._muscle_ranges)} muscles, {len(self.free_indices)} free DOFs")
        if self._compliant_attachments:
            print(f"  PN compliant attachments: "
                  f"{len(self._attachment_indices)} vertices, "
                  f"kappa={self._attachment_kappa:g}")

    def set_inter_muscle_constraints(self, constraints):
        # Store but don't use in Newton (handled by collision instead)
        self._n_dist_constraints = 0

    def update_targets_and_positions(self, muscles_data, use_lbs_init=False):
        for name, data in muscles_data.items():
            vs, ve, _, _ = self._muscle_ranges[name]
            if not self._has_previous_solution:
                if 'positions' in data:
                    self.positions[vs:ve] = data['positions']
            if 'fixed_targets' in data:
                fi = np.where(self._orig_fixed_mask[vs:ve])[0]
                global_fi = fi + vs
                if self._compliant_attachments:
                    attachment_lookup = {
                        int(g): i for i, g in enumerate(
                            self._attachment_indices)}
                    for i, gfi in enumerate(global_fi):
                        ai = attachment_lookup.get(int(gfi))
                        if ai is not None:
                            self._attachment_targets[ai] = (
                                data['fixed_targets'][i])
                    continue
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
        self._individual_bone_meshes = valid
        all_valid = valid + non_watertight
        self._merged_bone_mesh = trimesh.util.concatenate(all_valid) if all_valid else None
        self._bone_kdtree = None

    # Reuse collision methods from XPBD solver
    _compute_collision_targets = UnifiedFEMSolver._compute_collision_targets
    _compute_muscle_collision_targets = UnifiedFEMSolver._compute_muscle_collision_targets
    _compute_sliding_cohesion_targets = VBDSolver._compute_sliding_cohesion_targets

    def solve(self, mu, lam, vol_penalty=100.0, kappa=1e4, margin=0.002,
              max_lbfgs_iters=100, n_outer=3, bone_meshes=None, verbose=False,
              n_load_steps=0, bone_mesh_callback=None,
              muscle_contact=True):
        """Projected Newton solve with CG and inversion-safe line search."""
        if bone_meshes is not None:
            self._build_merged_bone_mesh(bone_meshes)

        N = self._n_verts
        M = self._n_tets
        effective_lam = lam + vol_penalty
        effective_mu = mu
        max_newton = max_lbfgs_iters
        max_cg = 50

        # Upload positions and set targets
        targets_full = self.positions.copy()
        targets_full[self.fixed_indices] = self.fixed_targets
        self.ti_targets.from_numpy(targets_full)
        self.ti_positions.from_numpy(self.positions)
        _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)

        initial_pos = self.ti_positions.to_numpy().copy()

        if verbose:
            X0 = self.ti_positions.to_numpy()
            T = self.tetrahedra
            x3 = X0[T[:, 3]]
            Ds0 = np.stack([X0[T[:, 0]] - x3, X0[T[:, 1]] - x3, X0[T[:, 2]] - x3], axis=-1)
            Js0 = np.linalg.det(Ds0 @ self.Bm_inv)
            n_inv0 = int(np.sum(Js0 <= 0))
            print(f"    PN initial: inv={n_inv0}/{M} J=[{np.min(Js0):.3f},{np.max(Js0):.3f}]")

        # For first frame: start from rest (0 inversions) with internal load stepping.
        # Newton + inversion-safe line search guarantees J > 0 throughout.
        # For subsequent frames: start from previous solution (temporal coherence).
        if self._compliant_attachments:
            coll_penalty_idx = self._attachment_indices.copy()
            coll_penalty_tgt = self._attachment_targets.copy()
            collision_kappa = self._attachment_kappa
        else:
            coll_penalty_idx = np.array([], dtype=np.int32)
            coll_penalty_tgt = np.zeros((0, 3), dtype=np.float64)
            collision_kappa = 0.0
        n_penalty = len(coll_penalty_idx)

        # Use LBS init + hard attachments.
        # The PN solver with inversion-safe line search preserves LBS inversions
        # (~2,235) but doesn't create new ones. Penalty collision avoids the
        # 5,600+ inversions that post-hoc projection was creating.
        E_prev = _pn_compute_energy(
            self.ti_positions, self.ti_invm,
            self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
            effective_mu, effective_lam, M)
        if n_penalty > 0:
            pos_np = self.ti_positions.to_numpy()
            penalty_delta = pos_np[coll_penalty_idx] - coll_penalty_tgt
            E_prev += 0.5 * collision_kappa * float(
                np.sum(penalty_delta * penalty_delta))

        # Final Newton iterations at target pose
        for newton_it in range(max_newton):
            # 1. Compute gradient (elastic + collision penalty)
            _pn_zero(self.ti_gradient, N)
            _pn_compute_gradient(
                self.ti_positions, self.ti_gradient, self.ti_invm,
                self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
                effective_mu, effective_lam, M)
            # Add collision penalty gradient: kappa * (x_i - target_i)
            if n_penalty > 0:
                pos_np = self.ti_positions.to_numpy()
                penalty_grad = np.zeros((N, 3), dtype=np.float64)
                for pi in range(n_penalty):
                    vi = coll_penalty_idx[pi]
                    penalty_grad[vi] += collision_kappa * (pos_np[vi] - coll_penalty_tgt[pi])
                # Upload penalty gradient to ti_gradient
                grad_np = self.ti_gradient.to_numpy()
                grad_np += penalty_grad
                self.ti_gradient.from_numpy(grad_np)
            _pn_zero_fixed(self.ti_gradient, self.ti_invm, N)

            grad_norm = _pn_dot(self.ti_gradient, self.ti_gradient, self.ti_invm, N) ** 0.5
            if verbose and newton_it < 5:
                print(f"      newton {newton_it}: E={E_prev:.6e} |g|={grad_norm:.6e}")
            if grad_norm < 1e-6:
                break
            # CG reuses ``ti_gradient`` as its Hessian-vector scratch buffer.
            # Preserve the true gradient for the Armijo directional
            # derivative below.
            armijo_gradient = self.ti_gradient.to_numpy()

            # 2. CG: solve H @ direction = -gradient
            # Initialize: r = -g, d = r, direction = 0
            _pn_zero(self.ti_direction, N)
            _pn_copy(self.ti_residual, self.ti_gradient, N)
            _pn_scale(self.ti_residual, -1.0, self.ti_invm, N)

            # d = r (initial search direction)
            _pn_zero(self.ti_hv, N)
            _pn_copy(self.ti_hv, self.ti_residual, N)  # use hv as d temporarily

            rr = _pn_dot(self.ti_residual, self.ti_residual, self.ti_invm, N)

            for cg_it in range(max_cg):
                if rr < 1e-12:
                    break
                # Compute (H + eps*I) @ d → store in gradient (reuse as temp)
                _pn_zero(self.ti_gradient, N)  # reuse as Hd
                _pn_hessian_vec(
                    self.ti_positions, self.ti_hv, self.ti_gradient,
                    self.ti_invm,
                    self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
                    effective_mu, effective_lam, M)
                # Regularization: add eps * d to make Hessian SPD.
                # Must be large enough to overcome negative curvature from inverted tets.
                _pn_axpy(self.ti_gradient, effective_lam * 0.1, self.ti_hv, self.ti_invm, N)
                # Add collision penalty Hessian: kappa * d_i for penalty vertices
                if n_penalty > 0:
                    hv_np = self.ti_gradient.to_numpy()
                    d_np = self.ti_hv.to_numpy()
                    for pi in range(n_penalty):
                        vi = coll_penalty_idx[pi]
                        hv_np[vi] += collision_kappa * d_np[vi]
                    self.ti_gradient.from_numpy(hv_np)
                _pn_zero_fixed(self.ti_gradient, self.ti_invm, N)

                dHd = _pn_dot(self.ti_hv, self.ti_gradient, self.ti_invm, N)
                if dHd <= 0.0:
                    # Negative curvature — use current direction as-is
                    if cg_it == 0:
                        # No progress yet, use steepest descent
                        _pn_copy(self.ti_direction, self.ti_residual, N)
                    break

                alpha_cg = rr / dHd
                _pn_axpy(self.ti_direction, alpha_cg, self.ti_hv, self.ti_invm, N)
                _pn_axpy(self.ti_residual, -alpha_cg, self.ti_gradient, self.ti_invm, N)

                rr_new = _pn_dot(self.ti_residual, self.ti_residual, self.ti_invm, N)
                beta = rr_new / (rr + 1e-30)
                # d = r + beta * d
                _pn_scale(self.ti_hv, beta, self.ti_invm, N)
                _pn_axpy(self.ti_hv, 1.0, self.ti_residual, self.ti_invm, N)
                rr = rr_new

            # 3. Inversion-safe step filter
            _pn_zero_fixed(self.ti_direction, self.ti_invm, N)
            alpha_safe = _pn_max_step_size(
                self.ti_positions, self.ti_direction,
                self.ti_tets, self.ti_Bm_inv, M)
            alpha_safe = min(alpha_safe, 1.0)

            # 4. Armijo backtracking line search (elastic + penalty energy)
            direction_np = self.ti_direction.to_numpy()
            dir_deriv = float(np.sum(
                armijo_gradient[self.free_indices]
                * direction_np[self.free_indices]))
            # Truncated CG may return a non-descent direction when the
            # Gauss-Newton system is poorly conditioned by moving attachment
            # and contact penalties. Armijo cannot accept such a direction.
            # Fall back to the negative gradient, which is guaranteed to be a
            # descent direction, and recompute the inversion-safe step bound.
            if not np.isfinite(dir_deriv) or dir_deriv >= 0.0:
                direction_np = -armijo_gradient
                direction_np[self.fixed_indices] = 0.0
                self.ti_direction.from_numpy(direction_np)
                _pn_zero_fixed(self.ti_direction, self.ti_invm, N)
                alpha_safe = _pn_max_step_size(
                    self.ti_positions, self.ti_direction,
                    self.ti_tets, self.ti_Bm_inv, M)
                alpha_safe = min(alpha_safe, 1.0)
                dir_deriv = -float(np.sum(
                    armijo_gradient[self.free_indices] ** 2))

            alpha = alpha_safe
            c1 = 1e-4
            _pn_copy(self.ti_pos_backup, self.ti_positions, N)

            line_search_accepted = False
            for ls_it in range(15):
                _pn_copy(self.ti_positions, self.ti_pos_backup, N)
                _pn_axpy(self.ti_positions, alpha, self.ti_direction, self.ti_invm, N)
                _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)

                E_new = _pn_compute_energy(
                    self.ti_positions, self.ti_invm,
                    self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
                    effective_mu, effective_lam, M)
                # Add penalty energy
                if n_penalty > 0:
                    pos_ls = self.ti_positions.to_numpy()
                    for pi in range(n_penalty):
                        vi = coll_penalty_idx[pi]
                        d = pos_ls[vi] - coll_penalty_tgt[pi]
                        E_new += 0.5 * collision_kappa * np.dot(d, d)

                if E_new <= E_prev + c1 * alpha * dir_deriv:
                    line_search_accepted = True
                    break
                alpha *= 0.5

            if line_search_accepted:
                E_prev = E_new
            else:
                _pn_copy(self.ti_positions, self.ti_pos_backup, N)
                break

        n_newton = newton_it + 1

        # Count inversions before collision
        # Phase 2: detect collisions on deformed mesh, then Newton with penalty
        self.positions = self.ti_positions.to_numpy()
        has_mm_coll = (muscle_contact
                       and hasattr(self, '_muscle_surface_faces')
                       and bool(self._muscle_surface_faces))
        if self._merged_bone_mesh is not None:
            bi, bt = self._compute_collision_targets(margin, verbose=verbose)
            if len(bi) > 0:
                coll_penalty_idx = np.concatenate([coll_penalty_idx, bi])
                coll_penalty_tgt = np.concatenate([coll_penalty_tgt, bt])
        if has_mm_coll:
            self.positions = self.ti_positions.to_numpy()
            mi, mt = self._compute_muscle_collision_targets(margin=margin, verbose=verbose)
            if len(mi) > 0:
                coll_penalty_idx = np.concatenate([coll_penalty_idx, mi])
                coll_penalty_tgt = np.concatenate([coll_penalty_tgt, mt])
            ci, ct = self._compute_sliding_cohesion_targets(
                max_gap=0.0015, verbose=verbose)
            if len(ci) > 0:
                coll_penalty_idx = np.concatenate([coll_penalty_idx, ci])
                coll_penalty_tgt = np.concatenate([coll_penalty_tgt, ct])

        n_penalty = len(coll_penalty_idx)
        if n_penalty > 0:
            if not self._compliant_attachments:
                collision_kappa = effective_lam  # strong: match volume stiffness
            if n_penalty > self._max_coll:
                self._max_coll = max(n_penalty, 1000)
                self.ti_coll_idx = ti.field(dtype=ti.i32, shape=self._max_coll)
                self.ti_coll_targets = ti.Vector.field(3, dtype=ti.f64, shape=self._max_coll)
            ip = np.zeros(self._max_coll, dtype=np.int32)
            tp = np.zeros((self._max_coll, 3), dtype=np.float64)
            ip[:n_penalty] = coll_penalty_idx
            tp[:n_penalty] = coll_penalty_tgt
            self.ti_coll_idx.from_numpy(ip)
            self.ti_coll_targets.from_numpy(tp)

            if verbose:
                print(f"    Phase 2: {n_penalty} collision penalty, kappa={collision_kappa:.0f}")

            # Re-run Newton with penalty for 5 more iterations
            E_prev = _pn_compute_energy(
                self.ti_positions, self.ti_invm,
                self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
                effective_mu, effective_lam, M)
            pos_np = self.ti_positions.to_numpy()
            for pi in range(n_penalty):
                vi = coll_penalty_idx[pi]
                d = pos_np[vi] - coll_penalty_tgt[pi]
                E_prev += 0.5 * collision_kappa * np.dot(d, d)

            for newton_it in range(5):
                _pn_zero(self.ti_gradient, N)
                _pn_compute_gradient(
                    self.ti_positions, self.ti_gradient, self.ti_invm,
                    self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
                    effective_mu, effective_lam, M)
                # Add penalty gradient
                pos_np = self.ti_positions.to_numpy()
                penalty_grad = np.zeros((N, 3), dtype=np.float64)
                for pi in range(n_penalty):
                    vi = coll_penalty_idx[pi]
                    penalty_grad[vi] += collision_kappa * (pos_np[vi] - coll_penalty_tgt[pi])
                grad_np = self.ti_gradient.to_numpy()
                grad_np += penalty_grad
                self.ti_gradient.from_numpy(grad_np)
                _pn_zero_fixed(self.ti_gradient, self.ti_invm, N)

                # Steepest descent with line search (fast for penalty-dominated)
                _pn_zero(self.ti_direction, N)
                _pn_copy(self.ti_direction, self.ti_gradient, N)
                _pn_scale(self.ti_direction, -1.0, self.ti_invm, N)
                _pn_zero_fixed(self.ti_direction, self.ti_invm, N)

                alpha_safe = _pn_max_step_size(
                    self.ti_positions, self.ti_direction,
                    self.ti_tets, self.ti_Bm_inv, M)
                alpha_safe = min(alpha_safe, 1.0)

                dir_deriv = _pn_dot(self.ti_gradient, self.ti_direction, self.ti_invm, N)
                alpha = alpha_safe
                _pn_copy(self.ti_pos_backup, self.ti_positions, N)
                for ls_it in range(15):
                    _pn_copy(self.ti_positions, self.ti_pos_backup, N)
                    _pn_axpy(self.ti_positions, alpha, self.ti_direction, self.ti_invm, N)
                    _snap_fixed(self.ti_positions, self.ti_targets, self.ti_invm, N)
                    E_new = _pn_compute_energy(
                        self.ti_positions, self.ti_invm,
                        self.ti_tets, self.ti_Bm_inv, self.ti_rest_volume,
                        effective_mu, effective_lam, M)
                    pos_ls = self.ti_positions.to_numpy()
                    for ppi in range(n_penalty):
                        vi = coll_penalty_idx[ppi]
                        dd = pos_ls[vi] - coll_penalty_tgt[ppi]
                        E_new += 0.5 * collision_kappa * np.dot(dd, dd)
                    if E_new <= E_prev + 1e-4 * alpha * dir_deriv:
                        break
                    alpha *= 0.5
                E_prev = E_new

        self.positions = self.ti_positions.to_numpy()

        free = self.free_indices
        diff = self.positions[free] - initial_pos[free]
        residual = float(np.sqrt(np.sum(diff ** 2)))

        if verbose:
            X = self.positions
            T = self.tetrahedra
            x3 = X[T[:, 3]]
            Ds = np.stack([X[T[:, 0]] - x3, X[T[:, 1]] - x3, X[T[:, 2]] - x3], axis=-1)
            Js = np.linalg.det(Ds @ self.Bm_inv)
            n_inv = int(np.sum(Js <= 0))
            # Measure attachment error: how far are fixed verts from targets
            if self._compliant_attachments:
                fixed_pos = X[self._attachment_indices]
                attach_err = np.linalg.norm(
                    fixed_pos - self._attachment_targets, axis=1)
            else:
                fixed_pos = X[self.fixed_indices]
                attach_err = np.linalg.norm(
                    fixed_pos - self.fixed_targets, axis=1)
            print(f"    PN final: newton={n_newton} ||dx||={residual:.4e} "
                  f"inv={n_inv}/{M} penalty={n_penalty} "
                  f"J=[{np.min(Js):.3f},{np.max(Js):.3f}]")
            if len(attach_err):
                print(f"    attachment error: max={attach_err.max()*1000:.3f}mm "
                      f"mean={attach_err.mean()*1000:.3f}mm "
                      f"({int(np.sum(attach_err > 0.001))} verts >1mm)")

        self._has_previous_solution = True
        return n_newton, residual

    def get_muscle_positions(self, name):
        vs, ve, _, _ = self._muscle_ranges[name]
        return self.positions[vs:ve].copy()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _update_muscles_from_skeleton(v, active_muscles, skel, skeleton_meshes):
    """Update muscle LBS positions and fixed targets from current skeleton pose."""
    for name, mobj in active_muscles.items():
        if hasattr(mobj, '_update_tet_positions_from_skeleton') and skel is not None:
            mobj._update_tet_positions_from_skeleton(skel)
        if hasattr(mobj, '_update_fixed_targets_from_skeleton') and skel is not None:
            mobj._update_fixed_targets_from_skeleton(skeleton_meshes, skel)


def _collect_muscles_update(active_muscles):
    """Collect current positions and fixed targets into solver update dict."""
    muscles_update = {}
    for name, mobj in active_muscles.items():
        sb = mobj.soft_body
        muscles_update[name] = {
            'positions': sb.positions,
            'fixed_targets': sb.fixed_targets,
        }
    return muscles_update


def _write_back_positions(solver, active_muscles):
    """Write solver results back to muscle objects."""
    for name, mobj in active_muscles.items():
        sb = mobj.soft_body
        new_pos = solver.get_muscle_positions(name)
        sb.positions[:] = new_pos
        if hasattr(mobj, 'tet_vertices'):
            mobj.tet_vertices[:] = new_pos.astype(mobj.tet_vertices.dtype)


def _untangle_lbs_positions(rest, tets, lbs, fixed_indices,
                            max_iterations=800, j_floor=0.08,
                            j_ceiling=5.0, plane_indices=None,
                            plane_targets=None, plane_normals=None,
                            plane_weight=20.0, report_residual=True,
                            max_projection_sweeps=4000):
    """Untangle an LBS tet pose while keeping attachment vertices exact.

    This is deliberately a positional preconditioner, not a material model.
    A squared signed-Jacobian hinge removes inversions; a weak LBS tether
    chooses the closest solution among the many positive-volume embeddings.
    """
    from scipy.optimize import minimize

    rest = np.asarray(rest, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int32)
    target = np.asarray(lbs, dtype=np.float64)
    fixed_mask = np.zeros(len(rest), dtype=bool)
    fixed_mask[np.asarray(fixed_indices, dtype=np.int32)] = True
    free = np.where(~fixed_mask)[0]
    free_slot = np.full(len(rest), -1, dtype=np.int32)
    free_slot[free] = np.arange(len(free), dtype=np.int32)

    if plane_indices is None:
        plane_indices = np.zeros(0, dtype=np.int32)
        plane_targets = np.zeros((0, 3), dtype=np.float64)
        plane_normals = np.zeros((0, 3), dtype=np.float64)
    else:
        plane_indices = np.asarray(plane_indices, dtype=np.int32)
        plane_targets = np.asarray(plane_targets, dtype=np.float64)
        plane_normals = np.asarray(plane_normals, dtype=np.float64)
        valid_plane = ~fixed_mask[plane_indices]
        plane_indices = plane_indices[valid_plane]
        plane_targets = plane_targets[valid_plane]
        plane_normals = plane_normals[valid_plane]
        plane_normals /= np.maximum(
            np.linalg.norm(plane_normals, axis=1, keepdims=True), 1e-12)

    r = rest[tets]
    Bm = np.stack([r[:, 0] - r[:, 3], r[:, 1] - r[:, 3],
                   r[:, 2] - r[:, 3]], axis=-1)
    det_rest = np.linalg.det(Bm)
    edge_lengths = np.stack([
        np.linalg.norm(r[:, i] - r[:, j], axis=1)
        for i, j in ((0, 1), (0, 2), (0, 3),
                     (1, 2), (1, 3), (2, 3))], axis=1)
    quality = (np.abs(det_rest) / 6.0) / np.maximum(
        np.max(edge_lengths, axis=1) ** 3, 1e-30)
    keep = quality > 1e-5
    tets = tets[keep]
    det_rest = det_rest[keep]
    position_scale = max(float(np.median(edge_lengths[keep])), 1e-4)

    base = target.copy()
    x0 = base[free].ravel()

    def evaluate(flat, floor):
        x = base.copy()
        x[free] = flat.reshape(-1, 3)
        q = x[tets]
        a = q[:, 0] - q[:, 3]
        b = q[:, 1] - q[:, 3]
        c = q[:, 2] - q[:, 3]
        J = np.einsum('ij,ij->i', a, np.cross(b, c)) / det_rest
        low_violation = np.maximum(floor - J, 0.0)
        high_violation = np.maximum(J - j_ceiling, 0.0)
        active = (low_violation > 0.0) | (high_violation > 0.0)
        # Mean normalization makes settings independent of tet resolution.
        energy_j = 0.5 * np.mean(low_violation * low_violation
                                 + high_violation * high_violation)
        displacement = x[free] - target[free]
        tether_weight = 2e-5
        energy_p = (0.5 * tether_weight
                    * np.mean(displacement * displacement)
                    / (position_scale * position_scale))

        if len(plane_indices):
            plane_residual = np.einsum(
                'ij,ij->i', x[plane_indices] - plane_targets,
                plane_normals)
            energy_plane = (0.5 * plane_weight
                            * np.mean(plane_residual * plane_residual)
                            / (position_scale * position_scale))
        else:
            plane_residual = np.zeros(0, dtype=np.float64)
            energy_plane = 0.0

        grad = np.zeros_like(x)
        if np.any(active):
            ta = a[active]; tb = b[active]; tc = c[active]
            denom = det_rest[active, None]
            ga = np.cross(tb, tc) / denom
            gb = np.cross(tc, ta) / denom
            gc = np.cross(ta, tb) / denom
            coeff = ((-low_violation[active] + high_violation[active])
                     / len(J))[:, None]
            ids = tets[active]
            np.add.at(grad, ids[:, 0], coeff * ga)
            np.add.at(grad, ids[:, 1], coeff * gb)
            np.add.at(grad, ids[:, 2], coeff * gc)
            np.add.at(grad, ids[:, 3], -coeff * (ga + gb + gc))
        grad[free] += (tether_weight * displacement
                       / (len(free) * 3.0 * position_scale * position_scale))
        if len(plane_indices):
            plane_grad = ((plane_weight / len(plane_indices))
                          * plane_residual[:, None] * plane_normals
                          / (position_scale * position_scale))
            np.add.at(grad, plane_indices, plane_grad)
        return energy_j + energy_p + energy_plane, grad[free].ravel()

    current = x0
    # Staging avoids asking a deeply inverted element to cross directly to the
    # final safety margin in one non-convex optimization.
    for floor in (-0.25, 0.0, 0.02, float(j_floor)):
        result = minimize(
            lambda z: evaluate(z, floor), current, jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iterations, 'ftol': 1e-14,
                     'gtol': 1e-9, 'maxls': 40})
        current = result.x

    result_x = base.copy()
    result_x[free] = current.reshape(-1, 3)
    # L-BFGS can settle at a compromise where two adjacent inverted tets trade
    # equal-and-opposite gradients. Finish with local nonlinear Gauss-Seidel
    # projections, always moving only free vertices of the offending tet.
    for _sweep in range(max_projection_sweeps):
        q = result_x[tets]
        a = q[:, 0] - q[:, 3]
        b = q[:, 1] - q[:, 3]
        c = q[:, 2] - q[:, 3]
        J_now = np.einsum('ij,ij->i', a, np.cross(b, c)) / det_rest
        bad = np.where((J_now < j_floor) | (J_now > j_ceiling))[0]
        if not len(bad):
            break
        changed = False
        for ti_idx in bad[np.argsort(J_now[bad])]:
            ids = tets[ti_idx]
            pts = result_x[ids]
            aa = pts[0] - pts[3]
            bb = pts[1] - pts[3]
            cc = pts[2] - pts[3]
            grads = np.stack((np.cross(bb, cc),
                              np.cross(cc, aa),
                              np.cross(aa, bb))) / det_rest[ti_idx]
            grads = np.vstack((grads, -np.sum(grads, axis=0)))
            movable = ~fixed_mask[ids]
            denom = float(np.sum(grads[movable] ** 2))
            if denom <= 1e-20:
                continue
            target_j = (j_floor if J_now[ti_idx] < j_floor
                        else j_ceiling)
            correction = ((target_j - J_now[ti_idx]) / denom
                          * grads[movable])
            # Avoid one ill-conditioned tet throwing a vertex across the body.
            norms = np.linalg.norm(correction, axis=1)
            scale = min(1.0, 0.001 / max(float(norms.max()), 1e-12))
            result_x[ids[movable]] += 0.8 * scale * correction
            changed = True
        if not changed:
            break
    q = result_x[tets]
    J = np.einsum('ij,ij->i', q[:, 0] - q[:, 3],
                  np.cross(q[:, 1] - q[:, 3],
                           q[:, 2] - q[:, 3])) / det_rest
    residual = np.where((J <= 0.05) | (J > j_ceiling * 1.01))[0]
    if len(residual) and report_residual:
        details = [(int(i), float(J[i]),
                    int(np.sum(fixed_mask[tets[i]])), tets[i].tolist())
                   for i in residual[:12]]
        print(f"      untangle residual (tet, J, fixed-count, verts): {details}")
    return result_x, J


def run_all_fem_sim(v, max_iterations=100, tolerance=1e-4, verbose=True):
    solve_start_time = time.time()

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

    use_vbd = getattr(v, 'use_vbd_sim', False)
    use_pn = getattr(v, 'use_pn_sim', False)
    solver = getattr(v, '_fem_unified_solver', None)
    # Rebuild solver if muscles changed or solver type changed
    def _solver_type_match(s):
        if use_pn:
            return isinstance(s, ProjectedNewtonSolver)
        elif use_vbd:
            return isinstance(s, VBDSolver)
        else:
            return isinstance(s, UnifiedFEMSolver)
    wrong_type = solver is not None and not _solver_type_match(solver)
    needs_build = (solver is None or wrong_type
                   or set(solver._muscle_ranges.keys()) != set(active_muscles.keys()))
    if needs_build:
        if use_pn:
            solver = ProjectedNewtonSolver()
        elif use_vbd:
            solver = VBDSolver()
        else:
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
        if hasattr(v, 'inter_muscle_constraints') and v.inter_muscle_constraints:
            solver.set_inter_muscle_constraints(v.inter_muscle_constraints)
        v._fem_unified_solver = solver

    # Projected Newton must begin from an orientation-preserving state. Move
    # bones and hard attachment targets from rest to the requested pose in
    # small increments, carrying each valid equilibrium into the next step.
    # The old direct LBS jump started frame 0 with hundreds of inverted tets,
    # which an inversion-safe line search cannot subsequently repair.
    if use_pn and needs_build and skel is not None:
        pn_material_only = (
            os.environ.get('MUSCLE_PN_MATERIAL_ONLY', '0') == '1')
        target_pose = skel.getPositions().copy()
        rest_pose = np.zeros_like(target_pose)
        # Direct target-pose initialization for large BVH poses. Use the
        # project's ordinary LBS positions exactly as shown by the viewer,
        # then overwrite every attachment vertex with its exact bone target.
        if os.environ.get('MUSCLE_PN_DIRECT_INIT', '0') == '1':
            for direct_name, mobj in active_muscles.items():
                mobj._update_tet_positions_from_skeleton(skel)
                mobj._update_fixed_targets_from_skeleton(
                    getattr(v, 'zygote_skeleton_meshes', None), skel)
                sb = mobj.soft_body
                fixed = np.asarray(sb.fixed_indices, dtype=np.int32)
                direct = np.asarray(sb.positions, dtype=np.float64).copy()
                direct[fixed] = np.asarray(sb.fixed_targets, dtype=np.float64)
                vs, ve, _, _ = solver._muscle_ranges[direct_name]
                solver.positions[vs:ve] = direct
            updates = {
                name: {'positions': solver.get_muscle_positions(name),
                       'fixed_targets': mobj.soft_body.fixed_targets}
                for name, mobj in active_muscles.items()
            }
            solver.update_targets_and_positions(updates)
            solver.positions[solver.fixed_indices] = solver.fixed_targets
            tt = solver.tetrahedra
            x3 = solver.positions[tt[:, 3]]
            Ds = np.stack([solver.positions[tt[:, 0]] - x3,
                           solver.positions[tt[:, 1]] - x3,
                           solver.positions[tt[:, 2]] - x3], axis=-1)
            direct_J = np.linalg.det(Ds @ solver.Bm_inv)
            print(f"  PN direct LBS init: J=[{direct_J.min():.3e},"
                  f"{direct_J.max():.3e}], inv={np.sum(direct_J <= 0)}")
            if float(direct_J.min()) <= 0.05:
                print("  Untangling LBS free vertices with signed-volume "
                      "optimization (attachments remain exact)...")
                for name, (vs, ve, _, _) in solver._muscle_ranges.items():
                    mobj = active_muscles[name]
                    sb = mobj.soft_body
                    before = solver.positions[vs:ve].copy()
                    repaired, local_J = _untangle_lbs_positions(
                        sb.rest_positions, sb.tetrahedra, before,
                        sb.fixed_indices, max_iterations=800, j_floor=0.08)
                    retry = 0
                    while np.any(local_J <= 0.05) and retry < 3:
                        repaired, local_J = _untangle_lbs_positions(
                            sb.rest_positions, sb.tetrahedra, repaired,
                            sb.fixed_indices, max_iterations=2000,
                            j_floor=0.12 + 0.04 * retry)
                        retry += 1
                    solver.positions[vs:ve] = repaired
                    if verbose:
                        print(f"    {name}: minJ={local_J.min():.3e}, "
                              f"inv={int(np.sum(local_J <= 0))}, "
                              f"max move={np.linalg.norm(repaired-before, axis=1).max()*1000:.2f}mm")
                solver.positions[solver.fixed_indices] = solver.fixed_targets
                x3 = solver.positions[tt[:, 3]]
                Ds = np.stack([solver.positions[tt[:, 0]] - x3,
                               solver.positions[tt[:, 1]] - x3,
                               solver.positions[tt[:, 2]] - x3], axis=-1)
                direct_J = np.linalg.det(Ds @ solver.Bm_inv)
                print(f"  Signed-volume untangled init: J=[{direct_J.min():.3e},"
                      f"{direct_J.max():.3e}], "
                      f"inv={np.sum(direct_J <= 0)}")
                if float(direct_J.min()) <= 0.05:
                    raise RuntimeError(
                        "LBS XPBD untangler did not reach positive volume: "
                        f"minJ={direct_J.min():.3e}, "
                        f"inversions={int(np.sum(direct_J <= 0))}")
            solver._has_previous_solution = True
            resume_dir = os.environ.get('MUSCLE_PN_RESUME_CACHE', '')
            resumed_interface = False
            if resume_dir:
                loaded = 0
                for resume_name, (vs, ve, _, _) in (
                        solver._muscle_ranges.items()):
                    resume_path = os.path.join(
                        resume_dir, f'{resume_name}_chunk_0000.npz')
                    if not os.path.isfile(resume_path):
                        continue
                    with np.load(resume_path) as resume_data:
                        cached = np.asarray(
                            resume_data['positions'][0], dtype=np.float64)
                    if cached.shape != (ve - vs, 3):
                        raise RuntimeError(
                            f"resume cache shape mismatch for {resume_name}: "
                            f"{cached.shape} != {(ve - vs, 3)}")
                    solver.positions[vs:ve] = cached
                    loaded += 1
                if loaded != len(solver._muscle_ranges):
                    raise RuntimeError(
                        f"resume cache incomplete: {loaded}/"
                        f"{len(solver._muscle_ranges)} muscles")
                solver.positions[solver.fixed_indices] = solver.fixed_targets
                solver.ti_positions.from_numpy(np.asarray(
                    solver.positions, dtype=np.float64))
                resumed_interface = True
                print(f"  PN resumed interface state from {resume_dir}")
            bone_meshes = _build_bone_trimeshes(v, active_muscles, skel,
                                                verbose=False)
            interface_passes = int(os.environ.get(
                'MUSCLE_PN_INTERFACE_PASSES', '12'))
            exclusion_only = (os.environ.get(
                'MUSCLE_PN_EXCLUSION_ONLY', '0') == '1')
            polish_passes = (0 if exclusion_only else int(os.environ.get(
                'MUSCLE_PN_EXCLUSION_POLISH_PASSES', '4')))
            for interface_pass in range(interface_passes + polish_passes):
                # Establish the unconstrained elastic equilibrium once. The
                # active-set iterations below then impose current normal
                # contact positions as temporary Dirichlet data and solve the
                # muscle interior around them. Correspondences are rebuilt on
                # every pass, so this is sliding contact rather than a weld.
                if interface_pass == 0 and not resumed_interface:
                    solver.solve(
                        mu=mu, lam=lam, vol_penalty=vol_penalty,
                        kappa=kappa,
                        max_lbfgs_iters=max_iterations,
                        bone_meshes=bone_meshes, muscle_contact=True,
                        verbose=verbose)
                    solver.positions = solver.ti_positions.to_numpy()
                bi, bt = solver._compute_collision_targets(
                    margin=getattr(v, 'unified_bone_contact_margin', 0.001),
                    verbose=False)
                mi, mt = solver._compute_muscle_collision_targets(
                    margin=getattr(v, 'inter_muscle_contact_margin', 0.001),
                    verbose=False)
                ci, ct = solver._compute_sliding_cohesion_targets(
                    max_gap=0.0015, verbose=False)
                if exclusion_only or interface_pass >= interface_passes:
                    ci = np.zeros(0, dtype=np.int32)
                    ct = np.zeros((0, 3), dtype=np.float64)
                # Complementarity/priority: never average an exclusion target
                # with a cohesion target on the same material point. Bone
                # exclusion wins over muscle exclusion; either exclusion wins
                # over gap-closing cohesion.
                if len(bi) and len(mi):
                    keep_m = ~np.isin(mi, np.unique(bi))
                    mi, mt = mi[keep_m], mt[keep_m]
                exclusion_idx = np.unique(np.concatenate((bi, mi)))
                if len(exclusion_idx) and len(ci):
                    keep_c = ~np.isin(ci, exclusion_idx)
                    ci, ct = ci[keep_c], ct[keep_c]
                active = len(bi) + len(mi) + len(ci)
                print(f"    interface pass {interface_pass}: bone={len(bi)} "
                      f"muscle={len(mi)} separated={len(ci)}")
                if active == 0:
                    break
                # Average only compatible constraints incident on a vertex.
                contact_idx = np.concatenate((bi, mi, ci))
                contact_target = np.concatenate((bt, mt, ct), axis=0)
                unique, inverse = np.unique(contact_idx,
                                            return_inverse=True)
                delta_sum = np.zeros((len(unique), 3), dtype=np.float64)
                count = np.zeros(len(unique), dtype=np.float64)
                np.add.at(delta_sum, inverse,
                          contact_target - solver.positions[contact_idx])
                np.add.at(count, inverse, 1.0)
                desired = solver.positions[unique] + delta_sum / count[:, None]
                movable = ~solver.fixed_mask[unique]
                unique, desired = unique[movable], desired[movable]

                moved_muscles = 0
                for name, (vs, ve, _, _) in solver._muscle_ranges.items():
                    selected = (unique >= vs) & (unique < ve)
                    if not np.any(selected):
                        continue
                    sb = active_muscles[name].soft_body
                    local_ids = unique[selected] - vs
                    local_desired = desired[selected]
                    base = solver.positions[vs:ve].copy()
                    delta = local_desired - base[local_ids]
                    # Gap closure is incremental; exclusion targets are already
                    # confined to a narrow collision band. A common 4 mm cap
                    # prevents a separated patch from jumping through its
                    # counterpart in one active-set update.
                    delta_norm = np.linalg.norm(delta, axis=1)
                    delta *= np.minimum(
                        1.0, 0.004 / np.maximum(delta_norm, 1e-12))[:, None]
                    plane_normals = delta / np.maximum(
                        np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
                    accepted = False
                    for target_scale in (1.0, 0.5, 0.25, 0.125):
                        repaired, local_J = _untangle_lbs_positions(
                            sb.rest_positions, sb.tetrahedra, base,
                            sb.fixed_indices, max_iterations=500,
                            j_floor=0.02, j_ceiling=5.0,
                            plane_indices=local_ids,
                            plane_targets=(base[local_ids]
                                           + target_scale * delta),
                            plane_normals=plane_normals,
                            plane_weight=30.0,
                            report_residual=False)
                        if (len(local_J)
                                and float(local_J.min()) > 1e-6
                                and float(local_J.max()) <= 5.5):
                            solver.positions[vs:ve] = repaired
                            moved_muscles += 1
                            accepted = True
                            break
                    if not accepted and verbose:
                        print(f"      hard contact rejected for {name}")
                if moved_muscles == 0:
                    print("    hard interface solve made no feasible progress")
                    break
                solver.positions[solver.fixed_indices] = solver.fixed_targets
                solver.ti_positions.from_numpy(np.asarray(
                    solver.positions, dtype=np.float64))
            # Validate the state that will actually be written, including the
            # last accepted projection (whose counts are otherwise not shown
            # until another active-set iteration).
            final_bi, _ = solver._compute_collision_targets(
                margin=getattr(v, 'unified_bone_contact_margin', 0.001),
                verbose=False)
            final_mi, _ = solver._compute_muscle_collision_targets(
                margin=getattr(v, 'inter_muscle_contact_margin', 0.001),
                verbose=False)
            final_ci, _ = solver._compute_sliding_cohesion_targets(
                max_gap=0.0015, verbose=False)
            final_x3 = solver.positions[solver.tetrahedra[:, 3]]
            final_Ds = np.stack([
                solver.positions[solver.tetrahedra[:, 0]] - final_x3,
                solver.positions[solver.tetrahedra[:, 1]] - final_x3,
                solver.positions[solver.tetrahedra[:, 2]] - final_x3], axis=-1)
            final_J = np.linalg.det(final_Ds @ solver.Bm_inv)
            print(f"    interface final: bone={len(final_bi)} "
                  f"muscle={len(final_mi)} separated={len(final_ci)} "
                  f"J=[{final_J.min():.3f},{final_J.max():.3f}]")
            if float(final_J.min()) < 1e-6 or float(final_J.max()) > 5.5:
                raise RuntimeError("PN interface bake failed Jacobian bounds")
            _write_back_positions(solver, active_muscles)
            return
        n_steps = max(int(getattr(v, 'fem_load_steps', 10)), 2)
        step_iters = max(3, int(np.ceil(max_iterations / n_steps)))
        print(f"  PN continuation: {n_steps} nominal pose steps, "
              f"{step_iters} Newton iterations/accepted step (adaptive J guard)")
        alpha = 0.0
        accepted_steps = 0
        nominal_delta = 1.0 / n_steps
        # Track an absolute skeleton-driven guide for each muscle. Between
        # continuation poses, transport the current equilibrium by the guide
        # delta before moving the hard caps. This lets the terminal rings move
        # with their attachments instead of asking a single thin tet layer to
        # absorb the complete boundary increment.
        from tools.bake_emu import compute_rigid_blend_positions

        rigid_guide_data = {}
        attachment_blend = {}
        for name, mobj in active_muscles.items():
            bindings = []
            blend_coordinate = []
            initial = getattr(mobj, 'tet_initial_bone_transforms', {})
            for binding in getattr(mobj, 'tet_skeleton_bindings', []):
                if binding is None:
                    bindings.append((np.zeros(3), []))
                    blend_coordinate.append(0.0)
                    continue
                origin, insertion, weight, rest_vertex = binding
                weighted_bones = []
                if origin in initial:
                    R0, rest_translation = initial[origin]
                    weighted_bones.append((origin, 1.0 - weight, R0,
                                           rest_translation))
                if insertion in initial:
                    R0, rest_translation = initial[insertion]
                    weighted_bones.append((insertion, weight, R0,
                                           rest_translation))
                bindings.append((np.asarray(rest_vertex, dtype=np.float64),
                                 weighted_bones))
                blend_coordinate.append(weight)
            rigid_guide_data[name] = (
                bindings, np.asarray(blend_coordinate, dtype=np.float64))
            rest = np.asarray(mobj.soft_body.rest_positions, dtype=np.float64)
            fixed_local = np.asarray(mobj.soft_body.fixed_indices,
                                     dtype=np.int64)
            from scipy.spatial import cKDTree
            distance, nearest = cKDTree(rest[fixed_local]).query(rest, k=1)
            # Four centimetres covers several coarse contour bands while
            # leaving the belly governed by the rotation-preserving guide.
            blend = np.clip(1.0 - distance / 0.04, 0.0, 1.0)
            blend = blend * blend * (3.0 - 2.0 * blend)
            attachment_blend[name] = (fixed_local, nearest, blend)

        def rigid_guides():
            return {
                name: compute_rigid_blend_positions(bindings, skel, coordinate)
                for name, (bindings, coordinate) in rigid_guide_data.items()
            }

        skel.setPositions(rest_pose)
        guide_previous = rigid_guides()
        # Put both guide and hard caps in the same rest-pose coordinate state.
        # Without this, the first delta mixes a rest guide with target-pose
        # attachment residuals and can invert a cap even as alpha approaches 0.
        for guide_name, mobj in active_muscles.items():
            if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
                mobj._update_fixed_targets_from_skeleton(
                    getattr(v, 'zygote_skeleton_meshes', None), skel)
            local_fixed = np.asarray(mobj.soft_body.fixed_indices,
                                     dtype=np.int64)
            fixed_target = np.asarray(mobj.soft_body.fixed_targets,
                                      dtype=np.float64)
            fixed_residual = fixed_target - guide_previous[guide_name][local_fixed]
            _, nearest, blend = attachment_blend[guide_name]
            guide_previous[guide_name] += blend[:, None] * fixed_residual[nearest]
            guide_previous[guide_name][local_fixed] = fixed_target
            vs, ve, _, _ = solver._muscle_ranges[guide_name]
            solver.positions[vs:ve] = guide_previous[guide_name]
        while alpha < 1.0 - 1e-12:
            delta = min(nominal_delta, 1.0 - alpha)
            accepted = False
            while not accepted:
                trial_alpha = alpha + delta
                skel.setPositions((1.0 - trial_alpha) * rest_pose
                                  + trial_alpha * target_pose)
                guide_trial = rigid_guides()
                for guide_trial_name, mobj in active_muscles.items():
                    if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
                        mobj._update_fixed_targets_from_skeleton(
                            getattr(v, 'zygote_skeleton_meshes', None), skel)
                    local_fixed = np.asarray(
                        mobj.soft_body.fixed_indices, dtype=np.int64)
                    fixed_target = np.asarray(
                        mobj.soft_body.fixed_targets, dtype=np.float64)
                    fixed_residual = (fixed_target
                                      - guide_trial[guide_trial_name][local_fixed])
                    _, nearest, blend = attachment_blend[guide_trial_name]
                    guide_trial[guide_trial_name] += (
                        blend[:, None] * fixed_residual[nearest])
                    guide_trial[guide_trial_name][local_fixed] = fixed_target
                updates = {}
                for name, mobj in active_muscles.items():
                    updates[name] = {
                        'positions': solver.get_muscle_positions(name),
                        'fixed_targets': mobj.soft_body.fixed_targets,
                    }
                solver.update_targets_and_positions(updates)

                # Test the boundary move before Newton. Positive-J Newton can
                # preserve orientation, but cannot repair an already inverted
                # starting state.
                candidate = solver.positions.copy()
                for name, (vs, ve, _, _) in solver._muscle_ranges.items():
                    if getattr(solver, '_compliant_attachments', False):
                        # Remove the common rigid translation from the load.
                        # This is orientation preserving and leaves only the
                        # relative origin/insertion motion for the elastic
                        # penalty solve.  Applying each nearest-cap delta here
                        # would recreate a discontinuous hard boundary.
                        local_fixed, _, _ = attachment_blend[name]
                        fixed_delta = (
                            guide_trial[name][local_fixed]
                            - guide_previous[name][local_fixed])
                        candidate[vs:ve] += np.mean(
                            fixed_delta, axis=0, keepdims=True)
                        continue
                    # Transport only the material near moving attachment caps.
                    # A full LBS/rigid-guide delta can fold a long wrapping
                    # muscle's belly (Rectus Femoris was the reproducible
                    # failure) before FEM gets a chance to equilibrate it.
                    # The nearest-cap field moves several contour bands with
                    # the boundary while leaving the belly to the solve.
                    local_fixed, nearest, blend = attachment_blend[name]
                    fixed_delta = (guide_trial[name][local_fixed]
                                   - guide_previous[name][local_fixed])
                    candidate[vs:ve] += (
                        blend[:, None] * fixed_delta[nearest])
                candidate[solver.fixed_indices] = solver.fixed_targets
                tt = solver.tetrahedra
                x3 = candidate[tt[:, 3]]
                Ds = np.stack([candidate[tt[:, 0]] - x3,
                               candidate[tt[:, 1]] - x3,
                               candidate[tt[:, 2]] - x3], axis=-1)
                Js = np.linalg.det(Ds @ solver.Bm_inv)
                if float(np.min(Js)) > 0.05:
                    accepted = True
                    solver.positions[:] = candidate
                    guide_previous = guide_trial
                else:
                    if verbose and delta <= nominal_delta / 8.0:
                        bad_tet = int(np.argmin(Js))
                        bad_name = next(
                            (n for n, (_, _, ts, te) in
                             solver._muscle_ranges.items()
                             if ts <= bad_tet < te), 'unknown')
                        base_x3 = solver.positions[tt[:, 3]]
                        base_Ds = np.stack([
                            solver.positions[tt[:, 0]] - base_x3,
                            solver.positions[tt[:, 1]] - base_x3,
                            solver.positions[tt[:, 2]] - base_x3], axis=-1)
                        base_Js = np.linalg.det(base_Ds @ solver.Bm_inv)
                        print(f"    continuation reject: alpha={alpha:.6f} "
                              f"delta={delta:.3e} muscle={bad_name} "
                              f"tet={bad_tet} J={Js[bad_tet]:.3e} "
                              f"base_minJ={np.min(base_Js):.3e}")
                    delta *= 0.5
                    if delta < 1e-6:
                        raise RuntimeError(
                            f"PN continuation blocked at alpha={alpha:.6f}: "
                            "attachment motion cannot keep all tets positive")

            alpha += delta
            accepted_steps += 1
            at_final_pose = alpha >= 1.0 - 1e-12
            # Route wrapping muscles throughout continuation. Deferring bone
            # and muscle contact until alpha=1 lets hamstrings take a straight
            # folding shortcut and irreversibly flatten before the final pass.
            bone_step = (None if pn_material_only else
                         _build_bone_trimeshes(
                             v, active_muscles, skel, verbose=False))
            solver.solve(mu=mu, lam=lam, vol_penalty=vol_penalty,
                         kappa=kappa, max_lbfgs_iters=step_iters,
                         bone_meshes=bone_step,
                         muscle_contact=not pn_material_only,
                         verbose=verbose and (accepted_steps == 1
                                              or at_final_pose))
            if at_final_pose and not pn_material_only:
                # Contact targets depend on the deformed surface. Refresh the
                # active set until no free muscle surface vertex is inside a
                # bone; one frozen projection pass is not a collision solve.
                for contact_pass in range(12):
                    bone_idx, bone_targets = solver._compute_collision_targets(
                        margin=getattr(v, 'unified_bone_contact_margin', 0.001),
                        verbose=verbose)
                    muscle_idx, muscle_targets = (
                        solver._compute_muscle_collision_targets(
                            margin=getattr(
                                v, 'inter_muscle_contact_margin', 0.001),
                            verbose=False))
                    contact_idx = np.concatenate((bone_idx, muscle_idx))
                    contact_targets = np.concatenate(
                        (bone_targets, muscle_targets), axis=0)
                    if len(contact_idx) == 0:
                        break
                    base = solver.positions.copy()
                    # Multiple contacts on one vertex are combined rather
                    # than allowing array assignment order to choose one.
                    unique_idx, inverse = np.unique(contact_idx,
                                                    return_inverse=True)
                    summed = np.zeros((len(unique_idx), 3), dtype=np.float64)
                    count = np.zeros(len(unique_idx), dtype=np.float64)
                    np.add.at(summed, inverse,
                              contact_targets - base[contact_idx])
                    np.add.at(count, inverse, 1.0)
                    bone_idx = unique_idx
                    displacement = summed / count[:, None]
                    # Move a local material neighborhood with each contact
                    # vertex. Point-only projection shears the incident tets
                    # and can have no positive-J step even for a 1 mm escape.
                    correction = np.zeros_like(base)
                    used_contour_transport = False
                    for muscle_name, (vs, ve, _, _) in solver._muscle_ranges.items():
                        selected = np.where((bone_idx >= vs) & (bone_idx < ve))[0]
                        if not len(selected):
                            continue
                        levels = np.asarray(getattr(
                            active_muscles[muscle_name],
                            'vertex_contour_level', []), dtype=np.int32)
                        if levels.shape != (ve - vs,):
                            continue
                        local_contact = bone_idx[selected] - vs
                        for contact_level in np.unique(levels[local_contact]):
                            at_level = selected[levels[local_contact] == contact_level]
                            ring_shift = np.median(displacement[at_level], axis=0)
                            for level in range(int(contact_level) - 3,
                                               int(contact_level) + 4):
                                ring = np.where(levels == level)[0]
                                if not len(ring):
                                    continue
                                weight = 1.0 - abs(level - contact_level) / 4.0
                                correction[vs + ring] += weight * ring_shift
                            used_contour_transport = True
                    if not used_contour_transport:
                        sigma = 0.015
                        delta_all = (base[:, None, :]
                                     - base[bone_idx][None, :, :])
                        weights = np.exp(
                            -np.sum(delta_all * delta_all, axis=2)
                            / (2.0 * sigma * sigma))
                        correction = ((weights @ displacement)
                                      / np.maximum(
                                          weights.sum(axis=1, keepdims=True),
                                          1e-12))
                    correction[bone_idx] = displacement
                    correction[solver.fixed_indices] = 0.0
                    accepted_contact = False
                    scale = 1.0
                    while scale >= 1e-6:
                        candidate = base.copy()
                        candidate += scale * correction
                        candidate[solver.fixed_indices] = solver.fixed_targets
                        tt = solver.tetrahedra
                        x3 = candidate[tt[:, 3]]
                        Ds = np.stack([
                            candidate[tt[:, 0]] - x3,
                            candidate[tt[:, 1]] - x3,
                            candidate[tt[:, 2]] - x3], axis=-1)
                        Js = np.linalg.det(Ds @ solver.Bm_inv)
                        if float(np.min(Js)) > 1e-8:
                            solver.positions[:] = candidate
                            accepted_contact = True
                            if verbose:
                                print(f"      contact projection pass "
                                      f"{contact_pass}: {len(bone_idx)} verts, "
                                      f"scale={scale:.3e}, "
                                      f"max={np.max(np.linalg.norm(displacement, axis=1))*1000:.3f}mm, "
                                      f"minJ={np.min(Js):.3e}")
                            break
                        scale *= 0.5
                    if not accepted_contact:
                        current_x3 = base[solver.tetrahedra[:, 3]]
                        current_Ds = np.stack([
                            base[solver.tetrahedra[:, 0]] - current_x3,
                            base[solver.tetrahedra[:, 1]] - current_x3,
                            base[solver.tetrahedra[:, 2]] - current_x3], axis=-1)
                        current_min_j = float(np.min(
                            np.linalg.det(current_Ds @ solver.Bm_inv)))
                        raise RuntimeError(
                            "PN contact projection has no positive-volume step: "
                            f"minJ={current_min_j:.3e}, "
                            f"max displacement={np.max(np.linalg.norm(displacement, axis=1)):.3e}")
                else:
                    bone_idx, _ = solver._compute_collision_targets(
                        margin=getattr(v, 'unified_bone_contact_margin', 0.001),
                        verbose=False)
                    muscle_idx, _ = solver._compute_muscle_collision_targets(
                        margin=getattr(v, 'inter_muscle_contact_margin', 0.001),
                        verbose=False)
                    raise RuntimeError(
                        "PN contact failed validation: "
                        f"{len(bone_idx)} bone and {len(muscle_idx)} "
                        "muscle penetrations remain")
        skel.setPositions(target_pose)
        _write_back_positions(solver, active_muscles)
        total = time.time() - solve_start_time
        if verbose:
            print(f"  PN continuation complete: {accepted_steps} accepted "
                  f"steps, total={total:.2f}s")
        return

    # Update muscles from current skeleton pose
    skeleton_meshes = getattr(v, 'zygote_skeleton_meshes', None)
    _update_muscles_from_skeleton(v, active_muscles, skel, skeleton_meshes)
    muscles_update = _collect_muscles_update(active_muscles)
    solver.update_targets_and_positions(muscles_update)

    # Build bone collision meshes at current pose
    bone_meshes = _build_bone_trimeshes(v, active_muscles, skel, verbose=False)

    # For first frame: build a bone mesh callback that rebuilds bone collision
    # geometry at interpolated skeleton poses during internal load stepping.
    # This ensures muscles collide against correctly-posed bones at each step.
    n_load_steps = getattr(v, 'fem_load_steps', 10)
    bone_mesh_callback = None
    if not solver._has_previous_solution and skel is not None and n_load_steps > 1:
        target_pose = skel.getPositions().copy()
        rest_pose = np.zeros_like(target_pose)

        def bone_mesh_callback(alpha):
            interp_pose = (1.0 - alpha) * rest_pose + alpha * target_pose
            skel.setPositions(interp_pose)
            meshes = _build_bone_trimeshes(v, active_muscles, skel, verbose=False)
            # Restore target pose after building meshes
            if alpha >= 1.0:
                skel.setPositions(target_pose)
            return meshes

    # Single solve call — solver handles internal load stepping + collision
    fevals, residual = solver.solve(
        mu=mu, lam=lam, vol_penalty=vol_penalty, kappa=kappa,
        max_lbfgs_iters=max_iterations,
        bone_meshes=bone_meshes, verbose=verbose,
        bone_mesh_callback=bone_mesh_callback,
    )

    _write_back_positions(solver, active_muscles)

    total = time.time() - solve_start_time
    tag = "PN" if use_pn else ("VBD" if use_vbd else "XPBD")
    if verbose:
        print(f"  {tag}: {fevals} iters, ||dx||={residual:.4e}, total={total:.2f}s")
    else:
        print(f"  {tag}: {fevals}it ||dx||={residual:.2e} {total:.1f}s")


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
            # Preserve identity for muscle-specific broad phase filtering.
            # Returning anonymous meshes forced callers to use a large AABB,
            # which could select unrelated bones in strongly posed limbs.
            tm.metadata['bone_name'] = mesh_name
            tm.metadata['body_name'] = body_name
            bone_meshes.append(tm)
            n_matched += 1
        except Exception:
            continue
    if verbose:
        print(f"    Bone meshes: {n_matched}/{len(skeleton_meshes)} matched")
    return bone_meshes
