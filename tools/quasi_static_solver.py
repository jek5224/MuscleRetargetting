"""Quasi-static FEM solver with one-sided bone contact.

Energy: E(x) = E_elastic(x) + κ * E_contact(x)
  E_elastic: co-rotated linear FEM (volume-preserving via high Poisson ratio)
  E_contact: one-sided barrier on bone surface (only penalizes inside-bone)

Solver: L-BFGS (scipy) — no Hessian needed, naturally combines elastic + contact.

Attachment: hard Dirichlet BC (fixed vertices excluded from optimization).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Pre-computation (once per side)
# ---------------------------------------------------------------------------
def precompute_tet_data(vertices, tetrahedra):
    """Compute Dm_inv (rest shape inverse) and volumes for all tets.
    Returns: Dm_inv (n_tets, 3, 3), volumes (n_tets,)
    """
    v = vertices[tetrahedra]  # (n_tets, 4, 3)
    # Dm = [v1-v0, v2-v0, v3-v0]^T columns
    Dm = np.stack([v[:, 1] - v[:, 0],
                   v[:, 2] - v[:, 0],
                   v[:, 3] - v[:, 0]], axis=-1)  # (n_tets, 3, 3)
    volumes = np.abs(np.linalg.det(Dm)) / 6.0
    Dm_inv = np.linalg.inv(Dm)
    return Dm_inv, volumes


def lame_parameters(youngs_modulus, poisson_ratio):
    """Convert Young's modulus and Poisson ratio to Lamé parameters."""
    mu = youngs_modulus / (2 * (1 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    return mu, lam


# ---------------------------------------------------------------------------
# Co-rotated FEM energy and gradient (vectorized)
# ---------------------------------------------------------------------------
def _deformation_gradients(positions, tetrahedra, Dm_inv):
    """Compute deformation gradient F for all tets. Returns (n_tets, 3, 3)."""
    v = positions[tetrahedra]  # (n_tets, 4, 3)
    Ds = np.stack([v[:, 1] - v[:, 0],
                   v[:, 2] - v[:, 0],
                   v[:, 3] - v[:, 0]], axis=-1)  # (n_tets, 3, 3)
    F = np.einsum('nij,njk->nik', Ds, Dm_inv)  # (n_tets, 3, 3)
    return F


def _polar_decomposition_batch(F):
    """Batch polar decomposition F = R @ S. Returns R (n, 3, 3).
    Handles reflections (ensures det(R) > 0).
    """
    U, sigma, Vt = np.linalg.svd(F)
    # Check for reflections
    det_UVt = np.linalg.det(U) * np.linalg.det(Vt)
    # For negative determinant, flip smallest singular vector
    flip = det_UVt < 0
    if np.any(flip):
        U[flip, :, 2] *= -1
        sigma[flip, 2] *= -1
    R = np.einsum('nij,njk->nik', U, Vt)
    return R, sigma


def elastic_energy_and_gradient(positions, tetrahedra, Dm_inv, volumes, mu, lam):
    """Co-rotated linear FEM energy and gradient.

    Energy per tet: Ψ = μ Σᵢ(σᵢ - 1)² + λ/2 (Σᵢ(σᵢ - 1))²
    PK1 stress:     P = R @ (2μ(S - I) + λ·tr(S - I)·I)

    Returns: (total_energy, gradient array same shape as positions)
    """
    n_tets = len(tetrahedra)
    n_verts = len(positions)

    F = _deformation_gradients(positions, tetrahedra, Dm_inv)
    R, sigma = _polar_decomposition_batch(F)

    # S = R^T @ F (stretch tensor), but we have sigma from SVD
    # Energy: μ Σ(σ-1)² + λ/2 (Σ(σ-1))²
    s_minus_1 = sigma - 1.0  # (n_tets, 3)
    trace_sm1 = s_minus_1.sum(axis=1)  # (n_tets,)

    energy_per_tet = mu * np.sum(s_minus_1 ** 2, axis=1) + 0.5 * lam * trace_sm1 ** 2
    total_energy = np.sum(volumes * energy_per_tet)

    # PK1 stress: P = R @ diag(2μ(σ-1) + λ·tr(σ-1))
    # = R @ diag(stress_diag)
    stress_diag = 2 * mu * s_minus_1 + lam * trace_sm1[:, None]  # (n_tets, 3)

    # P = R @ diag(stress_diag) @ Vt  ... no wait.
    # From SVD: F = U @ diag(σ) @ Vt, R = U @ Vt
    # S = Vt^T @ diag(σ) @ Vt
    # P = dΨ/dF. For co-rotated:
    # P = U @ diag(2μ(σ-1) + λ·tr(σ-1)) @ Vt
    # Since R = U @ Vt, this is: P = R @ Vt^T @ diag(...) @ Vt
    # But simpler: P_ij = Σ_k U_ik * stress_diag_k * Vt_kj

    U, _, Vt = np.linalg.svd(F)
    det_UVt = np.linalg.det(U) * np.linalg.det(Vt)
    flip = det_UVt < 0
    if np.any(flip):
        U[flip, :, 2] *= -1

    # P = U @ diag(stress) @ Vt
    P = np.einsum('nij,nj,njk->nik', U, stress_diag, Vt)  # (n_tets, 3, 3)

    # Force on vertices: H = volume * P @ Dm_inv^T
    # H columns → forces on v1, v2, v3; force on v0 = -sum
    H = np.einsum('n,nij,nkj->nik', volumes, P, Dm_inv)  # (n_tets, 3, 3)
    # H[:, :, 0] = force on v1, etc.

    gradient = np.zeros_like(positions)
    tets = tetrahedra
    for d in range(3):  # for v1, v2, v3
        np.add.at(gradient, tets[:, d + 1], H[:, :, d])
    # v0 gets negative sum
    H_sum = H.sum(axis=2)  # (n_tets, 3)
    np.add.at(gradient, tets[:, 0], -H_sum)

    return total_energy, gradient


# ---------------------------------------------------------------------------
# One-sided bone contact barrier
# ---------------------------------------------------------------------------
def _log_barrier(d, dhat):
    """IPC log barrier: B(d) = -(d/dhat - 1)^2 * ln(d/dhat) for 0 < d < dhat.
    Returns energy (scalar per element).
    """
    t = d / dhat
    return -(t - 1) ** 2 * np.log(np.clip(t, 1e-10, None))


def _log_barrier_derivative(d, dhat):
    """dB/dd for log barrier."""
    t = d / dhat
    t_clip = np.clip(t, 1e-10, None)
    return (-(t - 1) * (2 * np.log(t_clip) + (t - 1) / t_clip)) / dhat


def contact_energy_and_gradient(positions, free_mask, bone_data, dhat, kappa):
    """One-sided barrier contact energy and gradient.

    bone_data: list of (bone_trimesh, bone_tree, body_name, muscle_info)
      muscle_info: list of (offset, n_verts, fixed_set, attach_bones)

    Only penalizes vertices INSIDE bone (signed distance < 0).
    Also penalizes vertices approaching from outside within dhat.

    Returns: (energy, gradient)
    """
    import trimesh as _trimesh

    total_energy = 0.0
    gradient = np.zeros_like(positions)

    for bone_mesh, bone_tree, body_name, muscle_infos in bone_data:
        for off, n_v, fixed_set, attach_bones in muscle_infos:
            if body_name in attach_bones:
                continue

            pos = positions[off:off + n_v]

            # KDTree pre-filter
            dists_kd, _ = bone_tree.query(pos)
            near_mask = dists_kd < dhat + 0.015
            if not np.any(near_mask):
                continue
            near_idx = np.where(near_mask)[0]
            near_pos = pos[near_idx]

            # Closest point + signed distance
            closest, _, face_ids = _trimesh.proximity.closest_point(
                bone_mesh, near_pos)
            normals = bone_mesh.face_normals[face_ids]
            to_vert = near_pos - closest
            signed_dist = np.sum(to_vert * normals, axis=1)

            # One-sided: only vertices with signed_dist < dhat
            # (inside bone = signed_dist < 0, approaching = 0 < signed_dist < dhat)
            active = signed_dist < dhat
            if not np.any(active):
                continue

            active_idx = near_idx[active]
            active_signed = signed_dist[active]
            active_normals = normals[active]
            active_closest = closest[active]

            # One-sided contact energy:
            # Inside (signed_dist < 0): quadratic penalty E = 0.5 * (depth + margin)^2
            # Approaching (0 < signed_dist < dhat): log barrier B(d, dhat)
            for i, vi in enumerate(active_idx):
                if int(vi) in fixed_set:
                    continue
                sd = active_signed[i]
                n_dir = active_normals[i]

                if sd < 0:
                    # Inside bone: quadratic penalty proportional to depth²
                    depth = -sd
                    margin = 0.003  # 3mm target margin
                    pen_dist = depth + margin
                    E_pen = 0.5 * kappa * pen_dist ** 2
                    total_energy += E_pen
                    # gradient: dE/dx = kappa * pen_dist * d(pen_dist)/dx
                    # d(depth)/dx = -d(signed_dist)/dx = -normal
                    # d(pen_dist)/dx = d(depth+margin)/dx = -normal
                    gradient[off + int(vi)] += kappa * pen_dist * (-n_dir)
                else:
                    # Approaching from outside: log barrier
                    d = sd
                    if d < dhat:
                        B = _log_barrier(d, dhat)
                        total_energy += kappa * B
                        dBdd = _log_barrier_derivative(d, dhat)
                        gradient[off + int(vi)] += kappa * dBdd * n_dir

    return total_energy, gradient


# ---------------------------------------------------------------------------
# Quasi-static solver
# ---------------------------------------------------------------------------
def solve_quasi_static(positions, rest_positions, tetrahedra, Dm_inv, volumes,
                       mu, lam, fixed_mask, fixed_targets, bone_data,
                       dhat, kappa, max_iterations=100, tolerance=1e-4,
                       verbose=False):
    """Minimize E_elastic + E_shape + kappa * E_contact via L-BFGS.

    E_elastic:  co-rotated FEM (shape preservation + volume)
    E_shape:    spring to LBS initial guess (regularization)
    E_contact:  one-sided barrier on bone surface

    The LBS initial guess is used as both starting point and shape anchor,
    so the elastic energy doesn't fight the contact by pulling through bone.

    Returns: (final_positions, n_iterations, final_energy)
    """
    n = len(positions)
    free_idx = np.where(~fixed_mask)[0]
    fixed_idx = np.where(fixed_mask)[0]
    n_free = len(free_idx)

    # Set fixed positions
    full_pos = positions.copy()
    if fixed_targets is not None:
        full_pos[fixed_idx] = fixed_targets

    # Save LBS as shape reference (anchor)
    lbs_ref = full_pos[free_idx].copy()
    shape_weight = 1.0  # spring to LBS

    iter_count = [0]

    def energy_and_grad(x_flat):
        """Combined energy + gradient for L-BFGS."""
        x_free = x_flat.reshape(-1, 3)
        full_pos[free_idx] = x_free

        # 1. Co-rotated FEM elastic (volume + shape preservation)
        E_el, g_el = elastic_energy_and_gradient(
            full_pos, tetrahedra, Dm_inv, volumes, mu, lam)

        # 2. Shape anchor: spring to LBS (prevents drift)
        diff = x_free - lbs_ref
        E_shape = 0.5 * shape_weight * np.sum(diff ** 2)
        g_shape = np.zeros_like(full_pos)
        g_shape[free_idx] = shape_weight * diff

        # 3. Contact barrier (one-sided)
        E_co, g_co = contact_energy_and_gradient(
            full_pos, ~fixed_mask, bone_data, dhat, kappa)

        E_total = E_el + E_shape + E_co
        g_total = g_el + g_shape + g_co
        g_free = g_total[free_idx].ravel()

        iter_count[0] += 1
        if verbose and iter_count[0] % 10 == 0:
            print(f"  L-BFGS iter {iter_count[0]}: "
                  f"E={E_total:.2f} (fem={E_el:.2f}, shape={E_shape:.2f}, "
                  f"contact={E_co:.2f}), |g|={np.linalg.norm(g_free):.2e}")

        return E_total, g_free.astype(np.float64)

    x0 = full_pos[free_idx].ravel().astype(np.float64)

    result = minimize(
        energy_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': max_iterations,
            'ftol': 1e-12,
            'gtol': 1e-2,  # will likely not reach this
            'maxfun': max_iterations * 5,  # limit total function evals
            'maxcor': 30,
            'maxls': 40,
            'disp': False,
        }
    )

    full_pos[free_idx] = result.x.reshape(-1, 3)

    if verbose:
        print(f"  L-BFGS: {result.nit} iters, success={result.success}, "
              f"E={result.fun:.4f}, |g|={np.linalg.norm(result.jac):.2e}")

    return full_pos, result.nit, result.fun
