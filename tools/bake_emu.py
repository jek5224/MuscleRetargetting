#!/usr/bin/env python3
"""EMU: Efficient Muscle Simulation in Deformation Space (Modi et al. 2020).

Quasistatic muscle sim that optimizes per-tet deformation gradients F instead
of vertex positions. An ACAP continuity energy ties discontinuous F back to a
continuous mesh. Quasi-Newton solver with Woodbury Hessian approximation.

Usage:
    python tools/bake_emu.py --bvh data/motion/walk.bvh --sides L --end-frame 5
"""
import argparse
import gc
import os
import pickle
import re
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, eigsh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH

# ---------------------------------------------------------------------------
# Constants (same as other bake scripts)
# ---------------------------------------------------------------------------
SKEL_XML = "data/zygote_skel.xml"
SKEL_MESH_DIR = "Zygote_Meshes_251229/Skeleton"
MESH_SCALE = 0.01
FLUSH_INTERVAL = 20

UPLEG_MUSCLES = [
    "Adductor_Brevis", "Adductor_Longus", "Adductor_Magnus",
    "Biceps_Femoris", "Gluteus_Maximus", "Gluteus_Medius",
    "Gluteus_Minimus", "Gracilis", "Iliacus",
    "Inferior_Gemellus", "Obturator_Externus", "Obturator_Internus",
    "Pectineus", "Piriformis", "Popliteus",
    "Quadratus_Femoris", "Rectus_Femoris", "Sartorius",
    "Semimembranosus", "Semitendinosus", "Superior_Gemellus",
    "Tensor_Fascia_Lata", "Vastus_Intermedius", "Vastus_Lateralis",
    "Vastus_Medialis",
]

BONE_NAME_TO_BODY = {
    'L_Os_Coxae': 'L_Os_Coxae0', 'L_Femur': 'L_Femur0',
    'L_Tibia_Fibula': 'L_Tibia_Fibula0', 'L_Patella': 'L_Patella0',
    'R_Os_Coxae': 'R_Os_Coxae0', 'R_Femur': 'R_Femur0',
    'R_Tibia_Fibula': 'R_Tibia_Fibula0', 'R_Patella': 'R_Patella0',
    'Saccrum_Coccyx': 'Saccrum_Coccyx0',
}


# ---------------------------------------------------------------------------
# Skeleton / BVH / Muscle loading (reused from bake_contour_sim.py)
# ---------------------------------------------------------------------------
def _find_body(skel, name):
    bn = skel.getBodyNode(name)
    if bn is not None:
        return bn, name
    bn = skel.getBodyNode(name + "0")
    if bn is not None:
        return bn, name + "0"
    norm = name.lower().replace('_', '')
    for bi in range(skel.getNumBodyNodes()):
        b = skel.getBodyNode(bi)
        bn_norm = b.getName().lower().replace('_', '')
        if norm in bn_norm or bn_norm in norm:
            return b, b.getName()
    return None, name


def load_skeleton():
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    return skel, bvh_info, mesh_info


def _detect_bvh_tframe(bvh_path):
    with open(bvh_path, 'r') as f:
        content = f.read()
    for joint in ['LeftLeg', 'RightLeg']:
        pattern = rf'JOINT\s+{joint}\s*\{{[^}}]*?OFFSET\s+([\d.\-e]+)\s+([\d.\-e]+)\s+([\d.\-e]+)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            x, y, z = abs(float(match.group(1))), abs(float(match.group(2))), abs(float(match.group(3)))
            max_axis = max(x, y, z)
            if max_axis < 1e-6:
                continue
            if y / max_axis > 0.8:
                return None
            return 0
    return None


def load_muscle(tet_dir, name):
    """Load contour tet mesh with fiber data."""
    path = os.path.join(tet_dir, f"{name}_tet.npz")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)

    verts = data['vertices'].astype(np.float64)
    tets = data['tetrahedra'].astype(np.int32)

    # Fix tet orientation
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    neg = vol < 0
    if np.any(neg):
        tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()

    # Remove degenerate tets (quality < 1e-4)
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    edge_lens = []
    for i in range(4):
        for j in range(i + 1, 4):
            edge_lens.append(np.linalg.norm(verts[tets[:, i]] - verts[tets[:, j]], axis=1))
    avg_edge = np.mean(edge_lens, axis=0)
    quality = vol / (avg_edge ** 3 + 1e-30)
    good = quality > 1e-4
    if not np.all(good):
        tets = tets[good]

    # Collect fixed vertices
    cap_faces = data.get('cap_face_indices', [])
    fixed_verts = set()
    sim_faces = data.get('sim_faces', data.get('faces'))
    if sim_faces is not None and len(cap_faces) > 0:
        for fi in cap_faces:
            if fi < len(sim_faces):
                for vi in sim_faces[fi]:
                    fixed_verts.add(int(vi))
    anchors = data.get('anchor_vertices', np.array([], dtype=np.int32))
    for vi in anchors:
        fixed_verts.add(int(vi))

    return {
        'name': name,
        'vertices': verts,
        'rest_vertices': verts.copy(),
        'tetrahedra': tets,
        'fixed_vertices': sorted(fixed_verts),
        'cap_attachments': data.get('cap_attachments', []),
        'attach_skeleton_names': data.get('attach_skeleton_names', []),
        'vertex_contour_level': data.get('vertex_contour_level', None),
        'remap': None,
    }


def compute_multibone_lbs(muscle, skel):
    """Compute multi-bone LBS bindings via heat diffusion from cap anchors."""
    rest_verts = muscle['rest_vertices']
    n_verts = len(rest_verts)
    attach_names = muscle.get('attach_skeleton_names', [])
    cap_att = muscle.get('cap_attachments', [])
    remap = muscle.get('remap')

    if len(attach_names) == 0:
        return None

    skel.setPositions(np.zeros(skel.getNumDofs()))
    bone_info = {}
    for stream_names in attach_names:
        for raw_name in stream_names:
            if raw_name in bone_info:
                continue
            node, resolved = _find_body(skel, raw_name)
            if node is None:
                continue
            R0 = node.getWorldTransform().rotation()
            t0 = node.getWorldTransform().translation()
            bone_info[raw_name] = {'node_name': resolved, 'R0': R0, 't0': t0}

    if len(bone_info) == 0:
        return None

    bone_names = list(bone_info.keys())
    per_vertex = {}
    anchor_bone = {}

    # Assign cap attachment anchors
    if len(cap_att) > 0:
        for row in cap_att:
            orig_vi = int(row[0])
            stream_idx = int(row[1])
            end_idx = int(row[2])
            if stream_idx < len(attach_names) and end_idx < len(attach_names[stream_idx]):
                bone_name = attach_names[stream_idx][end_idx]
                vi = orig_vi
                if vi >= 0 and vi < n_verts and bone_name in bone_info:
                    per_vertex[vi] = [(bone_name, 1.0)]
                    anchor_bone[vi] = bone_name

    # Assign all fixed vertices to nearest bone
    fixed_verts = muscle.get('fixed_vertices', [])
    if len(anchor_bone) > 0 and len(fixed_verts) > len(anchor_bone):
        bone_centers = {bn: bi['t0'] for bn, bi in bone_info.items()}
        for vi in fixed_verts:
            if vi in per_vertex or vi >= n_verts:
                continue
            best_bone, best_dist = None, float('inf')
            for bname, center in bone_centers.items():
                d = np.linalg.norm(rest_verts[vi] - center)
                if d < best_dist:
                    best_dist, best_bone = d, bname
            if best_bone:
                per_vertex[vi] = [(best_bone, 1.0)]
                anchor_bone[vi] = best_bone

    # Heat diffusion
    tets = muscle['tetrahedra']
    adj = [set() for _ in range(n_verts)]
    for t in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                adj[t[i]].add(t[j])
                adj[t[j]].add(t[i])

    bone_cap_verts = {}
    for vi, bname in anchor_bone.items():
        bone_cap_verts.setdefault(bname, []).append(vi)

    fixed = set(anchor_bone.keys())
    bone_heat = {}
    for bname in bone_names:
        if bname not in bone_cap_verts:
            continue
        heat = np.zeros(n_verts)
        for vi in bone_cap_verts[bname]:
            heat[vi] = 1.0
        bone_heat[bname] = heat

    for _ in range(100):
        for bname, heat in bone_heat.items():
            new_heat = heat.copy()
            for vi in range(n_verts):
                if vi in fixed or len(adj[vi]) == 0:
                    continue
                new_heat[vi] = np.mean([heat[ni] for ni in adj[vi]])
            bone_heat[bname] = new_heat

    for vi in range(n_verts):
        if vi in per_vertex:
            continue
        weights = [(bname, heat[vi]) for bname, heat in bone_heat.items() if heat[vi] > 1e-8]
        if not weights:
            best_bone = min(bone_names, key=lambda b: np.linalg.norm(rest_verts[vi] - bone_info[b]['t0']))
            weights = [(best_bone, 1.0)]
        total = sum(w for _, w in weights)
        per_vertex[vi] = [(b, w / total) for b, w in weights]

    bindings = []
    for vi in range(n_verts):
        if vi not in per_vertex:
            per_vertex[vi] = [(bone_names[0], 1.0)]
        bone_weights = []
        for bname, w in per_vertex[vi]:
            if bname not in bone_info:
                continue
            bi = bone_info[bname]
            bone_weights.append((bi['node_name'], w, bi['R0'], bi['t0']))
        bindings.append((rest_verts[vi].copy(), bone_weights))
    return bindings


def compute_lbs_positions(bindings, skel, n_verts):
    if bindings is None:
        return None
    positions = np.zeros((n_verts, 3))
    for vi, (rest_pos, bone_weights) in enumerate(bindings):
        if len(bone_weights) == 0:
            positions[vi] = rest_pos
            continue
        pos = np.zeros(3)
        for node_name, w, R0, t0 in bone_weights:
            node = skel.getBodyNode(node_name)
            if node is None:
                pos += w * rest_pos
                continue
            R1 = node.getWorldTransform().rotation()
            t1 = node.getWorldTransform().translation()
            pos += w * (R1 @ (R0.T @ (rest_pos - t0)) + t1)
        positions[vi] = pos
    return positions


def lame_parameters(youngs, poisson):
    mu = youngs / (2 * (1 + poisson))
    lam = youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))
    return mu, lam


# ---------------------------------------------------------------------------
# EMU Core: G matrix, ACAP, Neo-Hookean
# ---------------------------------------------------------------------------
def build_gradient_operator(vertices, tetrahedra):
    """Build the gradient operator G (9m × 3n sparse).

    G maps vertex positions q (3n) to per-tet deformation gradients F (9m).
    F_i = Ds_i @ Dm_inv_i where Ds_i = [q1-q4, q2-q4, q3-q4] (current edges).

    Convention: F stored as row-major vec(F) = [F00,F01,F02,F10,...,F22].
    """
    n = len(vertices)
    m = len(tetrahedra)

    # Compute Dm_inv for each tet (rest shape)
    X = vertices
    tets = tetrahedra
    x4 = X[tets[:, 3]]
    Dm = np.stack([X[tets[:, 0]] - x4, X[tets[:, 1]] - x4, X[tets[:, 2]] - x4], axis=-1)
    # Dm shape: (m, 3, 3) — columns are edge vectors
    Dm_inv = np.linalg.inv(Dm)  # (m, 3, 3)

    # Compute tet volumes
    volumes = np.abs(np.linalg.det(Dm)) / 6.0  # (m,)

    # Build G as sparse matrix
    # F_i = Ds_i @ Dm_inv_i
    # Ds_i = [q_{t0} - q_{t3}, q_{t1} - q_{t3}, q_{t2} - q_{t3}]
    # F_{i,ab} = sum_c Ds_{i,ac} * Dm_inv_{i,cb}
    # Ds_{i,ac} = q_{t_c, a} - q_{t_3, a}  for c=0,1,2
    #
    # So F_{i,ab} = sum_{c=0,1,2} (q_{t_c, a} - q_{t_3, a}) * Dm_inv_i[c, b]
    #            = sum_{c=0,1,2} q_{t_c, a} * Dm_inv_i[c, b] - q_{t_3, a} * sum_c Dm_inv_i[c, b]
    #
    # In the 9m-vector: row = 9*i + 3*a + b (row-major F)
    # Column for q_{v, a}: col = 3*v + a
    #
    # Coefficient of q_{t_c, a} in F_{i,ab} = Dm_inv_i[c, b]
    # Coefficient of q_{t_3, a} in F_{i,ab} = -sum_c Dm_inv_i[c, b]

    rows = []
    cols = []
    vals = []

    for c in range(3):  # vertex index in tet (0,1,2)
        for a in range(3):  # F row
            for b in range(3):  # F col
                row_idx = 9 * np.arange(m) + 3 * a + b  # (m,)
                vert_idx = tets[:, c]
                col_idx = 3 * vert_idx + a  # (m,)
                val = Dm_inv[:, c, b]  # (m,)

                rows.append(row_idx)
                cols.append(col_idx)
                vals.append(val)

    # Vertex 3 (t3) contribution: coefficient = -sum_c Dm_inv[c, b]
    sum_Dm_inv = Dm_inv.sum(axis=1)  # (m, 3) — sum over c
    for a in range(3):
        for b in range(3):
            row_idx = 9 * np.arange(m) + 3 * a + b
            vert_idx = tets[:, 3]
            col_idx = 3 * vert_idx + a
            val = -sum_Dm_inv[:, b]

            rows.append(row_idx)
            cols.append(col_idx)
            vals.append(val)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    G = sp.csc_matrix((vals, (rows, cols)), shape=(9 * m, 3 * n))

    return G, Dm_inv, volumes


def compute_fiber_directions(vertices, tetrahedra, vertex_contour_level):
    """Compute per-tet fiber direction from contour level gradient.

    The fiber direction is along the gradient of the scalar contour level field
    within each tetrahedron. For cap tets (uniform level), use the muscle's
    principal axis as fallback.
    """
    m = len(tetrahedra)
    X = vertices
    tets = tetrahedra

    # Compute Dm_inv
    x4 = X[tets[:, 3]]
    Dm = np.stack([X[tets[:, 0]] - x4, X[tets[:, 1]] - x4, X[tets[:, 2]] - x4], axis=-1)
    Dm_inv = np.linalg.inv(Dm)

    # Contour level at each tet vertex
    if vertex_contour_level is None:
        fiber_dirs = np.tile(np.array([0, 1, 0], dtype=np.float64), (m, 1))
        return fiber_dirs

    # Pad vertex_contour_level if shorter than vertex count
    n = len(vertices)
    vcl = vertex_contour_level
    if len(vcl) < n:
        vcl_padded = np.zeros(n, dtype=vcl.dtype)
        vcl_padded[:len(vcl)] = vcl
        # Assign extra vertices to nearest existing vertex's level
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices[:len(vcl)])
        _, nearest = tree.query(vertices[len(vcl):])
        vcl_padded[len(vcl):] = vcl[nearest]
        vcl = vcl_padded

    levels = vcl[tetrahedra].astype(np.float64)  # (m, 4)
    dl = np.stack([levels[:, 0] - levels[:, 3],
                   levels[:, 1] - levels[:, 3],
                   levels[:, 2] - levels[:, 3]], axis=-1)  # (m, 3)

    # grad(level) = Dm_inv @ dl  (per-tet, but Dm_inv is (m,3,3) and dl is (m,3))
    grad = np.einsum('mij,mj->mi', Dm_inv, dl)  # (m, 3)
    norms = np.linalg.norm(grad, axis=1, keepdims=True)

    # Fallback for zero-gradient tets: use muscle principal axis
    zero_mask = (norms.ravel() < 1e-10)
    if np.any(zero_mask):
        # Muscle axis: from min to max centroid along Y
        centroids = X[tets].mean(axis=1)
        axis = centroids.max(axis=0) - centroids.min(axis=0)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis /= axis_norm
        else:
            axis = np.array([0, 1, 0])
        grad[zero_mask] = axis

    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    fiber_dirs = grad / norms

    return fiber_dirs


def precompute_emu(vertices, tetrahedra, fixed_vertices, vertex_contour_level, k_modes=48):
    """Precompute all EMU data structures.

    Builds a reduced ACAP system (Eq. 21) with fixed vertex Dirichlet
    constraints baked in, so the ACAP solve correctly pins cap vertices
    to bone positions while solving free vertices consistently.
    """
    n = len(vertices)
    m = len(tetrahedra)
    print(f"    EMU precompute: {n} verts, {m} tets, {len(fixed_vertices)} fixed")

    t0 = time.time()
    G, Dm_inv, volumes = build_gradient_operator(vertices, tetrahedra)
    GtG = G.T @ G  # (3n, 3n) sparse
    Gt = G.T.tocsc()
    print(f"    G: {G.shape}, GtG: {GtG.shape}, built in {time.time()-t0:.2f}s")

    # Build fixed/free index sets in the 3n DOF space
    # Each vertex i has 3 DOFs: 3i, 3i+1, 3i+2
    fixed_set = set(fixed_vertices)
    free_vert_idx = np.array([i for i in range(n) if i not in fixed_set], dtype=np.int32)
    fixed_vert_idx = np.array(sorted(fixed_set), dtype=np.int32)
    free_dofs = np.concatenate([3 * free_vert_idx + d for d in range(3)])
    free_dofs.sort()
    fixed_dofs = np.concatenate([3 * fixed_vert_idx + d for d in range(3)])
    fixed_dofs.sort()

    # Reduced ACAP system: L_ff q_free = rhs_free - L_fc q_fixed
    t0 = time.time()
    reg = 1e-6
    GtG_reg = GtG + reg * sp.eye(3 * n)
    GtG_csc = GtG_reg.tocsc()
    L_ff = GtG_csc[np.ix_(free_dofs, free_dofs)].tocsc()
    L_fc = GtG_csc[np.ix_(free_dofs, fixed_dofs)].tocsc()
    acap_solver = splu(L_ff)
    print(f"    Reduced ACAP: {len(free_dofs)} free, {len(fixed_dofs)} fixed, "
          f"factorized in {time.time()-t0:.2f}s")

    # Eigendecomposition for Woodbury (Eq. 13)
    t0 = time.time()
    k = min(k_modes, 3 * n - 2)
    eigenvalues, eigenvectors = eigsh(GtG_csc, k=k, which='LM')
    Phi = eigenvectors  # (3n, k)
    Lambda_inv = np.diag(1.0 / (eigenvalues + 1e-30))  # (k, k)

    B_sparse = Phi.T @ Gt  # (k, 9m)
    B = B_sparse if isinstance(B_sparse, np.ndarray) else B_sparse.toarray()
    print(f"    Eigenmodes: k={k}, captured {eigenvalues.sum():.2f} "
          f"({time.time()-t0:.2f}s)")

    # Fiber directions
    fiber_dirs = compute_fiber_directions(vertices, tetrahedra, vertex_contour_level)

    return {
        'G': G,
        'Gt': Gt,
        'GtG': GtG,
        'GtG_csc': GtG_csc,
        'acap_solver': acap_solver,  # reduced system solver
        'L_fc': L_fc,
        'free_dofs': free_dofs,
        'fixed_dofs': fixed_dofs,
        'free_vert_idx': free_vert_idx,
        'fixed_vert_idx': fixed_vert_idx,
        'Phi': Phi,
        'Lambda_inv': Lambda_inv,
        'B': B,
        'Dm_inv': Dm_inv,
        'volumes': volumes,
        'fiber_dirs': fiber_dirs,
        'n_verts': n,
        'n_tets': m,
    }


# ---------------------------------------------------------------------------
# Stable Neo-Hookean energy, gradient, Hessian (vectorized)
# ---------------------------------------------------------------------------
def _deformation_gradients_from_q(q, tetrahedra, Dm_inv):
    """Compute per-tet F from vertex positions q. Returns (m, 3, 3)."""
    tets = tetrahedra
    x4 = q[tets[:, 3]]
    Ds = np.stack([q[tets[:, 0]] - x4, q[tets[:, 1]] - x4, q[tets[:, 2]] - x4], axis=-1)
    F = np.einsum('mij,mjk->mik', Ds, Dm_inv)  # (m, 3, 3)
    return F


def _F_to_vec(F):
    """(m, 3, 3) → (9m,) row-major."""
    return F.reshape(-1)


def _vec_to_F(Fvec, m):
    """(9m,) → (m, 3, 3)."""
    return Fvec.reshape(m, 3, 3)


def stable_neohookean_energy(F, volumes, mu, lam):
    """Stable Neo-Hookean energy, vectorized over all tets.

    Ψ = μ/2(I_C - 3) - μ(J - 1) + λ/2(J - 1)²
    where I_C = tr(F^T F), J = det(F).
    Total energy = sum_i V_i * Ψ_i
    """
    I_C = np.einsum('mij,mij->m', F, F)  # tr(F^T F), (m,)
    J = np.linalg.det(F)  # (m,)

    psi = mu / 2 * (I_C - 3) - mu * (J - 1) + lam / 2 * (J - 1) ** 2
    return np.sum(volumes * psi)


def stable_neohookean_gradient(F, volumes, mu, lam):
    """∂Ψ/∂F for Stable Neo-Hookean, returns (m, 3, 3).

    P = μF - μ J F^{-T} + λ(J-1) J F^{-T}
    ∂(VΨ)/∂F = V * P
    """
    m = len(F)
    J = np.linalg.det(F)  # (m,)

    # F^{-T} = (F^{-1})^T — handle near-singular with pseudoinverse
    F_invT = np.zeros_like(F)
    for i in range(m):
        try:
            F_invT[i] = np.linalg.inv(F[i]).T
        except np.linalg.LinAlgError:
            F_invT[i] = np.linalg.pinv(F[i]).T

    P = mu * F - mu * J[:, None, None] * F_invT + lam * ((J - 1) * J)[:, None, None] * F_invT

    return volumes[:, None, None] * P  # (m, 3, 3)


def stable_neohookean_hessian_blocks(F, volumes, mu, lam):
    """Per-tet 9×9 Hessian blocks of V*Ψ_iso w.r.t. vec(F_i).

    Uses the SPD projection from [SGK18] — guaranteed positive definite.
    For simplicity, we use the analytical Hessian clamped to SPD via eigenvalue fix.
    """
    m = len(F)
    H = np.zeros((m, 9, 9))
    min_diag = 1e-4  # minimum diagonal value for SPD guarantee

    for i in range(m):
        Fi = F[i]
        V = max(volumes[i], 1e-15)
        Ji = np.linalg.det(Fi)

        try:
            FiT_inv = np.linalg.inv(Fi).T
        except np.linalg.LinAlgError:
            # Degenerate F — use identity-like Hessian
            H[i] = V * mu * np.eye(9) + min_diag * np.eye(9)
            continue

        g = FiT_inv.ravel()
        H[i] = V * (mu * np.eye(9) + lam * max(Ji * Ji, 0) * np.outer(g, g))

        # Ensure SPD
        H[i] += min_diag * np.eye(9)

    return H  # (m, 9, 9)


# ---------------------------------------------------------------------------
# ACAP energy and position recovery
# ---------------------------------------------------------------------------
def acap_positions(Fvec, precomp, fixed_targets_flat):
    """Recover vertex positions via ACAP with Dirichlet constraints (Eq. 4, 21).

    Solves the reduced system:
        L_ff q_free = rhs_free - L_fc q_fixed
    where q_fixed = fixed_targets_flat[fixed_dofs].
    """
    n = precomp['n_verts']
    free_dofs = precomp['free_dofs']
    fixed_dofs = precomp['fixed_dofs']

    full_rhs = (precomp['Gt'] @ Fvec).toarray().ravel() if sp.issparse(precomp['Gt'] @ Fvec) else precomp['Gt'] @ Fvec

    q_fixed = fixed_targets_flat[fixed_dofs]
    rhs_free = full_rhs[free_dofs] - precomp['L_fc'] @ q_fixed

    q_free = precomp['acap_solver'].solve(rhs_free)

    q_flat = np.zeros(3 * n)
    q_flat[free_dofs] = q_free
    q_flat[fixed_dofs] = q_fixed

    return q_flat.reshape(-1, 3)


def acap_energy(Fvec, precomp, fixed_targets_flat):
    """E_C = ½ ||G q(F) - F||² with constrained ACAP solve."""
    q = acap_positions(Fvec, precomp, fixed_targets_flat)
    q_flat = q.ravel()
    residual = precomp['G'] @ q_flat - Fvec
    return 0.5 * np.dot(residual, residual)


# ---------------------------------------------------------------------------
# EMU total energy and gradient
# ---------------------------------------------------------------------------
def emu_energy(Fvec, precomp, fixed_targets_flat, mu, lam, alpha):
    """Total EMU energy: Ψ_iso + α·E_C (Eq. 5, no fiber activation)."""
    m = precomp['n_tets']
    F = _vec_to_F(Fvec, m)
    E_iso = stable_neohookean_energy(F, precomp['volumes'], mu, lam)
    E_c = acap_energy(Fvec, precomp, fixed_targets_flat)
    return E_iso + alpha * E_c


def emu_gradient(Fvec, precomp, fixed_targets_flat, mu, lam, alpha):
    """Gradient dE/dF (Eq. 9).

    dE/dF = ∂Ψ_iso/∂F + α·∂E_C/∂F
    ∂E_C/∂F = F - G q(F)  where q(F) is the constrained ACAP solve (Eq. 10)
    """
    m = precomp['n_tets']
    F = _vec_to_F(Fvec, m)

    # Neo-Hookean gradient
    P = stable_neohookean_gradient(F, precomp['volumes'], mu, lam)
    g_iso = _F_to_vec(P)

    # ACAP gradient with constrained solve
    q = acap_positions(Fvec, precomp, fixed_targets_flat)
    Gq = precomp['G'] @ q.ravel()
    g_acap = Fvec - Gq  # (9m,)

    return g_iso + alpha * g_acap


# ---------------------------------------------------------------------------
# Newton step with Woodbury (Eq. 14-17)
# ---------------------------------------------------------------------------
def newton_step_woodbury(gradient, Fvec, precomp, mu, lam, alpha):
    """Compute Newton descent direction using Woodbury Hessian approximation.

    H_approx = H + α I  (block diagonal, H from Ψ_iso)
    Full Hessian ≈ H_approx - α G (G^TG)^{-1} G^T  (via Woodbury)

    Uses precomputed eigenmodes (Eq. 13-17).
    """
    m = precomp['n_tets']
    F = _vec_to_F(Fvec, m)

    # Block-diagonal H: (m, 9, 9) per tet
    H_blocks = stable_neohookean_hessian_blocks(F, precomp['volumes'], mu, lam)

    # Add α·I to each block (ACAP Hessian diagonal contribution)
    for i in range(m):
        H_blocks[i] += alpha * np.eye(9)

    # H^{-1} g (block-diagonal inverse applied to gradient)
    g_blocks = gradient.reshape(m, 9)
    Hinv_g = np.zeros_like(g_blocks)
    for i in range(m):
        try:
            Hinv_g[i] = np.linalg.solve(H_blocks[i], g_blocks[i])
        except np.linalg.LinAlgError:
            Hinv_g[i] = g_blocks[i] / (np.diag(H_blocks[i]).mean() + 1e-8)
    Hinv_g_flat = Hinv_g.ravel()

    # Woodbury correction (Eq. 17):
    # d = H^{-1}g - H^{-1} B^T (Λ/α + B H^{-1} B^T)^{-1} B H^{-1} g
    B = precomp['B']  # (k, 9m)
    Lambda_inv = precomp['Lambda_inv']  # (k, k)

    # B @ H^{-1} — need H^{-1} applied to each column of B^T
    # Instead compute B @ Hinv_g directly (vector), and B @ H^{-1} @ B^T (matrix)
    B_Hinv_g = B @ Hinv_g_flat  # (k,)

    # B H^{-1} B^T: (k, k) — expensive but k is small (48)
    # Apply H^{-1} to each row of B^T
    BT = B.T  # (9m, k)
    BT_blocks = BT.reshape(m, 9, -1)  # (m, 9, k)
    Hinv_BT = np.zeros_like(BT_blocks)
    for i in range(m):
        try:
            Hinv_BT[i] = np.linalg.solve(H_blocks[i], BT_blocks[i])
        except np.linalg.LinAlgError:
            Hinv_BT[i] = BT_blocks[i] / (np.diag(H_blocks[i]).mean() + 1e-8)
    Hinv_BT_flat = Hinv_BT.reshape(9 * m, -1)  # (9m, k)
    B_Hinv_BT = B @ Hinv_BT_flat  # (k, k)

    # Middle matrix: (Λ/α + B H^{-1} B^T)^{-1}
    middle = Lambda_inv / alpha + B_Hinv_BT  # Wait, paper says Λ/α
    # Eq 17: (Λ - αBH^{-1}B^T)^{-1} ... actually let me re-derive.
    # The Woodbury identity for (A + UCV)^{-1}:
    # Here our full Hessian is H_block + α(I - G(G^TG)^{-1}G^T)
    # The low-rank part is -α G(G^TG)^{-1}G^T ≈ -α Φ Λ^{-1} Φ^T (in F-space via B)
    # So: (H_block + αI - α B^T Λ^{-1} B)
    # Woodbury: (A - UCV)^{-1} = A^{-1} + A^{-1}U(C^{-1} - VA^{-1}U)^{-1}VA^{-1}
    # A = H_block + αI, U = B^T, C = αΛ^{-1}, V = B
    # middle = (C^{-1} - B A^{-1} B^T)^{-1} = (Λ/α - B_Hinv_BT)^{-1}

    eigenvalues = np.diag(1.0 / (np.diag(Lambda_inv) + 1e-30))  # Λ
    C_inv = eigenvalues / alpha  # Λ/α
    middle_matrix = C_inv - B_Hinv_BT
    try:
        middle_inv = np.linalg.inv(middle_matrix)
    except np.linalg.LinAlgError:
        middle_inv = np.linalg.pinv(middle_matrix)

    # Correction: A^{-1} U (middle)^{-1} V A^{-1} g
    correction = Hinv_BT_flat @ (middle_inv @ B_Hinv_g)  # (9m,)

    direction = -(Hinv_g_flat + correction)
    return direction


# ---------------------------------------------------------------------------
# EMU solver (Algorithm 1)
# ---------------------------------------------------------------------------
def emu_solve(q_init, precomp, fixed_mask, fixed_targets,
              mu, lam, alpha, max_iters=30, verbose=False):
    """Run EMU solver using L-BFGS optimization in deformation gradient space.

    Args:
        q_init: (n, 3) initial vertex positions (from LBS)
        precomp: dict from precompute_emu
        fixed_mask: (n,) bool array
        fixed_targets: (n, 3) target positions for fixed vertices
        mu, lam: Lamé parameters
        alpha: ACAP continuity weight

    Returns:
        q_final: (n, 3) optimized vertex positions
        info: dict with convergence stats
    """
    from scipy.optimize import minimize as sp_minimize

    fixed_targets_flat = fixed_targets.ravel()

    # Initialize F from q_init
    q_flat = q_init.ravel()
    Fvec_init = (precomp['G'] @ q_flat).copy()

    if verbose:
        E0 = emu_energy(Fvec_init, precomp, fixed_targets_flat, mu, lam, alpha)
        print(f"    EMU init: E={E0:.6e}")

    iter_count = [0]
    def callback(Fvec):
        iter_count[0] += 1

    def objective(Fvec):
        return emu_energy(Fvec, precomp, fixed_targets_flat, mu, lam, alpha)

    def gradient(Fvec):
        return emu_gradient(Fvec, precomp, fixed_targets_flat, mu, lam, alpha)

    result = sp_minimize(
        objective, Fvec_init, jac=gradient, method='L-BFGS-B',
        options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-6,
                 'disp': verbose},
        callback=callback)

    Fvec_final = result.x

    # Recover vertex positions with proper constraints
    q = acap_positions(Fvec_final, precomp, fixed_targets_flat)

    if verbose:
        print(f"    EMU done: E={result.fun:.6e}, iters={result.nit}, "
              f"|g|={np.linalg.norm(result.jac):.2e}, success={result.success}")

    info = {'iterations': result.nit, 'energy': result.fun,
            'grad_norm': np.linalg.norm(result.jac)}
    return q, info


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def flush_bake_data(bake_buffers, output_dir, chunk_counters):
    for mname, frames_dict in bake_buffers.items():
        if not frames_dict:
            continue
        frame_nums = sorted(frames_dict.keys())
        positions = np.stack([frames_dict[f] for f in frame_nums], axis=0)
        chunk_idx = chunk_counters.get(mname, 0)
        path = os.path.join(output_dir, f"{mname}_chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(path,
                            frames=np.array(frame_nums, dtype=np.int32),
                            positions=positions.astype(np.float32))
        chunk_counters[mname] = chunk_idx + 1
        frames_dict.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EMU muscle simulation bake")
    parser.add_argument('--bvh', required=True, help='BVH motion file')
    parser.add_argument('--sides', default='L', help='L, R, or LR')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=None)
    parser.add_argument('--tet-dir', default='tet')
    parser.add_argument('--output-dir', default='data/motion_cache')
    parser.add_argument('--youngs', type=float, default=6e6,
                        help='Young\'s modulus (Pa, default: 6e6 for muscle)')
    parser.add_argument('--poisson', type=float, default=0.49)
    parser.add_argument('--alpha', type=float, default=1e3,
                        help='ACAP continuity weight')
    parser.add_argument('--max-iters', type=int, default=30,
                        help='Newton iterations per frame')
    parser.add_argument('--k-modes', type=int, default=48,
                        help='Eigenmodes for Woodbury approximation')
    args = parser.parse_args()

    mu, lam = lame_parameters(args.youngs, args.poisson)

    # ── Load skeleton ────────────────────────────────────────────────
    print("[1] Loading skeleton...")
    skel, bvh_info, mesh_info = load_skeleton()
    print(f"    DOFs: {skel.getNumDofs()}, Bodies: {skel.getNumBodyNodes()}")

    # ── Load BVH ─────────────────────────────────────────────────────
    print(f"[2] Loading BVH: {args.bvh}")
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    end_frame = args.end_frame if args.end_frame is not None else n_frames - 1
    end_frame = min(end_frame, n_frames - 1)
    total_frames = end_frame - args.start_frame + 1
    print(f"    Frames: {n_frames}, baking {args.start_frame}-{end_frame} ({total_frames})")

    # ── Load muscles ─────────────────────────────────────────────────
    print(f"[3] Loading muscles from {args.tet_dir}/...")
    muscles = []
    for side in args.sides:
        for muscle_name in UPLEG_MUSCLES:
            name = f"{side}_{muscle_name}"
            data = load_muscle(args.tet_dir, name)
            if data is not None:
                muscles.append(data)

    if not muscles:
        print("ERROR: No muscles loaded!")
        sys.exit(1)

    total_verts = sum(len(m['vertices']) for m in muscles)
    total_tets = sum(len(m['tetrahedra']) for m in muscles)
    print(f"    {len(muscles)} muscles: {total_verts} verts, {total_tets} tets")

    # ── Compute LBS bindings ─────────────────────────────────────────
    print("[4] Computing LBS bindings...")
    for m in muscles:
        m['lbs_bindings'] = compute_multibone_lbs(m, skel)

    # ── EMU precomputation (per muscle) ──────────────────────────────
    print("[5] EMU precomputation...")
    for m in muscles:
        m['emu'] = precompute_emu(
            m['vertices'], m['tetrahedra'], m['fixed_vertices'],
            m.get('vertex_contour_level'), k_modes=args.k_modes)

    # ── Setup output ─────────────────────────────────────────────────
    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    side_tag = ''.join(args.sides)
    out_dir = os.path.join(args.output_dir, bvh_stem, f"emu_{side_tag}_UpLeg")
    os.makedirs(out_dir, exist_ok=True)
    print(f"    Output: {out_dir}")

    bake_buffers = {m['name']: {} for m in muscles}
    chunk_counters = {m['name']: 0 for m in muscles}

    # ── Bake loop ────────────────────────────────────────────────────
    print(f"\n[6] Baking frames {args.start_frame}-{end_frame}...")
    t_total = time.time()

    for frame in range(args.start_frame, end_frame + 1):
        t0 = time.time()
        pose = motion_bvh.mocap_refs[frame].copy()
        skel.setPositions(pose)

        for m in muscles:
            if m['lbs_bindings'] is None:
                continue

            n_v = len(m['vertices'])
            lbs_pos = compute_lbs_positions(m['lbs_bindings'], skel, n_v)

            # Build fixed mask and targets
            fixed_mask = np.zeros(n_v, dtype=bool)
            for vi in m['fixed_vertices']:
                if vi < n_v:
                    fixed_mask[vi] = True

            # EMU solve
            q, info = emu_solve(
                lbs_pos, m['emu'], fixed_mask, lbs_pos,
                mu, lam, args.alpha,
                max_iters=args.max_iters,
                verbose=(frame == args.start_frame))

            bake_buffers[m['name']][frame] = q.astype(np.float32)

        dt = time.time() - t0
        print(f"  Frame {frame}: {dt:.2f}s", flush=True)

        if (frame - args.start_frame + 1) % FLUSH_INTERVAL == 0:
            flush_bake_data(bake_buffers, out_dir, chunk_counters)
            gc.collect()

    flush_bake_data(bake_buffers, out_dir, chunk_counters)

    elapsed = time.time() - t_total
    print(f"\nDone. {len(muscles)} muscles, {total_frames} frames, "
          f"{elapsed:.1f}s total ({elapsed/max(total_frames,1):.2f}s/frame)")
    print(f"Output: {out_dir}")


if __name__ == '__main__':
    main()
