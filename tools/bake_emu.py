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

# Limit BLAS threads for multiprocessing (must be set before numpy import)
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, eigsh
from scipy.optimize import minimize

import taichi as ti

_ti_initialized = False
def _ensure_ti():
    global _ti_initialized
    if not _ti_initialized:
        ti.init(arch=ti.cuda, default_fp=ti.f64)
        _ti_initialized = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH

# ---------------------------------------------------------------------------
# Taichi GPU kernels
# ---------------------------------------------------------------------------
@ti.kernel
def _ti_hessian_build(
    F_field: ti.types.ndarray(dtype=ti.f64, ndim=2),    # (m, 9)
    vol_field: ti.types.ndarray(dtype=ti.f64, ndim=1),   # (m,)
    mu_val: ti.f64, lam_val: ti.f64, alpha_val: ti.f64,
    H_out: ti.types.ndarray(dtype=ti.f64, ndim=3),      # (m, 9, 9)
):
    """Per-tet: build Hessian H+αI on GPU."""
    for tet_i in range(F_field.shape[0]):
        F = ti.Matrix.zero(ti.f64, 3, 3)
        for a in range(3):
            for b in range(3):
                F[a, b] = F_field[tet_i, 3 * a + b]

        V = ti.max(vol_field[tet_i], 1e-15)
        J = F.determinant()
        # Regularize near-singular F
        if ti.abs(J) < 1e-12:
            for a in range(3):
                F[a, a] += 1e-8
            J = F.determinant()
        g = F.inverse().transpose()

        c = lam_val * (J - 1.0) - mu_val
        coeff1 = J * (lam_val * (2.0 * J - 1.0) - mu_val)
        coeff2 = c * J

        for ii in range(9):
            ai = ii // 3
            bi = ii % 3
            for jj in range(9):
                cj = jj // 3
                dj = jj % 3
                val = coeff1 * g[ai, bi] * g[cj, dj] - coeff2 * g[cj, bi] * g[ai, dj]
                if ai == cj and bi == dj:
                    val += mu_val
                H_out[tet_i, ii, jj] = V * val
        for ii in range(9):
            H_out[tet_i, ii, ii] += alpha_val


@ti.kernel
def _ti_snh_gradient(
    F_field: ti.types.ndarray(dtype=ti.f64, ndim=2),   # (m, 9)
    vol_field: ti.types.ndarray(dtype=ti.f64, ndim=1),  # (m,)
    mu_val: ti.f64, lam_val: ti.f64,
    P_out: ti.types.ndarray(dtype=ti.f64, ndim=2),      # (m, 9)
):
    """Per-tet SNH gradient (PK1 stress × volume)."""
    for tet_i in range(F_field.shape[0]):
        F = ti.Matrix.zero(ti.f64, 3, 3)
        for a in range(3):
            for b in range(3):
                F[a, b] = F_field[tet_i, 3 * a + b]
        V = ti.max(vol_field[tet_i], 1e-15)
        J = F.determinant()
        F_invT = F.inverse().transpose()
        coeff = (lam_val * (J - 1.0) - mu_val) * J
        for a in range(3):
            for b in range(3):
                P_out[tet_i, 3 * a + b] = V * (mu_val * F[a, b] + coeff * F_invT[a, b])


@ti.kernel
def _ti_snh_energy(
    F_field: ti.types.ndarray(dtype=ti.f64, ndim=2),   # (m, 9)
    vol_field: ti.types.ndarray(dtype=ti.f64, ndim=1),  # (m,)
    mu_val: ti.f64, lam_val: ti.f64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),     # (1,) output
):
    """Sum of per-tet SNH energy."""
    for tet_i in range(F_field.shape[0]):
        F = ti.Matrix.zero(ti.f64, 3, 3)
        for a in range(3):
            for b in range(3):
                F[a, b] = F_field[tet_i, 3 * a + b]
        V = ti.max(vol_field[tet_i], 1e-15)
        I_C = 0.0
        for a in range(3):
            for b in range(3):
                I_C += F[a, b] * F[a, b]
        J = F.determinant()
        psi = mu_val / 2.0 * (I_C - 3.0) - mu_val * (J - 1.0) + lam_val / 2.0 * (J - 1.0) ** 2
        result[0] += V * psi


# ---------------------------------------------------------------------------
# GPU-accelerated wrappers
# ---------------------------------------------------------------------------
def stable_neohookean_energy_gpu(F, volumes, mu, lam):
    """GPU SNH energy."""
    _ensure_ti()
    m = len(F)
    F_flat = F.reshape(m, 9).copy()
    result = np.zeros(1, dtype=np.float64)
    _ti_snh_energy(F_flat, volumes, mu, lam, result)
    return result[0]


def stable_neohookean_gradient_gpu(F, volumes, mu, lam):
    """GPU SNH gradient, returns (m, 3, 3)."""
    _ensure_ti()
    m = len(F)
    F_flat = F.reshape(m, 9).copy()
    P_out = np.zeros((m, 9), dtype=np.float64)
    _ti_snh_gradient(F_flat, volumes, mu, lam, P_out)
    return P_out.reshape(m, 3, 3)


def newton_step_woodbury_gpu(gradient, Fvec, precomp, mu, lam, alpha):
    """Woodbury Newton step: GPU Hessian build + numpy batched solve."""
    _ensure_ti()
    m = precomp['n_tets']
    F = _vec_to_F(Fvec, m)
    F_flat = F.reshape(m, 9).copy()

    # GPU: build V*H blocks WITHOUT αI (m × 9 × 9)
    H_blocks = np.zeros((m, 9, 9), dtype=np.float64)
    _ti_hessian_build(F_flat, precomp['volumes'], mu, lam, 0.0, H_blocks)

    # SPD projection on V*H (same as CPU), then add αI
    # Replace NaN/Inf from degenerate tets with identity
    bad = ~np.isfinite(H_blocks).all(axis=(1, 2))
    if np.any(bad):
        H_blocks[bad] = np.eye(9)
    eigvals, eigvecs = np.linalg.eigh(H_blocks)
    eigvals = np.maximum(eigvals, 1e-4)
    H_blocks = np.einsum('mij,mj,mkj->mik', eigvecs, eigvals, eigvecs)
    H_blocks += alpha * np.eye(9)[None, :, :]

    # Batched solves — use per-tet fallback if batched solve fails
    g_blocks = gradient.reshape(m, 9)
    try:
        Hinv_g = np.linalg.solve(H_blocks, g_blocks[:, :, None]).squeeze(-1)
    except np.linalg.LinAlgError:
        # Per-tet fallback with lstsq
        Hinv_g = np.zeros_like(g_blocks)
        for i in range(m):
            Hinv_g[i], _, _, _ = np.linalg.lstsq(H_blocks[i], g_blocks[i], rcond=None)
    Hinv_g_flat = Hinv_g.ravel()

    B = precomp['B']  # (k, 9m)
    Lambda = precomp['Lambda']

    B_Hinv_g = B @ Hinv_g_flat

    BT = B.T.reshape(m, 9, -1)  # (m, 9, k)
    try:
        Hinv_BT = np.linalg.solve(H_blocks, BT)
    except np.linalg.LinAlgError:
        Hinv_BT = np.zeros_like(BT)
        for i in range(m):
            Hinv_BT[i], _, _, _ = np.linalg.lstsq(H_blocks[i], BT[i], rcond=None)
    Hinv_BT_flat = Hinv_BT.reshape(9 * m, -1)
    B_Hinv_BT = B @ Hinv_BT_flat

    middle = Lambda - alpha * B_Hinv_BT
    eigvals_m, eigvecs_m = np.linalg.eigh(middle)
    eigvals_m = np.maximum(eigvals_m, np.max(np.abs(eigvals_m)) * 1e-6 + 1e-10)
    middle_inv = eigvecs_m @ np.diag(1.0 / eigvals_m) @ eigvecs_m.T

    correction = alpha * Hinv_BT_flat @ (middle_inv @ B_Hinv_g)
    direction = -(Hinv_g_flat + correction)
    return direction


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
        'origin_attachment_vertices': np.asarray(
            data.get('origin_attachment_vertices', []), dtype=np.int32),
        'insertion_attachment_vertices': np.asarray(
            data.get('insertion_attachment_vertices', []), dtype=np.int32),
        'cap_attachments': data.get('cap_attachments', []),
        'attach_skeleton_names': data.get('attach_skeleton_names', []),
        'vertex_contour_level': data.get('vertex_contour_level', None),
        'fiber_directions': data.get('fiber_directions', None),
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

    # Assign fixed (cap) vertices to their anchor's bone via flood-fill
    # through tet adjacency. Nearest-anchor by distance fails when cap
    # verts are geometrically closer to the wrong anchor (e.g., Gluteus
    # Maximus origin verts near the insertion end).
    fixed_verts = muscle.get('fixed_vertices', [])
    if len(anchor_bone) > 0 and len(fixed_verts) > len(anchor_bone):
        # Build adjacency among fixed vertices only (cap connectivity)
        fixed_set_local = set(fixed_verts)
        fix_adj = {vi: set() for vi in fixed_verts}
        for t in muscle['tetrahedra']:
            verts_in_fix = [int(v) for v in t if int(v) in fixed_set_local]
            for a in verts_in_fix:
                for b in verts_in_fix:
                    if a != b:
                        fix_adj[a].add(b)

        # BFS from each anchor to assign connected cap verts
        assigned = set(anchor_bone.keys())
        from collections import deque
        for anchor_vi, bname in list(anchor_bone.items()):
            queue = deque([anchor_vi])
            while queue:
                vi = queue.popleft()
                for nb in fix_adj.get(vi, []):
                    if nb not in assigned and nb in fixed_set_local:
                        per_vertex[nb] = [(bname, 1.0)]
                        anchor_bone[nb] = bname
                        assigned.add(nb)
                        queue.append(nb)

        # Any remaining unassigned fixed verts: nearest anchor fallback
        for vi in fixed_verts:
            if vi in per_vertex or vi >= n_verts:
                continue
            best_anchor, best_dist = None, float('inf')
            for avi in anchor_bone:
                d = np.linalg.norm(rest_verts[vi] - rest_verts[avi])
                if d < best_dist:
                    best_dist, best_anchor = d, avi
            if best_anchor is not None:
                per_vertex[vi] = [(anchor_bone[best_anchor], 1.0)]
                anchor_bone[vi] = anchor_bone[best_anchor]

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


def compute_rigid_blend_positions(bindings, skel, blend_coordinate):
    """Pose vertices by SLERP interpolation of the two attachment transforms.

    Linear blend skinning averages rotation matrices and can become singular
    near a strongly rotated joint.  This initializer interpolates the relative
    rigid rotations on SO(3), so the starting volume cannot suffer LBS matrix
    collapse/candy-wrapper artifacts.
    """
    from scipy.spatial.transform import Rotation, Slerp

    bone_data = {}
    rest_positions = np.zeros((len(bindings), 3), dtype=np.float64)
    for vi, (rest, weighted_bones) in enumerate(bindings):
        rest_positions[vi] = rest
        for name, _weight, R0, t0 in weighted_bones:
            bone_data.setdefault(name, (np.asarray(R0), np.asarray(t0)))
    if len(bone_data) != 2:
        return compute_lbs_positions(bindings, skel, len(bindings))

    names = list(bone_data)
    relative_rotations = []
    affine_offsets = []
    for name in names:
        R0, t0 = bone_data[name]
        body = skel.getBodyNode(name)
        if body is None:
            return compute_lbs_positions(bindings, skel, len(bindings))
        transform = body.getWorldTransform()
        R = np.asarray(transform.rotation()) @ R0.T
        t = np.asarray(transform.translation()) - R @ t0
        relative_rotations.append(R)
        affine_offsets.append(t)

    u = np.clip(np.asarray(blend_coordinate, dtype=np.float64), 0.0, 1.0)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(relative_rotations))
    rotations = slerp(u).as_matrix()
    offsets = ((1.0 - u)[:, None] * affine_offsets[0] +
               u[:, None] * affine_offsets[1])
    return np.einsum('mij,mj->mi', rotations, rest_positions) + offsets


def compute_rigid_blend_positions_at_fraction(
        bindings, skel, blend_coordinate, pose_fraction):
    """Evaluate attachment targets along a rigid DART load path.

    Linear interpolation between rest and final *vertex coordinates* turns a
    rotating cap by the affine matrix ``(1-s)I+sR``.  That matrix shrinks and
    can become singular for large rotations.  Instead, interpolate each DART
    body's world rotation on SO(3) and its translation linearly, then retain
    the usual harmonic origin/insertion transform blend across the muscle.
    """
    from scipy.spatial.transform import Rotation, Slerp

    s = float(np.clip(pose_fraction, 0.0, 1.0))
    if s <= 0.0:
        return np.asarray([item[0] for item in bindings], dtype=np.float64)
    if s >= 1.0:
        return compute_rigid_blend_positions(bindings, skel, blend_coordinate)

    bone_data = {}
    rest_positions = np.zeros((len(bindings), 3), dtype=np.float64)
    for vi, (rest, weighted_bones) in enumerate(bindings):
        rest_positions[vi] = rest
        for name, _weight, R0, t0 in weighted_bones:
            bone_data.setdefault(
                name, (np.asarray(R0, dtype=np.float64),
                       np.asarray(t0, dtype=np.float64)))
    if len(bone_data) != 2:
        # This EMU path expects two attachment bodies. Falling back to the
        # endpoint implementation is safer than reintroducing affine chords.
        return compute_rigid_blend_positions(bindings, skel, blend_coordinate)

    names = list(bone_data)
    relative_rotations = []
    affine_offsets = []
    for name in names:
        R0, t0 = bone_data[name]
        body = skel.getBodyNode(name)
        if body is None:
            return compute_rigid_blend_positions(bindings, skel, blend_coordinate)
        transform = body.getWorldTransform()
        R1 = np.asarray(transform.rotation(), dtype=np.float64)
        t1 = np.asarray(transform.translation(), dtype=np.float64)
        R_body = Slerp(
            [0.0, 1.0], Rotation.from_matrix(np.stack((R0, R1))))([s]
        ).as_matrix()[0]
        t_body = (1.0 - s) * t0 + s * t1
        R_relative = R_body @ R0.T
        relative_rotations.append(R_relative)
        affine_offsets.append(t_body - R_relative @ t0)

    u = np.clip(np.asarray(blend_coordinate, dtype=np.float64), 0.0, 1.0)
    rotations = Slerp(
        [0.0, 1.0], Rotation.from_matrix(relative_rotations))(u).as_matrix()
    offsets = ((1.0 - u)[:, None] * affine_offsets[0] +
               u[:, None] * affine_offsets[1])
    return np.einsum('mij,mj->mi', rotations, rest_positions) + offsets


def harmonic_attachment_positions(rest_positions, tetrahedra, fixed_vertices,
                                  fixed_targets):
    """Smoothly extend exact DART attachment motion through a tet volume.

    This is a robust initializer: attachment neighborhoods follow their bones
    exactly, while free-vertex displacement minimizes graph Dirichlet energy.
    It avoids singular matrix blending and spatially varying SLERP distortion.
    """
    rest = np.asarray(rest_positions, dtype=np.float64)
    targets = np.asarray(fixed_targets, dtype=np.float64)
    n = len(rest)
    edge_set = set()
    for tet in np.asarray(tetrahedra, dtype=np.int32):
        for i in range(4):
            for j in range(i + 1, 4):
                edge_set.add(tuple(sorted((int(tet[i]), int(tet[j])))))
    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    weights = 1.0 / np.maximum(lengths, 1e-8)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    values = np.concatenate((weights, weights))
    adjacency = sp.csr_matrix((values, (rows, cols)), shape=(n, n))
    laplacian = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency

    fixed = np.asarray(sorted(set(int(v) for v in fixed_vertices)), dtype=np.int32)
    fixed_set = set(int(v) for v in fixed)
    free = np.asarray([v for v in range(n) if v not in fixed_set], dtype=np.int32)
    displacement = np.zeros_like(rest)
    displacement[fixed] = targets[fixed] - rest[fixed]
    if len(free):
        Lff = laplacian[free][:, free].tocsc()
        Lfc = laplacian[free][:, fixed].tocsc()
        solver = splu(Lff + 1e-10 * sp.eye(len(free), format='csc'))
        rhs = -(Lfc @ displacement[fixed])
        for axis in range(3):
            displacement[free, axis] = solver.solve(rhs[:, axis])
    return rest + displacement


def lame_parameters(youngs, poisson):
    mu = youngs / (2 * (1 + poisson))
    lam = youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))
    return mu, lam


def endpoint_tendon_youngs(muscle, muscle_youngs, tendon_youngs, rings):
    """Build a face-adjacent, geometrically tapered insertion tendon."""
    tets = muscle['tetrahedra']
    insertion = set(int(v) for v in muscle.get(
        'insertion_attachment_vertices', []))
    result = np.full(len(tets), float(muscle_youngs), dtype=np.float64)
    if rings <= 0 or not insertion:
        return result, np.full(len(tets), -1, dtype=np.int32)

    face_to_tets = {}
    for ti, tet in enumerate(tets):
        for opposite in range(4):
            face = tuple(sorted(int(tet[j]) for j in range(4)
                                if j != opposite))
            face_to_tets.setdefault(face, []).append(ti)
    adjacency = [set() for _ in range(len(tets))]
    for incident in face_to_tets.values():
        if len(incident) == 2:
            a, b = incident
            adjacency[a].add(b)
            adjacency[b].add(a)

    layer = np.full(len(tets), -1, dtype=np.int32)
    frontier = [ti for ti, tet in enumerate(tets)
                if any(int(v) in insertion for v in tet)]
    layer[frontier] = 0
    for depth in range(1, rings):
        next_frontier = []
        for ti in frontier:
            for neighbor in adjacency[ti]:
                if layer[neighbor] < 0:
                    layer[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    log_muscle = np.log(float(muscle_youngs))
    log_tendon = np.log(float(tendon_youngs))
    for depth in range(rings):
        # Geometric taper avoids an equally abrupt tendon/muscle interface.
        blend = 1.0 - depth / float(rings)
        result[layer == depth] = np.exp(
            log_muscle + blend * (log_tendon - log_muscle))
    return result, layer


def farthest_attachment_references(vertices, indices, count):
    """Choose deterministic, well-spaced hard references on an endpoint cap."""
    indices = np.asarray(indices, dtype=np.int32)
    if count <= 0 or len(indices) <= count:
        return indices
    points = np.asarray(vertices, dtype=np.float64)[indices]
    center = np.mean(points, axis=0)
    selected = [int(np.argmax(np.linalg.norm(points - center, axis=1)))]
    minimum_distance = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < count:
        next_local = int(np.argmax(minimum_distance))
        selected.append(next_local)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(points - points[next_local], axis=1))
    return indices[np.asarray(selected, dtype=np.int32)]


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


def precompute_emu(vertices, tetrahedra, fixed_vertices, vertex_contour_level,
                   k_modes=48, fiber_directions=None):
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
    # Eq. 13 approximates (G^T G)^-1, whose dominant modes correspond to
    # the *lowest* eigenvalues of G^T G.  Requesting LM here retained local,
    # high-frequency tet modes and produced visibly stair-stepped surfaces.
    # Shift-invert around zero finds the low modes efficiently on this SPD
    # regularized matrix.
    eigenvalues, eigenvectors = eigsh(GtG_csc, k=k, sigma=0.0, which='LM')
    Phi = eigenvectors  # (3n, k)
    Lambda = np.diag(eigenvalues)  # (k, k)

    # B = Φ^T G^T (k × 9m) per Eq. 15
    B_sparse = Phi.T @ Gt  # (k, 9m)
    B = B_sparse if isinstance(B_sparse, np.ndarray) else B_sparse.toarray()
    print(f"    Eigenmodes: k={k}, captured {eigenvalues.sum():.2f} "
          f"({time.time()-t0:.2f}s)")

    # Fiber directions
    if fiber_directions is not None:
        fiber_dirs = np.asarray(fiber_directions, dtype=np.float64)
        if fiber_dirs.shape != (m, 3):
            raise ValueError(
                f"fiber direction shape {fiber_dirs.shape} != {(m, 3)}")
        fiber_norm = np.linalg.norm(fiber_dirs, axis=1, keepdims=True)
        fiber_dirs = fiber_dirs / np.maximum(fiber_norm, 1e-12)
        print(f"    Fibers: loaded {m} volumetric Laplace gradients")
    else:
        fiber_dirs = compute_fiber_directions(
            vertices, tetrahedra, vertex_contour_level)

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
        'Lambda': Lambda,
        'B': B,
        'Dm_inv': Dm_inv,
        'volumes': volumes,
        'fiber_dirs': fiber_dirs,
        'tetrahedra': np.asarray(tetrahedra, dtype=np.int32),
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


def stable_neohookean_energy(F, volumes, mu, lam, active_inverse=None):
    """Stable Neo-Hookean energy, vectorized over all tets.

    Ψ = μ/2(I_C - 3) - μ(J - 1) + λ/2(J - 1)²
    where I_C = tr(F^T F), J = det(F).
    Total energy = sum_i V_i * Ψ_i
    """
    elastic_F = (F if active_inverse is None else
                 np.einsum('mij,mjk->mik', F, active_inverse))
    I_C = np.einsum('mij,mij->m', elastic_F, elastic_F)
    J = np.linalg.det(elastic_F)

    # The polynomial Stable-NH expression is finite for J < 0 unless an
    # explicit inversion guard is applied.  Without this, line search accepted
    # flipped tets because their numerical energy was still finite.
    if np.any(~np.isfinite(J)) or np.any(J <= 0.02):
        return 1e40

    mu = np.asarray(mu, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    psi = mu / 2 * (I_C - 3) - mu * (J - 1) + lam / 2 * (J - 1) ** 2
    return np.sum(volumes * psi)


def stable_neohookean_gradient(F, volumes, mu, lam, active_inverse=None):
    """∂Ψ/∂F for Stable Neo-Hookean, returns (m, 3, 3). Vectorized."""
    elastic_F = (F if active_inverse is None else
                 np.einsum('mij,mjk->mik', F, active_inverse))
    J = np.linalg.det(elastic_F)
    # Batch inverse — handle singular via regularization
    F_reg = elastic_F.copy()
    degen = np.abs(J) < 1e-12
    if np.any(degen):
        F_reg[degen] += 1e-8 * np.eye(3)
    F_invT = np.linalg.inv(F_reg).transpose(0, 2, 1)  # (m, 3, 3)

    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), J.shape)
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), J.shape)
    coeff = (lam * (J - 1) - mu) * J  # (m,)
    P = mu[:, None, None] * elastic_F + coeff[:, None, None] * F_invT
    if active_inverse is not None:
        P = np.einsum('mij,mkj->mik', P, active_inverse)
    return volumes[:, None, None] * P


def relax_positions_with_jacobian_barrier(
        q_init, precomp, fixed_mask, fixed_targets, mu, lam,
        max_iters=20, jacobian_floor=0.02, barrier_start=0.5,
        barrier_scale=5.0, activation=None,
        stretch_frobenius_limit=12.0,
        soft_attachment_indices=None, soft_attachment_targets=None,
        soft_attachment_weight=0.0, optimization_coordinate_scale=6e-4):
    """Exact q-space fallback for poses unsafe in reduced EMU coordinates.

    The reduced deformation-space solve is normally much faster, but at a
    sharply bent joint its finite Woodbury basis can reconstruct a continuous
    mesh with collapsed tets even while every optimized F is valid.  This
    fallback optimizes the actual free vertex coordinates and adds a smooth
    log barrier before J reaches ``jacobian_floor``.  Fixed attachment vertices
    remain exact.
    """
    q = np.asarray(q_init, dtype=np.float64).copy()
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    q[fixed_mask] = np.asarray(fixed_targets, dtype=np.float64)[fixed_mask]
    free_dofs = np.flatnonzero(np.repeat(~fixed_mask, 3))
    q_flat = q.ravel()
    tets = precomp['tetrahedra']
    Dm_inv = precomp['Dm_inv']
    volumes = precomp['volumes']
    mu_arr = np.broadcast_to(np.asarray(mu, dtype=np.float64), (len(tets),))
    lam_arr = np.broadcast_to(np.asarray(lam, dtype=np.float64), (len(tets),))
    activation_arr = None if activation is None else np.broadcast_to(
        np.asarray(activation, dtype=np.float64), (len(tets),))
    beta = barrier_scale * mu_arr
    s0 = max(float(barrier_start - jacobian_floor), 1e-6)
    # Very small attachment tets contributed almost no elastic energy, so the
    # barrier could trade inversion for 10x--30x stretch. Give only excessive
    # distortion a volume-floored quality penalty; ordinary EMU deformation
    # below the threshold is unchanged.
    quality_volume = np.maximum(volumes, np.quantile(volumes, 0.10))
    quality_weight = 0.5 * mu_arr * quality_volume

    initial_F = _deformation_gradients_from_q(q, tets, Dm_inv)
    initial_J = np.linalg.det(initial_F)
    repaired_iterations = 0
    repair_target = max(0.08, barrier_start * 0.25)
    if np.min(initial_J) < repair_target:
        # A rotating attachment cap often leaves one free vertex behind in a
        # 3-fixed/1-free sliver. Move near-barrier elements into the safe
        # interior too; starting L-BFGS at J=0.02--0.04 made every useful trial
        # cross the barrier and falsely converge without moving. Do not
        # volume-weight this repair: tiny slivers are exactly the elements that
        # otherwise get ignored.
        q_repair = q.copy()
        for repair_it in range(20):
            F_repair = _deformation_gradients_from_q(q_repair, tets, Dm_inv)
            J_repair = np.linalg.det(F_repair)
            if np.min(J_repair) >= repair_target:
                repaired_iterations = repair_it
                break
            bad_tets = np.where(J_repair < repair_target)[0]
            if len(bad_tets) == 0:
                break
            # Worst elements first. For one tet J is linear in each vertex;
            # distributing (J_target-J) along its free-vertex gradients is the
            # minimum-norm local correction satisfying the linearized target.
            bad_tets = bad_tets[np.argsort(J_repair[bad_tets])]
            moved = 0
            for ti_idx in bad_tets:
                tet = tets[ti_idx]
                x3 = q_repair[tet[3]]
                Ds = np.stack((q_repair[tet[0]] - x3,
                               q_repair[tet[1]] - x3,
                               q_repair[tet[2]] - x3), axis=-1)
                F_local = Ds @ Dm_inv[ti_idx]
                J_local = float(np.linalg.det(F_local))
                if J_local >= repair_target:
                    continue
                f0, f1, f2 = (F_local[:, 0], F_local[:, 1],
                              F_local[:, 2])
                cof = np.stack((np.cross(f1, f2), np.cross(f2, f0),
                                np.cross(f0, f1)), axis=-1)
                rows = Dm_inv[ti_idx]
                grads = [cof @ rows[c] for c in range(3)]
                grads.append(-(grads[0] + grads[1] + grads[2]))
                denominator = sum(
                    float(np.dot(grads[lv], grads[lv]))
                    for lv in range(4) if not fixed_mask[tet[lv]])
                if denominator < 1e-24:
                    continue
                correction_scale = 0.6 * (repair_target - J_local) / denominator
                for lv in range(4):
                    vi = int(tet[lv])
                    if fixed_mask[vi]:
                        continue
                    delta = correction_scale * grads[lv]
                    length = float(np.linalg.norm(delta))
                    if length > 5e-4:
                        delta *= 5e-4 / length
                    q_repair[vi] += delta
                    moved += 1
            q_repair[fixed_mask] = np.asarray(
                fixed_targets, dtype=np.float64)[fixed_mask]
            if moved == 0:
                break
        q = q_repair
        q_flat = q.ravel()
        initial_F = _deformation_gradients_from_q(q, tets, Dm_inv)
        initial_J = np.linalg.det(initial_F)
        if np.min(initial_J) <= jacobian_floor:
            return None, {
                'success': False,
                'reason': 'fixed constraints could not be untangled',
                'min_j': float(np.min(initial_J)),
                'repair_iterations': int(repaired_iterations),
            }

    def objective(x):
        q_flat[free_dofs] = x
        current = q_flat.reshape(-1, 3)
        F = _deformation_gradients_from_q(current, tets, Dm_inv)
        J = np.linalg.det(F)
        if np.any(~np.isfinite(J)) or np.min(J) <= jacobian_floor:
            return 1e40, np.zeros_like(x)

        active_inverse = precomp.get('active_strain_inverse')
        elastic_F = (F if active_inverse is None else
                     np.einsum('mij,mjk->mik', F, active_inverse))
        elastic_J = np.linalg.det(elastic_F)
        I_C = np.einsum('mij,mij->m', elastic_F, elastic_F)
        psi = (mu_arr / 2.0 * (I_C - 3.0)
               - mu_arr * (elastic_J - 1.0) +
               lam_arr / 2.0 * (elastic_J - 1.0) ** 2)
        energy = float(np.sum(volumes * psi))
        grad_F = stable_neohookean_gradient(
            F, volumes, mu_arr, lam_arr,
            active_inverse=precomp.get('active_strain_inverse'))
        excessive_stretch = np.maximum(
            I_C - float(stretch_frobenius_limit), 0.0)
        if np.any(excessive_stretch > 0.0):
            energy += float(np.sum(
                0.5 * quality_weight * excessive_stretch ** 2))
            grad_F += (
                2.0 * quality_weight * excessive_stretch
            )[:, None, None] * F
        if activation_arr is not None:
            energy += fiber_energy(
                F, precomp['fiber_dirs'], activation_arr, volumes)
            grad_F += fiber_gradient(
                F, precomp['fiber_dirs'], activation_arr, volumes)

        active = J < barrier_start
        if np.any(active):
            s = J[active] - jacobian_floor
            ratio = s / s0
            phi = -np.log(ratio) + ratio - 1.0
            energy += float(np.sum(volumes[active] * beta[active] * phi))
            invT = np.linalg.inv(F[active]).transpose(0, 2, 1)
            cofactor = J[active, None, None] * invT
            dphi = -1.0 / s + 1.0 / s0
            grad_F[active] += (
                volumes[active] * beta[active] * dphi
            )[:, None, None] * cofactor

        grad_q = np.asarray(precomp['Gt'] @ grad_F.ravel()).ravel()
        if (soft_attachment_indices is not None
                and soft_attachment_targets is not None
                and soft_attachment_weight > 0.0):
            attachment_indices = np.asarray(
                soft_attachment_indices, dtype=np.int64)
            attachment_delta = (
                current[attachment_indices]
                - np.asarray(soft_attachment_targets, dtype=np.float64))
            energy += float(
                0.5 * soft_attachment_weight
                * np.sum(attachment_delta * attachment_delta))
            grad_q_view = grad_q.reshape(-1, 3)
            grad_q_view[attachment_indices] += (
                soft_attachment_weight * attachment_delta)
        return energy, grad_q[free_dofs]

    # Optimize in centimeter-scaled coordinates. In raw meters the elastic
    # gradient is large while coordinates are O(1), so L-BFGS proposes a huge
    # first step, backtracks to machine precision, and falsely reports relative
    # convergence without moving a single vertex.
    # Calibrated from the q-gradient magnitude (typically 1e3--1e4): because
    # an L-BFGS identity step in physical coordinates scales as scale²*g,
    # 3e-4 produces an initial displacement around 0.1 mm rather than 0.1 m.
    coordinate_scale = float(optimization_coordinate_scale)
    x0 = q_flat[free_dofs].copy()

    # Feasible steepest-descent startup. Near the determinant barrier SciPy's
    # first quasi-Newton trial can be infeasible at every line-search scale it
    # samples, after which it reports relative-energy convergence with zero
    # motion. A few explicitly bounded physical-space steps establish a real
    # descent direction before L-BFGS builds its inverse-Hessian estimate.
    for _startup in range(5):
        startup_energy, startup_gradient = objective(x0)
        gradient_vertices = startup_gradient.reshape(-1, 3)
        max_gradient = float(np.max(np.linalg.norm(
            gradient_vertices, axis=1))) if len(gradient_vertices) else 0.0
        if not np.isfinite(max_gradient) or max_gradient < 1e-10:
            break
        direction = -startup_gradient
        step = 2e-4 / max_gradient
        accepted_startup = False
        for _backtrack in range(24):
            trial = x0 + step * direction
            trial_energy, _ = objective(trial)
            if (np.isfinite(trial_energy) and
                    trial_energy < startup_energy - 1e-12):
                x0 = trial
                accepted_startup = True
                break
            step *= 0.5
        if not accepted_startup:
            break
    q_flat[free_dofs] = x0

    def scaled_objective(y):
        energy, gradient_x = objective(y * coordinate_scale)
        return energy, gradient_x * coordinate_scale

    result = minimize(
        scaled_objective, x0 / coordinate_scale,
        method='L-BFGS-B', jac=True,
        options={'maxiter': int(max_iters), 'ftol': 1e-16,
                 'gtol': 1e-6, 'maxls': 50})
    q_flat[free_dofs] = result.x * coordinate_scale
    relaxed = q_flat.reshape(-1, 3).copy()
    final_J = np.linalg.det(_deformation_gradients_from_q(
        relaxed, tets, Dm_inv))
    info = {'success': bool(np.min(final_J) > jacobian_floor),
            'iterations': int(result.nit), 'energy': float(result.fun),
            'min_j': float(np.min(final_J)), 'message': str(result.message)}
    info['move_norm'] = float(np.linalg.norm(q_flat[free_dofs] - x0))
    info['gradient_norm'] = float(np.linalg.norm(result.jac))
    info['repair_iterations'] = int(repaired_iterations)
    return relaxed if info['success'] else None, info


def stable_neohookean_hessian_blocks(
        F, volumes, mu, lam, active_inverse=None):
    """Per-tet 9×9 analytical Hessian of V*Ψ_iso w.r.t. vec(F_i). Vectorized.

    Full Stable Neo-Hookean [SGK18] Hessian including all cross-terms
    from ∂²J/∂F². SPD-projected via eigenvalue clamping.

    H[3a+b, 3c+d] = μ δ_{ac}δ_{bd}
                   + coeff1 · g_{ab} · g_{cd}
                   - coeff2 · g_{cb} · g_{ad}
    """
    m = len(F)
    elastic_F = (F if active_inverse is None else
                 np.einsum('mij,mjk->mik', F, active_inverse))
    eps_spd = 1e-6
    J = np.linalg.det(elastic_F)  # (m,)
    V = np.maximum(volumes, 1e-15)  # (m,)

    # Batch inverse
    F_reg = elastic_F.copy()
    degen = np.abs(J) < 1e-12
    if np.any(degen):
        F_reg[degen] += 1e-8 * np.eye(3)
    g = np.linalg.inv(F_reg).transpose(0, 2, 1)  # g[m,a,b] = (F^{-T})_{ab}

    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), J.shape)
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), J.shape)
    c = lam * (J - 1) - mu                    # (m,)
    coeff1 = J * (lam * (2 * J - 1) - mu)     # (m,)
    coeff2 = c * J                             # (m,) note: sign absorbed below

    # Vectorize: build (m, 9, 9) without Python loops
    # g_vec[m, i] = g[m, a, b] where i = 3a+b
    g_vec = g.reshape(m, 9)  # (m, 9)

    # Term 1: μ I  →  (m, 9, 9)
    H = mu[:, None, None] * np.broadcast_to(np.eye(9), (m, 9, 9)).copy()

    # Term 2: coeff1 * g_ab * g_cd = coeff1 * outer(g_vec, g_vec)
    H += coeff1[:, None, None] * (g_vec[:, :, None] * g_vec[:, None, :])

    # Term 3: -coeff2 * g_{cb} * g_{ad}  (fully vectorized, no loops)
    # M[i,j] = g[m, c_j, b_i] * g[m, a_i, d_j]
    # where i=3a+b, j=3c+d
    # g_cb[m,i,j] = g[m, j//3, i%3]  and  g_ad[m,i,j] = g[m, i//3, j%3]
    row_a = np.arange(9) // 3  # (9,)
    row_b = np.arange(9) % 3   # (9,)
    col_c = np.arange(9) // 3  # (9,)
    col_d = np.arange(9) % 3   # (9,)
    # g_cb: shape (m, 9_rows, 9_cols) = g[m, col_c[j], row_b[i]]
    g_cb = g[:, col_c, :][:, :, row_b].transpose(0, 2, 1)  # (m, 9, 9)
    # g_ad: shape (m, 9_rows, 9_cols) = g[m, row_a[i], col_d[j]]
    g_ad = g[:, row_a, :][:, :, col_d]  # (m, 9, 9)
    H -= coeff2[:, None, None] * (g_cb * g_ad)

    # Volume-weight and SPD-project each block
    H = V[:, None, None] * H

    # Batch SPD projection
    eigvals, eigvecs = np.linalg.eigh(H)  # (m, 9), (m, 9, 9)
    eigvals = np.maximum(eigvals, eps_spd)
    H = np.einsum('mij,mj,mkj->mik', eigvecs, eigvals, eigvecs)

    if active_inverse is not None:
        # Row-major vec(F A^-1) = T vec(F), with A^-T repeated for
        # the three spatial rows. Chain rule: H_F = T^T H_e T.
        transform = np.zeros((m, 9, 9), dtype=np.float64)
        block = np.transpose(active_inverse, (0, 2, 1))
        for row in range(3):
            sl = slice(3 * row, 3 * row + 3)
            transform[:, sl, sl] = block
        H = np.einsum('mji,mjk,mkl->mil', transform, H, transform)
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.maximum(eigvals, eps_spd)
        H = np.einsum('mij,mj,mkj->mik', eigvecs, eigvals, eigvecs)

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
def fiber_energy(F, fiber_dirs, activation, volumes):
    """EMU Eq. 8: sum V_i a_i ||F_i u_i||²."""
    if activation is None:
        return 0.0
    activation = np.broadcast_to(np.asarray(activation, dtype=np.float64), (len(F),))
    Fu = np.einsum('mij,mj->mi', F, fiber_dirs)
    return float(np.sum(volumes * activation * np.einsum('mi,mi->m', Fu, Fu)))


def fiber_gradient(F, fiber_dirs, activation, volumes):
    """Gradient of EMU Eq. 8 with respect to each deformation gradient."""
    if activation is None:
        return np.zeros_like(F)
    activation = np.broadcast_to(np.asarray(activation, dtype=np.float64), (len(F),))
    Fu = np.einsum('mij,mj->mi', F, fiber_dirs)
    return (2.0 * volumes * activation)[:, None, None] * np.einsum(
        'mi,mj->mij', Fu, fiber_dirs)


def fiber_hessian_blocks(fiber_dirs, activation, volumes):
    """Constant per-tet Hessian of EMU Eq. 8 in row-major vec(F)."""
    m = len(fiber_dirs)
    if activation is None:
        return np.zeros((m, 9, 9), dtype=np.float64)
    activation = np.broadcast_to(np.asarray(activation, dtype=np.float64), (m,))
    # For each F row, d²||F u||²/dF_row² = 2 uu^T; rows do not couple.
    uu = np.einsum('mi,mj->mij', fiber_dirs, fiber_dirs)
    H = np.zeros((m, 9, 9), dtype=np.float64)
    scale = 2.0 * volumes * activation
    for row in range(3):
        sl = slice(3 * row, 3 * row + 3)
        H[:, sl, sl] = scale[:, None, None] * uu
    return H


def emu_energy(Fvec, precomp, fixed_targets_flat, mu, lam, alpha, use_gpu=True,
               activation=None):
    """Total EMU energy Ψiso + Ψfiber + α EC (paper Eq. 5)."""
    m = precomp['n_tets']
    F = _vec_to_F(Fvec, m)
    if use_gpu and precomp.get('active_strain_inverse') is None:
        E_iso = stable_neohookean_energy_gpu(F, precomp['volumes'], mu, lam)
    else:
        E_iso = stable_neohookean_energy(
            F, precomp['volumes'], mu, lam,
            active_inverse=precomp.get('active_strain_inverse'))
    E_fiber = fiber_energy(
        F, precomp['fiber_dirs'], activation, precomp['volumes'])
    E_c = acap_energy(Fvec, precomp, fixed_targets_flat)
    return E_iso + E_fiber + alpha * E_c


def emu_energy_terms(Fvec, precomp, fixed_targets_flat, mu, lam, alpha,
                     use_gpu=True, activation=None):
    """Return separately named terms of paper Eq. 5 for diagnostics."""
    F = _vec_to_F(Fvec, precomp['n_tets'])
    if use_gpu and precomp.get('active_strain_inverse') is None:
        isotropic = stable_neohookean_energy_gpu(
            F, precomp['volumes'], mu, lam)
    else:
        isotropic = stable_neohookean_energy(
            F, precomp['volumes'], mu, lam,
            active_inverse=precomp.get('active_strain_inverse'))
    fiber = fiber_energy(
        F, precomp['fiber_dirs'], activation, precomp['volumes'])
    acap_raw = acap_energy(Fvec, precomp, fixed_targets_flat)
    return {
        'isotropic': float(isotropic),
        'fiber': float(fiber),
        'acap_raw': float(acap_raw),
        'acap_weighted': float(alpha * acap_raw),
        'total_volume': float(np.sum(precomp['volumes'])),
    }


def emu_gradient(Fvec, precomp, fixed_targets_flat, mu, lam, alpha, use_gpu=True,
                 activation=None):
    """Gradient dE/dF (Eq. 9)."""
    m = precomp['n_tets']
    F = _vec_to_F(Fvec, m)

    if use_gpu and precomp.get('active_strain_inverse') is None:
        P = stable_neohookean_gradient_gpu(F, precomp['volumes'], mu, lam)
    else:
        P = stable_neohookean_gradient(
            F, precomp['volumes'], mu, lam,
            active_inverse=precomp.get('active_strain_inverse'))
    P_fiber = fiber_gradient(
        F, precomp['fiber_dirs'], activation, precomp['volumes'])
    g_iso = _F_to_vec(P + P_fiber)

    # ACAP gradient with constrained solve
    q = acap_positions(Fvec, precomp, fixed_targets_flat)
    Gq = precomp['G'] @ q.ravel()
    g_acap = Fvec - Gq  # (9m,)

    return g_iso + alpha * g_acap


# ---------------------------------------------------------------------------
# Newton step with Woodbury (Eq. 14-17)
# ---------------------------------------------------------------------------
def newton_step_woodbury(gradient, Fvec, precomp, mu, lam, alpha,
                         activation=None):
    """Compute Newton descent direction using Woodbury Hessian (Eq. 14-17).

    Paper notation:
      H = ∂²Ψ_iso/∂F² + αI  (block-diagonal, includes αI per Eq. 15)
      B = Φ^T G^T  (k × 9m, Eq. 15)
      Λ = eigenvalues of G^T G  (k × k)

    Full Hessian (Eq. 14): d²E/dF² ≈ H - αB^T Λ^{-1} B

    Woodbury (Eq. 17):
      (d²E/dF²)^{-1} = H^{-1} + α H^{-1} B^T (Λ - α B H^{-1} B^T)^{-1} B H^{-1}
    """
    m = precomp['n_tets']
    F = _vec_to_F(Fvec, m)

    # H = ∂²Ψ_iso/∂F² + αI  (block-diagonal, m × 9 × 9)
    H_blocks = stable_neohookean_hessian_blocks(
        F, precomp['volumes'], mu, lam,
        active_inverse=precomp.get('active_strain_inverse'))
    H_blocks += fiber_hessian_blocks(
        precomp['fiber_dirs'], activation, precomp['volumes'])
    for i in range(m):
        H_blocks[i] += alpha * np.eye(9)

    # H^{-1} g  (batched block-diagonal solve)
    bad = ~np.isfinite(H_blocks).all(axis=(1, 2))
    if np.any(bad):
        H_blocks[bad] = (alpha + 1e-4) * np.eye(9)
    g_blocks = gradient.reshape(m, 9)
    try:
        Hinv_g = np.linalg.solve(H_blocks, g_blocks[:, :, None]).squeeze(-1)
    except np.linalg.LinAlgError:
        Hinv_g = np.zeros_like(g_blocks)
        for i in range(m):
            Hinv_g[i], _, _, _ = np.linalg.lstsq(H_blocks[i], g_blocks[i], rcond=None)
    Hinv_g_flat = Hinv_g.ravel()

    B = precomp['B']
    Lambda = precomp['Lambda']
    B_Hinv_g = B @ Hinv_g_flat

    BT = B.T
    BT_blocks = BT.reshape(m, 9, -1)
    try:
        Hinv_BT = np.linalg.solve(H_blocks, BT_blocks)
    except np.linalg.LinAlgError:
        Hinv_BT = np.zeros_like(BT_blocks)
        for i in range(m):
            Hinv_BT[i], _, _, _ = np.linalg.lstsq(H_blocks[i], BT_blocks[i], rcond=None)
    Hinv_BT_flat = Hinv_BT.reshape(9 * m, -1)  # (9m, k)
    B_Hinv_BT = B @ Hinv_BT_flat  # (k, k)

    # Middle matrix (Eq. 17): (Λ - α B H^{-1} B^T)^{-1}
    middle = Lambda - alpha * B_Hinv_BT
    # SPD regularization
    eigvals_m, eigvecs_m = np.linalg.eigh(middle)
    eigvals_m = np.maximum(eigvals_m, np.max(np.abs(eigvals_m)) * 1e-6 + 1e-10)
    middle_inv = eigvecs_m @ np.diag(1.0 / eigvals_m) @ eigvecs_m.T

    # Eq. 17: d = -[H^{-1} + α H^{-1} B^T (middle)^{-1} B H^{-1}] g
    correction = alpha * Hinv_BT_flat @ (middle_inv @ B_Hinv_g)  # (9m,)
    direction = -(Hinv_g_flat + correction)

    return direction


# ---------------------------------------------------------------------------
# Collision forces (EMU Section 3.5, Algorithm 1 lines 28-35)
# ---------------------------------------------------------------------------
def compute_collision_forces(q, bone_trimeshes, muscle_surfaces, margin=0.002,
                              neighbor_positions=None, fascia_pairs=None,
                              stiffness=1e7):
    """Detect collisions and compute per-vertex contact forces.

    Includes bone-muscle collision AND inter-muscle fascia proximity.
    Fascia forces pull vertices toward their rest-distance neighbors from
    other muscles, distributed as per-vertex forces so the EMU solver
    propagates them smoothly through the elastic energy.

    Returns (n, 3) force array.
    """
    import trimesh
    from scipy.spatial import cKDTree

    n = len(q)
    forces = np.zeros((n, 3))

    if not bone_trimeshes and not fascia_pairs:
        return forces

    if not bone_trimeshes:
        bone_trimeshes = []

    bone_kdtree = cKDTree(np.vstack([bm.vertices for bm in bone_trimeshes])) if bone_trimeshes else None

    # Bone-muscle collision
    for surf_info in muscle_surfaces:
        if bone_kdtree is None:
            break
        surf_verts = surf_info['surf_verts']
        fixed_set = surf_info['fixed_set']
        exempt_set = surf_info.get('collision_exempt_set', set())
        offset = surf_info['offset']

        sv_pos = q[np.array(surf_verts) + offset]
        dists, _ = bone_kdtree.query(sv_pos)
        near = dists < 0.03

        if not np.any(near):
            continue

        near_sv = np.array(surf_verts)[near]
        near_pos = q[near_sv + offset]

        for bone_mesh in bone_trimeshes:
            try:
                bmin = bone_mesh.bounds[0] - 0.03
                bmax = bone_mesh.bounds[1] + 0.03
                in_bbox = np.all((near_pos >= bmin) & (near_pos <= bmax), axis=1)
                if not np.any(in_bbox):
                    continue
                bbox_pos = near_pos[in_bbox]
                bbox_sv = near_sv[in_bbox]
                inside = bone_mesh.contains(bbox_pos)
                if not np.any(inside):
                    continue
                inside_pos = bbox_pos[inside]
                inside_sv = bbox_sv[inside]
                closest, _, face_ids = trimesh.proximity.closest_point(
                    bone_mesh, inside_pos)
                for k in range(len(inside_sv)):
                    vi = int(inside_sv[k]) + offset
                    if ((vi - offset) in fixed_set or
                            (vi - offset) in exempt_set):
                        continue
                    # closest - inside points toward the exterior regardless
                    # of triangle winding; face normals are not reliable for
                    # all imported Zygote OBJ files.
                    outward = closest[k] - inside_pos[k]
                    outward_len = np.linalg.norm(outward)
                    if outward_len > 1e-12:
                        outward /= outward_len
                    else:
                        outward = bone_mesh.face_normals[face_ids[k]]
                    target = closest[k] + outward * margin
                    # Spring-like force toward target
                    forces[vi] = (target - q[vi]) * stiffness
            except Exception:
                continue

    # Muscle-muscle collision
    all_surf_pts = []
    all_surf_info = []
    for si, surf_info in enumerate(muscle_surfaces):
        offset = surf_info['offset']
        for vi in surf_info['surf_verts']:
            gvi = vi + offset
            all_surf_pts.append(q[gvi])
            all_surf_info.append((si, gvi))

    if len(all_surf_pts) > 1:
        pts = np.array(all_surf_pts)
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=margin * 2)
        for i, j in pairs:
            si_i, gvi_i = all_surf_info[i]
            si_j, gvi_j = all_surf_info[j]
            if si_i == si_j:
                continue
            diff = q[gvi_j] - q[gvi_i]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            push = (margin * 2 - dist) * 0.5 * diff / dist * stiffness
            forces[gvi_i] -= push
            forces[gvi_j] += push

    # Fascia proximity: pull vertices toward rest-distance neighbors
    # These forces feed into EMU's Newton iteration, so the solver
    # distributes them smoothly through the elastic energy (whole muscle moves)
    if fascia_pairs and neighbor_positions is not None:
        names_map = neighbor_positions.get('_names', {})
        # Determine which muscle index we're solving (from muscle_surfaces)
        my_surf_verts = set()
        my_fixed = set()
        for surf_info in muscle_surfaces:
            my_surf_verts.update(surf_info['surf_verts'])
            my_fixed.update(surf_info['fixed_set'])

        for mi_idx, vi, mj_idx, vj, rest_dist in fascia_pairs:
            # Determine which vertex is ours and which is the neighbor
            if vi in my_surf_verts and vi not in my_fixed:
                other_name = names_map.get(mj_idx)
                if other_name and other_name in neighbor_positions:
                    other_pos = neighbor_positions[other_name][vj]
                    diff = other_pos - q[vi]
                    dist = np.linalg.norm(diff)
                    if dist > rest_dist and dist > 1e-10:
                        excess = dist - rest_dist
                        forces[vi] += (diff / dist) * excess * 1e3
            elif vj in my_surf_verts and vj not in my_fixed:
                other_name = names_map.get(mi_idx)
                if other_name and other_name in neighbor_positions:
                    other_pos = neighbor_positions[other_name][vi]
                    diff = other_pos - q[vj]
                    dist = np.linalg.norm(diff)
                    if dist > rest_dist and dist > 1e-10:
                        excess = dist - rest_dist
                        forces[vj] += (diff / dist) * excess * 1e3

    return forces


def generalized_force_on_F(forces_q, precomp, fixed_targets_flat):
    """Convert per-vertex forces to per-tet generalized forces on F (Eq. 22).

    f_ext on F = F^T (G^TG)^{-1} G^T)^T f_q  (adapted from Eq. 22)
    Simplified: propagate vertex forces through G^T.
    """
    f_flat = forces_q.ravel()  # (3n,)
    # The generalized force on F from vertex forces:
    # Since q = (G^TG)^{-1} G^T F, forces on q map to F via:
    # f_F = G (G^TG)^{-1} f_q  (adjoint of the ACAP mapping)
    n = precomp['n_verts']
    free_dofs = precomp['free_dofs']
    fixed_dofs = precomp['fixed_dofs']

    # Only free DOF forces matter (fixed vertices don't move)
    f_free = f_flat[free_dofs]
    q_force_free = precomp['acap_solver'].solve(f_free)

    q_force_flat = np.zeros(3 * n)
    q_force_flat[free_dofs] = q_force_free

    f_F = precomp['G'] @ q_force_flat  # (9m,)
    return f_F


def project_out_of_bones(q, tetrahedra, bone_trimeshes, surface_vertices,
                         fixed_vertices, margin=0.002, passes=3):
    """Resolve residual bone penetration with a harmonic displacement field.

    Contact vertices receive exterior targets, attachment pins receive zero
    displacement, and all remaining vertices solve a graph-Laplacian Dirichlet
    problem.  Unlike independent per-vertex projection, this propagates contact
    smoothly through neighboring tetrahedra and does not create surface stairs.
    """
    import trimesh
    positions = np.asarray(q, dtype=np.float64).copy()
    candidates = np.asarray(surface_vertices, dtype=np.int32)
    fixed = set(int(v) for v in fixed_vertices)
    moved = set()

    n = len(positions)
    edge_set = set()
    for tet in np.asarray(tetrahedra, dtype=np.int32):
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = sorted((int(tet[i]), int(tet[j])))
                edge_set.add((a, b))
    if not edge_set:
        return positions, 0
    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sp.diags(degree) - adjacency

    for _ in range(max(int(passes), 1)):
        targets = {}
        for bone in bone_trimeshes or []:
            pts = positions[candidates]
            in_bbox = np.all(
                (pts >= bone.bounds[0] - margin) &
                (pts <= bone.bounds[1] + margin), axis=1)
            if not np.any(in_bbox):
                continue
            ids = candidates[in_bbox]
            pts = positions[ids]
            try:
                inside = bone.contains(pts)
                if not np.any(inside):
                    continue
                ids = ids[inside]
                pts = pts[inside]
                closest, _, face_ids = trimesh.proximity.closest_point(bone, pts)
                for j, vi in enumerate(ids):
                    vi = int(vi)
                    if vi in fixed:
                        continue
                    outward = closest[j] - pts[j]
                    length = np.linalg.norm(outward)
                    if length > 1e-12:
                        outward /= length
                    else:
                        outward = bone.face_normals[face_ids[j]]
                    target = closest[j] + outward * margin
                    delta = target - positions[vi]
                    # If overlapping bones constrain one vertex, retain the
                    # correction with the greatest penetration depth.
                    if vi not in targets or np.linalg.norm(delta) > np.linalg.norm(targets[vi]):
                        targets[vi] = delta
            except Exception:
                continue
        if not targets:
            break

        constrained = sorted(fixed | set(targets))
        constrained_set = set(constrained)
        free = np.asarray([i for i in range(n) if i not in constrained_set], dtype=np.int32)
        constrained = np.asarray(constrained, dtype=np.int32)
        prescribed = np.zeros((len(constrained), 3), dtype=np.float64)
        constrained_lookup = {int(vi): i for i, vi in enumerate(constrained)}
        for vi, delta in targets.items():
            prescribed[constrained_lookup[vi]] = delta

        displacement = np.zeros_like(positions)
        displacement[constrained] = prescribed
        if len(free):
            Lff = laplacian[free][:, free].tocsc()
            Lfc = laplacian[free][:, constrained].tocsc()
            # Small diagonal handles isolated numerical components safely.
            solver = splu(Lff + 1e-10 * sp.eye(len(free), format='csc'))
            rhs = -(Lfc @ prescribed)
            for axis in range(3):
                displacement[free, axis] = solver.solve(rhs[:, axis])
        positions += displacement
        moved.update(targets)
    return positions, len(moved)


def project_embedded_skin_out_of_bones(
        q, precomp, bone_trimeshes, skin_rest, skin_indices, skin_weights,
        tet_rest, fixed_vertices, exempt_tet_vertices=None,
        exempt_skin_vertices=None, margin=0.002, passes=3):
    """Resolve collisions visible on an anatomical skin embedded in tets.

    The render shell may not coincide with the regularized simulation boundary.
    Contact displacements detected on that shell are mapped through its tet
    barycentrics, then harmonically propagated through the simulation mesh.
    """
    import trimesh

    positions = np.asarray(q, dtype=np.float64).copy()
    skin_rest = np.asarray(skin_rest, dtype=np.float64)
    skin_indices = np.asarray(skin_indices, dtype=np.int32)
    skin_weights = np.asarray(skin_weights, dtype=np.float64)
    tet_rest = np.asarray(tet_rest, dtype=np.float64)
    fixed = set(int(v) for v in fixed_vertices)
    exempt_values = set() if exempt_tet_vertices is None else exempt_tet_vertices
    exempt = set(int(v) for v in exempt_values)
    skin_exempt_values = (
        set() if exempt_skin_vertices is None else exempt_skin_vertices)
    skin_exempt = set(int(v) for v in skin_exempt_values)
    dominant = skin_indices[
        np.arange(len(skin_indices)), np.argmax(np.abs(skin_weights), axis=1)]
    eligible_skin = ~np.isin(dominant, np.asarray(sorted(fixed | exempt), dtype=np.int32))
    if skin_exempt:
        eligible_skin[np.asarray(sorted(skin_exempt), dtype=np.int32)] = False

    n = len(positions)
    edge_set = set()
    for tet in np.asarray(precomp['tetrahedra'], dtype=np.int32):
        for i in range(4):
            for j in range(i + 1, 4):
                edge_set.add(tuple(sorted((int(tet[i]), int(tet[j])))))
    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    laplacian = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency

    def skin_positions(current):
        displacement = current[skin_indices] - tet_rest[skin_indices]
        return skin_rest + np.einsum('nk,nkj->nj', skin_weights, displacement)

    moved_skin = set()
    for _ in range(max(1, int(passes))):
        skin = skin_positions(positions)
        skin_targets = {}
        for bone in bone_trimeshes or []:
            in_bbox = eligible_skin & np.all(
                (skin >= bone.bounds[0] - margin) &
                (skin <= bone.bounds[1] + margin), axis=1)
            ids = np.where(in_bbox)[0]
            if len(ids) == 0:
                continue
            try:
                inside = bone.contains(skin[ids])
                ids = ids[inside]
                if len(ids) == 0:
                    continue
                points = skin[ids]
                closest, _, face_ids = trimesh.proximity.closest_point(bone, points)
                for local, si in enumerate(ids):
                    outward = closest[local] - points[local]
                    length = float(np.linalg.norm(outward))
                    if length > 1e-12:
                        outward /= length
                    else:
                        outward = bone.face_normals[face_ids[local]]
                    delta = closest[local] + margin * outward - points[local]
                    if (si not in skin_targets or
                            np.linalg.norm(delta) > np.linalg.norm(skin_targets[si])):
                        skin_targets[int(si)] = delta
            except Exception:
                continue
        if not skin_targets:
            break

        accum = {}
        accum_weight = {}
        for si, delta in skin_targets.items():
            weights = skin_weights[si]
            movable = np.asarray([
                int(vi) not in fixed and int(vi) not in exempt
                for vi in skin_indices[si]], dtype=bool)
            denom = float(np.sum(weights[movable] ** 2))
            if denom < 1e-16:
                continue
            for local in np.where(movable)[0]:
                vi = int(skin_indices[si, local])
                contribution = (weights[local] / denom) * delta
                accum[vi] = accum.get(vi, np.zeros(3)) + contribution
                accum_weight[vi] = accum_weight.get(vi, 0) + 1
            moved_skin.add(int(si))
        if not accum:
            break

        targets = {vi: delta / accum_weight[vi] for vi, delta in accum.items()}
        constrained_ids = sorted(fixed | exempt | set(targets))
        constrained_set = set(constrained_ids)
        free = np.asarray([i for i in range(n) if i not in constrained_set], dtype=np.int32)
        constrained = np.asarray(constrained_ids, dtype=np.int32)
        prescribed = np.zeros((len(constrained), 3), dtype=np.float64)
        lookup = {int(vi): i for i, vi in enumerate(constrained)}
        for vi, delta in targets.items():
            prescribed[lookup[vi]] = delta
        displacement = np.zeros_like(positions)
        displacement[constrained] = prescribed
        if len(free):
            solver = splu(
                laplacian[free][:, free].tocsc() +
                1e-10 * sp.eye(len(free), format='csc'))
            rhs = -(laplacian[free][:, constrained] @ prescribed)
            for axis in range(3):
                displacement[free, axis] = solver.solve(rhs[:, axis])

        scale = 1.0
        accepted = False
        for _line_search in range(16):
            candidate = positions + scale * displacement
            F = _deformation_gradients_from_q(
                candidate, precomp['tetrahedra'], precomp['Dm_inv'])
            if np.all(np.isfinite(F)) and np.min(np.linalg.det(F)) > 0.02:
                positions = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break

    residual_skin = set()
    final_skin = skin_positions(positions)
    residual_offsets = np.zeros_like(final_skin)
    for bone in bone_trimeshes or []:
        ids = np.where(eligible_skin & np.all(
            (final_skin >= bone.bounds[0] - margin) &
            (final_skin <= bone.bounds[1] + margin), axis=1))[0]
        if len(ids):
            try:
                inside = bone.contains(final_skin[ids])
                inside_ids = ids[inside]
                residual_skin.update(int(i) for i in inside_ids)
                if len(inside_ids):
                    points = final_skin[inside_ids]
                    closest, _, face_ids = trimesh.proximity.closest_point(
                        bone, points)
                    for local, si in enumerate(inside_ids):
                        outward = closest[local] - points[local]
                        length = float(np.linalg.norm(outward))
                        if length > 1e-12:
                            outward /= length
                        else:
                            outward = bone.face_normals[face_ids[local]]
                        delta = closest[local] + margin * outward - points[local]
                        if np.linalg.norm(delta) > np.linalg.norm(residual_offsets[si]):
                            residual_offsets[si] = delta
            except Exception:
                pass
    return positions, len(moved_skin), len(residual_skin), residual_offsets


# ---------------------------------------------------------------------------
# EMU solver (Algorithm 1)
# ---------------------------------------------------------------------------
def emu_solve(q_init, precomp, fixed_mask, fixed_targets,
              mu, lam, alpha, max_iters=30, verbose=False,
              bone_trimeshes=None, muscle_surfaces=None, margin=0.002,
              use_gpu=True, warm_F=None,
              neighbor_positions=None, fascia_pairs=None, activation=None,
              collision_stiffness=1e7, collision_iterations=1):
    """Run EMU quasi-Newton solver (Algorithm 1 from the paper).

    Optimizations:
    - #2 Early termination when energy change < threshold
    - #3 warm_F: carry forward F from previous frame
    - #4 Hessian reuse: recompute every 3 iterations
    - #7 Cache ACAP solve within iteration (gradient+energy share)
    """
    m = precomp['n_tets']
    n = precomp['n_verts']

    fixed_targets_flat = fixed_targets.ravel()

    # #3: Warm-start from previous frame's F or compute from q_init
    if warm_F is not None and len(warm_F) == 9 * m:
        Fvec = warm_F.copy()
    else:
        Fvec = (precomp['G'] @ q_init.ravel()).copy()

    # Algorithm 1 parameters
    sigma_init = 10.0
    rho = 0.5
    eps = 1e-3
    e1 = 1e-4
    # EMU evaluates convergence using an absolute energy change below 1e-2
    # (paper Sec. 4), which is also a better scale for SI material energies.
    e2 = 1e-2

    heterogeneous = np.ndim(mu) > 0 or np.ndim(lam) > 0
    active_strain = precomp.get('active_strain_inverse') is not None
    if use_gpu and (heterogeneous or activation is not None or active_strain):
        print("    EMU GPU kernels do not support heterogeneous/fiber material yet; using CPU")
        use_gpu = False

    E = emu_energy(Fvec, precomp, fixed_targets_flat, mu, lam, alpha,
                   use_gpu=use_gpu, activation=activation)
    if verbose:
        initial_terms = emu_energy_terms(
            Fvec, precomp, fixed_targets_flat, mu, lam, alpha,
            use_gpu=use_gpu, activation=activation)
        print(f"    EMU iter 0: E={E:.6e} "
              f"(iso={initial_terms['isotropic']:.3e}, "
              f"fiber={initial_terms['fiber']:.3e}, "
              f"alpha*ACAP={initial_terms['acap_weighted']:.3e})")

    g_norm = float('inf')
    n_coll = 0
    max_coll = 0
    collision_history = []
    H_cached = None  # #4: cached Hessian blocks

    for iteration in range(max_iters):
        # Compute gradient
        g = emu_gradient(Fvec, precomp, fixed_targets_flat, mu, lam, alpha,
                         use_gpu=use_gpu, activation=activation)
        g_norm = np.linalg.norm(g)

        # #4: Recompute Hessian every 3 iterations (or first iteration)
        recompute_H = (iteration % 3 == 0) or H_cached is None
        if recompute_H:
            if use_gpu:
                d = newton_step_woodbury_gpu(g, Fvec, precomp, mu, lam, alpha)
            else:
                d = newton_step_woodbury(
                    g, Fvec, precomp, mu, lam, alpha, activation=activation)
            # Cache the Hessian blocks for reuse
            F_mat = _vec_to_F(Fvec, m)
            if use_gpu:
                H_cached = np.zeros((m, 9, 9), dtype=np.float64)
                _ti_hessian_build(F_mat.reshape(m, 9).copy(), precomp['volumes'], mu, lam, 0.0, H_cached)
            else:
                H_cached = stable_neohookean_hessian_blocks(
                    F_mat, precomp['volumes'], mu, lam,
                    active_inverse=precomp.get('active_strain_inverse'))
                H_cached += fiber_hessian_blocks(
                    precomp['fiber_dirs'], activation, precomp['volumes'])
            eigv, eigvc = np.linalg.eigh(H_cached)
            eigv = np.maximum(eigv, 1e-6)
            H_cached = np.einsum('mij,mj,mkj->mik', eigvc, eigv, eigvc)
            H_cached += alpha * np.eye(9)[None, :, :]
        else:
            # Reuse cached H for cheaper Newton step
            H_reg = H_cached.copy()
            g_blocks = g.reshape(m, 9)
            try:
                Hinv_g = np.linalg.solve(H_reg, g_blocks[:, :, None]).squeeze(-1)
            except np.linalg.LinAlgError:
                Hinv_g = np.zeros_like(g_blocks)
                for bi in range(m):
                    Hinv_g[bi], _, _, _ = np.linalg.lstsq(H_reg[bi], g_blocks[bi], rcond=None)
            Hinv_g_flat = Hinv_g.ravel()

            B = precomp['B']
            Lambda = precomp['Lambda']
            B_Hinv_g = B @ Hinv_g_flat
            BT = B.T.reshape(m, 9, -1)
            try:
                Hinv_BT = np.linalg.solve(H_reg, BT)
            except np.linalg.LinAlgError:
                Hinv_BT = np.zeros_like(BT)
                for bi in range(m):
                    Hinv_BT[bi], _, _, _ = np.linalg.lstsq(H_reg[bi], BT[bi], rcond=None)
            Hinv_BT_flat = Hinv_BT.reshape(9 * m, -1)
            B_Hinv_BT = B @ Hinv_BT_flat
            middle = Lambda - alpha * B_Hinv_BT
            eigvals_m, eigvecs_m = np.linalg.eigh(middle)
            eigvals_m = np.maximum(eigvals_m, np.max(np.abs(eigvals_m)) * 1e-6 + 1e-10)
            middle_inv = eigvecs_m @ np.diag(1.0 / eigvals_m) @ eigvecs_m.T
            correction = alpha * Hinv_BT_flat @ (middle_inv @ B_Hinv_g)
            d = -(Hinv_g_flat + correction)

        # Check descent direction
        dir_deriv = np.dot(g, d)
        if dir_deriv > 0:
            d = -g
            dir_deriv = np.dot(g, d)

        # Backtracking line search
        sigma = sigma_init
        E_prev = E
        ls_success = False
        for ls_iter in range(20):
            F_new = Fvec + sigma * d
            E_new = emu_energy(
                F_new, precomp, fixed_targets_flat, mu, lam, alpha,
                use_gpu=use_gpu, activation=activation)
            if np.isfinite(E_new) and E_new <= E_prev + eps * sigma * dir_deriv:
                # F is discontinuous per tet; valid det(F) does not guarantee
                # that the ACAP-reconstructed continuous mesh is valid. Check
                # the actual displayed/simulated tet positions before accepting.
                q_new = acap_positions(F_new, precomp, fixed_targets_flat)
                q_new_F = _deformation_gradients_from_q(
                    q_new, precomp['tetrahedra'], precomp['Dm_inv'])
                q_new_J = np.linalg.det(q_new_F)
                if (np.all(np.isfinite(q_new_J)) and
                        np.min(q_new_J) > 0.02):
                    ls_success = True
                    break
            sigma *= rho

        if ls_success:
            Fvec = F_new
            E = E_new

        # Collision resolution (inner loop)
        if bone_trimeshes is not None or muscle_surfaces is not None:
            for coll_iter in range(max(1, int(collision_iterations))):
                q_current = acap_positions(Fvec, precomp, fixed_targets_flat)
                f_coll = compute_collision_forces(
                    q_current, bone_trimeshes or [], muscle_surfaces or [], margin,
                    stiffness=collision_stiffness)
                n_coll = int(np.sum(np.linalg.norm(f_coll, axis=1) > 1e-10))
                max_coll = max(max_coll, n_coll)
                collision_history.append(n_coll)
                if n_coll == 0:
                    break
                g_ext = generalized_force_on_F(f_coll, precomp, fixed_targets_flat)
                g_ext_blocks = g_ext.reshape(m, 9)
                try:
                    d_contact = np.linalg.solve(H_cached, g_ext_blocks[:, :, None]).squeeze(-1)
                    contact_step = d_contact.ravel()
                    accepted_contact = False
                    scale = 1.0
                    for _contact_ls in range(16):
                        candidate = Fvec + scale * contact_step
                        candidate_J = np.linalg.det(_vec_to_F(candidate, m))
                        if (np.all(np.isfinite(candidate_J)) and
                                np.min(candidate_J) > 0.02):
                            candidate_q = acap_positions(
                                candidate, precomp, fixed_targets_flat)
                            candidate_q_F = _deformation_gradients_from_q(
                                candidate_q, precomp['tetrahedra'],
                                precomp['Dm_inv'])
                            candidate_q_J = np.linalg.det(candidate_q_F)
                            if (np.all(np.isfinite(candidate_q_J)) and
                                    np.min(candidate_q_J) > 0.02):
                                Fvec = candidate
                                accepted_contact = True
                                break
                        scale *= 0.5
                    if not accepted_contact:
                        break
                except np.linalg.LinAlgError:
                    break  # Hessian singular, skip remaining collision iters

            E = emu_energy(
                Fvec, precomp, fixed_targets_flat, mu, lam, alpha,
                use_gpu=use_gpu, activation=activation)

        if verbose and (iteration + 1) % 5 == 0:
            print(f"    EMU iter {iteration+1}: E={E:.6e}, |g|={g_norm:.2e}, "
                  f"σ={sigma:.4f}, coll={n_coll}")

        # #2: Early termination
        if g_norm < e1:
            break
        if ls_success and abs(E_prev - E) < e2 and n_coll == 0:
            break

    q = acap_positions(Fvec, precomp, fixed_targets_flat)

    projected = 0
    if bone_trimeshes and muscle_surfaces:
        q_before_projection = q.copy()
        surface_vertices = sorted({
            int(vi) + int(info.get('offset', 0))
            for info in muscle_surfaces for vi in info.get('surf_verts', [])
        })
        projection_fixed = set(np.where(np.asarray(fixed_mask, dtype=bool))[0])
        for surface_info in muscle_surfaces:
            offset = int(surface_info.get('offset', 0))
            projection_fixed.update(
                int(vi) + offset
                for vi in surface_info.get('collision_exempt_set', set()))
        projected_q, projected = project_out_of_bones(
            q, precomp['tetrahedra'], bone_trimeshes, surface_vertices,
            projection_fixed,
            margin=margin, passes=3)
        # Harmonic contact is smooth, but a large correction can still invert
        # a thin attachment tet. Backtrack only the projection displacement.
        projection_delta = projected_q - q_before_projection
        projection_scale = 1.0
        accepted_projection = False
        for _projection_ls in range(16):
            candidate_q = q_before_projection + projection_scale * projection_delta
            candidate_F = _deformation_gradients_from_q(
                candidate_q, precomp['tetrahedra'], precomp['Dm_inv'])
            candidate_J = np.linalg.det(candidate_F)
            if (np.all(np.isfinite(candidate_J)) and
                    np.min(candidate_J) > 0.02):
                q = candidate_q
                accepted_projection = True
                break
            projection_scale *= 0.5
        if not accepted_projection:
            q = q_before_projection
            projected = 0

    if verbose:
        final_terms = emu_energy_terms(
            Fvec, precomp, fixed_targets_flat, mu, lam, alpha,
            use_gpu=use_gpu, activation=activation)
        print(f"    EMU done: E={E:.6e}, iters={iteration+1}, "
              f"|g|={g_norm:.2e}, coll={n_coll}; "
              f"iso={final_terms['isotropic']:.3e}, "
              f"fiber={final_terms['fiber']:.3e}, "
              f"alpha*ACAP={final_terms['acap_weighted']:.3e}")
    else:
        final_terms = emu_energy_terms(
            Fvec, precomp, fixed_targets_flat, mu, lam, alpha,
            use_gpu=use_gpu, activation=activation)

    info = {'iterations': iteration + 1, 'energy': E, 'grad_norm': g_norm,
            'collisions': max_coll,
            'final_collisions': collision_history[-1] if collision_history else 0,
            'projected_collisions': projected,
            'collision_history': collision_history,
            'energy_terms': final_terms,
            'Fvec': Fvec}  # #3: return F for warm-start
    return q, info


# ---------------------------------------------------------------------------
# Fascia proximity constraints (inter-muscle attachment)
# ---------------------------------------------------------------------------
def compute_fascia_pairs(muscles, threshold=0.005):
    """Find pairs of surface vertices from different muscles within threshold.

    Returns list of (muscle_i, vert_i, muscle_j, vert_j, rest_dist).
    Only pairs where both vertices are free (not fixed).
    """
    from scipy.spatial import cKDTree
    pairs = []
    for i in range(len(muscles)):
        mi = muscles[i]
        sv_i = np.array(mi['_surf_verts'])
        pos_i = mi['rest_vertices'][sv_i]
        fixed_i = set(mi['fixed_vertices'])
        tree_i = cKDTree(pos_i)

        for j in range(i + 1, len(muscles)):
            mj = muscles[j]
            sv_j = np.array(mj['_surf_verts'])
            pos_j = mj['rest_vertices'][sv_j]
            fixed_j = set(mj['fixed_vertices'])

            # Query pairs within threshold
            nearby = tree_i.query_ball_point(pos_j, threshold)
            for jj, near_list in enumerate(nearby):
                vj = sv_j[jj]
                if vj in fixed_j:
                    continue
                for ii_idx in near_list:
                    vi = sv_i[ii_idx]
                    if vi in fixed_i:
                        continue
                    rest_dist = np.linalg.norm(pos_i[ii_idx] - pos_j[jj])
                    pairs.append((i, int(vi), j, int(vj), rest_dist))
    return pairs


def apply_fascia_constraints(bake_buffers, muscles, fascia_pairs, frame,
                              stiffness=0.5, max_iters=5):
    """Pull neighboring muscle surface vertices together when they drift apart.

    One-sided: only penalize when distance EXCEEDS rest distance (fascia
    stretching). Distance decreasing (sliding/compression) is free.
    Modifies bake_buffers in-place.
    """
    if not fascia_pairs:
        return 0

    # Collect current positions
    positions = {}
    for m in muscles:
        if frame in bake_buffers.get(m['name'], {}):
            positions[m['name']] = bake_buffers[m['name']][frame].copy()

    n_corrected = 0
    for _ in range(max_iters):
        iter_corrected = 0
        for mi_idx, vi, mj_idx, vj, rest_dist in fascia_pairs:
            mi_name = muscles[mi_idx]['name']
            mj_name = muscles[mj_idx]['name']
            if mi_name not in positions or mj_name not in positions:
                continue

            pi = positions[mi_name][vi]
            pj = positions[mj_name][vj]
            diff = pj - pi
            dist = np.linalg.norm(diff)

            if dist < 1e-10 or dist <= rest_dist:
                continue  # not stretched — no correction

            # Pull both vertices toward each other
            excess = dist - rest_dist
            direction = diff / dist
            correction = stiffness * excess * 0.5 * direction
            positions[mi_name][vi] += correction
            positions[mj_name][vj] -= correction
            iter_corrected += 1

        n_corrected += iter_corrected
        if iter_corrected == 0:
            break

    # Write back
    for m in muscles:
        if m['name'] in positions:
            bake_buffers[m['name']][frame] = positions[m['name']]

    return n_corrected


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------
_parallel_ctx = {}  # shared state for forked workers

def _solve_one_worker(idx):
    """Worker function for parallel EMU solve. Uses shared _parallel_ctx (fork)."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    ctx = _parallel_ctx
    mi, m, q_init, lbs_pos, fixed_mask = ctx['tasks'][idx]
    warm_F = ctx.get('warm_F', {}).get(m['name'])

    # Build neighbor_positions for fascia from ARAP/init positions of all muscles
    neighbor_positions = ctx.get('neighbor_positions', None)
    fascia_pairs_for_muscle = ctx.get('fascia_pairs_for_muscle', {}).get(mi, [])

    q, info = emu_solve(
        q_init, m['emu'], fixed_mask, lbs_pos,
        ctx['mu'], ctx['lam'], ctx['alpha'],
        max_iters=ctx['max_iters'],
        verbose=False,
        bone_trimeshes=ctx['bone_tms'],
        muscle_surfaces=[ctx['muscle_surfaces'][mi]],
        margin=0.002,
        use_gpu=False,
        warm_F=warm_F,
        neighbor_positions=neighbor_positions,
        fascia_pairs=fascia_pairs_for_muscle)
    return m['name'], q.astype(np.float32), info.get('Fvec')


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
    parser.add_argument(
        '--muscles', default='',
        help='Comma-separated exact tet muscle names. When supplied, bypasses '
             'the built-in side/up-leg list; for example '
             'L_Adductor_Longus_Subdivided.')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=None)
    parser.add_argument('--tet-dir', default='tet')
    parser.add_argument('--output-dir', default='data/motion_cache')
    parser.add_argument('--youngs', type=float, default=6e6,
                        help='Young\'s modulus (Pa, default: 6e6 for muscle)')
    parser.add_argument('--poisson', type=float, default=0.49)
    parser.add_argument('--insertion-tendon-rings', type=int, default=0,
                        help='Face-adjacent free tet layers forming a stiff, '
                             'geometrically tapered insertion tendon.')
    parser.add_argument('--tendon-youngs', type=float, default=4.5e8)
    parser.add_argument('--activation-max', type=float, default=0.0,
                        help='Maximum EMU fiber activation coefficient (Pa).')
    parser.add_argument('--activation-full-shortening', type=float, default=0.20,
                        help='Origin/insertion span shortening fraction that '
                             'reaches --activation-max.')
    parser.add_argument('--active-strain-from-span', action='store_true',
                        help='Use det-one fiber active strain derived from the '
                             'current attachment-span ratio instead of linear '
                             'EMU fiber-force activation.')
    parser.add_argument('--minimum-active-stretch', type=float, default=0.70)
    parser.add_argument('--repair-invalid-q', action='store_true',
                        help='Run the exact q-space Jacobian-barrier fallback '
                             'and warm-start from its continuous deformation.')
    parser.add_argument('--attachment-reference-count', type=int, default=0,
                        help='If positive, hard-fix only this many well-spaced '
                             'vertices on each endpoint cap; the rest deform '
                             'through the tendon material.')
    parser.add_argument('--q-refine-every-frame', action='store_true',
                        help='Run the exact near-incompressible q-space '
                             'material refinement after every EMU solve.')
    parser.add_argument('--soft-cap-weight', type=float, default=0.0,
                        help='q-space spring stiffness (N/m) for all authored '
                             'cap vertices not selected as hard references.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='ACAP continuity weight (paper: start at 1, increase until Newton iters spike)')
    parser.add_argument('--max-iters', type=int, default=30,
                        help='Newton iterations per frame')
    parser.add_argument('--k-modes', type=int, default=32,
                        help='Eigenmodes for Woodbury approximation (paper: 48, reduced to 32 for speed)')
    parser.add_argument('--arap-cache', default=None,
                        help='ARAP cache dir for initial guess (e.g. data/motion_cache/walk/_old_cache)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel workers for per-muscle solve (0=auto, 1=serial with GPU)')
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
    requested_muscles = [
        name.strip() for name in args.muscles.split(',') if name.strip()]
    if requested_muscles:
        for name in requested_muscles:
            data = load_muscle(args.tet_dir, name)
            if data is not None:
                muscles.append(data)
            else:
                print(f"    WARNING: requested muscle not found: {name}")
    else:
        for side in args.sides:
            for muscle_name in UPLEG_MUSCLES:
                name = f"{side}_{muscle_name}"
                data = load_muscle(args.tet_dir, name)
                if data is not None:
                    muscles.append(data)

    if not muscles:
        print("ERROR: No muscles loaded!")
        sys.exit(1)

    if args.insertion_tendon_rings > 0 and args.workers != 1:
        print("    Tendon material is heterogeneous; forcing --workers 1")
        args.workers = 1

    for muscle in muscles:
        if args.attachment_reference_count > 0:
            origin_reference = farthest_attachment_references(
                muscle['vertices'], muscle['origin_attachment_vertices'],
                args.attachment_reference_count)
            insertion_reference = farthest_attachment_references(
                muscle['vertices'], muscle['insertion_attachment_vertices'],
                args.attachment_reference_count)
            muscle['fixed_vertices'] = sorted(set(
                origin_reference.tolist() + insertion_reference.tolist()))
            print(f"    {muscle['name']} compliant caps: "
                  f"{len(origin_reference)} origin + "
                  f"{len(insertion_reference)} insertion hard references")
        youngs_field, tendon_layer = endpoint_tendon_youngs(
            muscle, args.youngs, args.tendon_youngs,
            args.insertion_tendon_rings)
        if args.insertion_tendon_rings > 0:
            muscle['mu'], muscle['lam'] = lame_parameters(
                youngs_field, args.poisson)
            muscle['tendon_layer'] = tendon_layer
            counts = [int(np.sum(tendon_layer == depth))
                      for depth in range(args.insertion_tendon_rings)]
            print(f"    {muscle['name']} insertion tendon tet layers: "
                  f"{counts}")

    total_verts = sum(len(m['vertices']) for m in muscles)
    total_tets = sum(len(m['tetrahedra']) for m in muscles)
    print(f"    {len(muscles)} muscles: {total_verts} verts, {total_tets} tets")

    # ── Load ARAP cache for initial guess ────────────────────────────
    arap_cache = {}  # {muscle_name: {frame: positions}}
    if args.arap_cache and os.path.isdir(args.arap_cache):
        import glob as glob_mod
        print(f"[3b] Loading ARAP cache from {args.arap_cache}...")
        for m in muscles:
            chunks = sorted(glob_mod.glob(
                os.path.join(args.arap_cache, f"{m['name']}_chunk_*.npz")))
            if not chunks:
                continue
            frame_data = {}
            for cp in chunks:
                d = np.load(cp)
                for fi, frame_num in enumerate(d['frames']):
                    frame_data[int(frame_num)] = d['positions'][fi]
            arap_cache[m['name']] = frame_data
        n_cached = sum(len(v) for v in arap_cache.values())
        print(f"    Loaded {len(arap_cache)} muscles, {n_cached} frames")
    elif args.arap_cache:
        print(f"    WARNING: ARAP cache dir not found: {args.arap_cache}")

    # ── Compute LBS bindings ─────────────────────────────────────────
    print("[4] Computing LBS bindings...")
    for m in muscles:
        m['lbs_bindings'] = compute_multibone_lbs(m, skel)

    # ── EMU precomputation (per muscle) ──────────────────────────────
    print("[5] EMU precomputation...")
    for m in muscles:
        m['emu'] = precompute_emu(
            m['vertices'], m['tetrahedra'], m['fixed_vertices'],
            m.get('vertex_contour_level'), k_modes=args.k_modes,
            fiber_directions=m.get('fiber_directions'))
        origin_vertices = np.asarray(
            m.get('origin_attachment_vertices', []), dtype=np.int32)
        insertion_vertices = np.asarray(
            m.get('insertion_attachment_vertices', []), dtype=np.int32)
        if len(origin_vertices) and len(insertion_vertices):
            m['rest_attachment_span'] = float(np.linalg.norm(
                np.mean(m['vertices'][insertion_vertices], axis=0)
                - np.mean(m['vertices'][origin_vertices], axis=0)))

    # ── (Optional) unified precomputation for reference ──────────────
    print("[5b] Building unified vertex/tet arrays...")

    # Build combined vertex/tet arrays with offset tracking
    vert_offset = {}
    tet_offset = {}
    all_verts = []
    all_tets = []
    all_fixed = []
    all_vcl = []
    v_off = 0
    t_off = 0
    for m in muscles:
        vert_offset[m['name']] = v_off
        tet_offset[m['name']] = t_off
        all_verts.append(m['vertices'])
        all_tets.append(m['tetrahedra'] + v_off)
        all_fixed.extend([vi + v_off for vi in m['fixed_vertices']])
        vcl = m.get('vertex_contour_level')
        if vcl is not None:
            if len(vcl) < len(m['vertices']):
                vcl_pad = np.zeros(len(m['vertices']), dtype=vcl.dtype)
                vcl_pad[:len(vcl)] = vcl
                vcl = vcl_pad
            all_vcl.append(vcl)
        else:
            all_vcl.append(np.zeros(len(m['vertices']), dtype=np.int32))
        v_off += len(m['vertices'])
        t_off += len(m['tetrahedra'])

    combined_verts = np.concatenate(all_verts, axis=0)
    combined_tets = np.concatenate(all_tets, axis=0)
    combined_fixed = sorted(all_fixed)
    combined_vcl = np.concatenate(all_vcl)

    total_n = len(combined_verts)
    total_m = len(combined_tets)
    print(f"    Combined: {total_n} verts, {total_m} tets, {len(combined_fixed)} fixed")

    unified_precomp = precompute_emu(
        combined_verts, combined_tets, combined_fixed,
        combined_vcl, k_modes=args.k_modes)

    # ── Load bone meshes for collision ──────────────────────────────
    print("[6] Loading bone meshes...")
    import trimesh
    bone_meshes_data = {}
    bone_names_for_side = {
        'L': ['L_Os_Coxae', 'L_Femur', 'L_Tibia_Fibula', 'L_Patella'],
        'R': ['R_Os_Coxae', 'R_Femur', 'R_Tibia_Fibula', 'R_Patella'],
    }
    bone_names = set()
    for side in args.sides:
        bone_names.update(bone_meshes_data.get(side, bone_names_for_side.get(side, [])))
    bone_names.add('Saccrum_Coccyx')
    for bn in list(bone_names_for_side.get(args.sides[0], [])) + ['Saccrum_Coccyx']:
        path = os.path.join(SKEL_MESH_DIR, f'{bn}.obj')
        if os.path.exists(path):
            m_obj = trimesh.load(path, process=False)
            bone_meshes_data[bn] = {
                'vertices': np.array(m_obj.vertices, dtype=np.float64),
                'faces': np.array(m_obj.faces, dtype=np.int32),
            }
    # Cache rest transforms
    saved_pos = skel.getPositions().copy()
    skel.setPositions(np.zeros(skel.getNumDofs()))
    bone_rest_transforms = {}
    for bn in bone_meshes_data:
        body_name = BONE_NAME_TO_BODY.get(bn, bn + '0')
        body_node = skel.getBodyNode(body_name)
        if body_node:
            wt = body_node.getWorldTransform()
            bone_rest_transforms[body_name] = (wt.rotation().copy(), wt.translation().copy())
    skel.setPositions(saved_pos)
    print(f"    {len(bone_meshes_data)} bones loaded")

    # Precompute surface data for collision
    from collections import Counter as _Counter
    for m in muscles:
        tets = m['tetrahedra']
        fc = _Counter()
        fo = {}
        for t in tets:
            v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            for f in [(v0,v2,v1),(v0,v1,v3),(v1,v2,v3),(v0,v3,v2)]:
                key = tuple(sorted(f)); fc[key] += 1
                if key not in fo: fo[key] = f
        sf = [fo[k] for k, c in fc.items() if c == 1]
        m['_surf_verts'] = sorted(set(v for f in sf for v in f))

    # ── Compute fascia proximity pairs ─────────────────────────────
    print("[6b] Computing fascia proximity pairs...")
    fascia_pairs = compute_fascia_pairs(muscles, threshold=0.005)
    print(f"    {len(fascia_pairs)} inter-muscle fascia pairs")

    # ── Setup output ─────────────────────────────────────────────────
    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    side_tag = ''.join(args.sides)
    out_dir = os.path.join(args.output_dir, bvh_stem, f"emu_{side_tag}_UpLeg")
    os.makedirs(out_dir, exist_ok=True)
    print(f"    Output: {out_dir}")

    bake_buffers = {m['name']: {} for m in muscles}
    chunk_counters = {m['name']: 0 for m in muscles}

    # Warm-start: carry forward F from previous frame
    prev_Fvec = {}

    # Per-muscle parallel mode
    n_workers = args.workers if args.workers > 0 else min(os.cpu_count() or 4, len(muscles))
    use_gpu_solve = (n_workers <= 1)
    if use_gpu_solve:
        _ensure_ti()
        print(f"    Mode: serial with GPU (Taichi CUDA)")
    else:
        print(f"    Mode: parallel {n_workers} workers (CPU only)")

    # ── Bake loop ────────────────────────────────────────────────────
    print(f"\n[7] Baking frames {args.start_frame}-{end_frame}...")
    t_total = time.time()

    for frame in range(args.start_frame, end_frame + 1):
        t0 = time.time()
        pose = motion_bvh.mocap_refs[frame].copy()
        skel.setPositions(pose)

        # Build posed bone trimeshes for collision
        bone_tms = []
        for bn, bm in bone_meshes_data.items():
            body_name = BONE_NAME_TO_BODY.get(bn, bn + '0')
            body_node = skel.getBodyNode(body_name)
            if body_node is None or body_name not in bone_rest_transforms:
                continue
            R_rest, t_rest = bone_rest_transforms[body_name]
            R_cur = body_node.getWorldTransform().rotation()
            t_cur = body_node.getWorldTransform().translation()
            v_m = bm['vertices'] * MESH_SCALE
            local = (R_rest.T @ (v_m - t_rest).T).T
            posed = (R_cur @ local.T).T + t_cur
            bone_tms.append(trimesh.Trimesh(vertices=posed, faces=bm['faces'].copy(), process=True))

        # Build muscle surface info for collision
        muscle_surfaces = []
        for m in muscles:
            muscle_surfaces.append({
                'surf_verts': m['_surf_verts'],
                'fixed_set': set(m['fixed_vertices']),
                'offset': 0,
            })

        # Prepare per-muscle solve arguments
        solve_args = []
        for mi, m in enumerate(muscles):
            if m['lbs_bindings'] is None:
                continue
            n_v = len(m['vertices'])
            lbs_pos = compute_lbs_positions(m['lbs_bindings'], skel, n_v)
            q_init = lbs_pos
            if m['name'] in arap_cache and frame in arap_cache[m['name']]:
                arap_pos = arap_cache[m['name']][frame]
                if len(arap_pos) == n_v:
                    q_init = arap_pos.astype(np.float64)
            fixed_mask = np.zeros(n_v, dtype=bool)
            for vi in m['fixed_vertices']:
                if vi < n_v:
                    fixed_mask[vi] = True
            activation = None
            if ((args.activation_max > 0.0 or args.active_strain_from_span)
                    and 'rest_attachment_span' in m):
                origin_vertices = m['origin_attachment_vertices']
                insertion_vertices = m['insertion_attachment_vertices']
                current_span = float(np.linalg.norm(
                    np.mean(lbs_pos[insertion_vertices], axis=0)
                    - np.mean(lbs_pos[origin_vertices], axis=0)))
                shortening = max(
                    0.0, 1.0 - current_span / m['rest_attachment_span'])
                level = min(
                    1.0, shortening
                    / max(args.activation_full_shortening, 1e-6))
                tendon_layer = m.get('tendon_layer')
                if args.active_strain_from_span:
                    active_stretch = float(np.clip(
                        current_span / m['rest_attachment_span'],
                        args.minimum_active_stretch, 1.0))
                    fibers = m['emu']['fiber_dirs']
                    uu = np.einsum('mi,mj->mij', fibers, fibers)
                    identity = np.broadcast_to(
                        np.eye(3), (len(fibers), 3, 3)).copy()
                    # F_a^-1 = lambda^-1 uu^T
                    #          + sqrt(lambda)(I-uu^T), det(F_a)=1.
                    active_inverse = (
                        (1.0 / active_stretch) * uu
                        + np.sqrt(active_stretch) * (identity - uu))
                    if tendon_layer is not None:
                        active_inverse[
                            np.asarray(tendon_layer) >= 0] = np.eye(3)
                    m['emu']['active_strain_inverse'] = active_inverse
                    print(f"    {m['name']} active_stretch="
                          f"{active_stretch:.3f} span={current_span:.6g}m",
                          flush=True)
                else:
                    m['emu'].pop('active_strain_inverse', None)
                    activation = np.full(
                        len(m['tetrahedra']),
                        args.activation_max * level, dtype=np.float64)
                    if tendon_layer is not None:
                        activation[np.asarray(tendon_layer) >= 0] = 0.0
                    print(f"    {m['name']} activation={level:.3f} "
                          f"span={current_span:.6g}m", flush=True)
            solve_args.append((mi, m, q_init, lbs_pos, fixed_mask, activation))

        if n_workers > 1:
            import multiprocessing as mp
            _parallel_ctx.update({
                'mu': mu, 'lam': lam, 'alpha': args.alpha,
                'max_iters': args.max_iters,
                'bone_tms': bone_tms,
                'muscle_surfaces': muscle_surfaces,
                'tasks': solve_args,
                'warm_F': prev_Fvec,
                'neighbor_positions': None,
                'fascia_pairs_for_muscle': {},
            })
            ctx_mp = mp.get_context('fork')
            with ctx_mp.Pool(n_workers) as pool:
                results = pool.map(_solve_one_worker, range(len(solve_args)))
            for mname, q_result, fvec_result in results:
                bake_buffers[mname][frame] = q_result
                if fvec_result is not None:
                    prev_Fvec[mname] = fvec_result
        else:
            for task in solve_args:
                mi, m_data, q_init, lbs_pos, fixed_mask, activation = task
                warm_F = prev_Fvec.get(m_data['name'])
                q, info = emu_solve(
                    q_init, m_data['emu'], fixed_mask, lbs_pos,
                    m_data.get('mu', mu), m_data.get('lam', lam), args.alpha,
                    max_iters=args.max_iters,
                    verbose=(frame == args.start_frame),
                    bone_trimeshes=bone_tms,
                    muscle_surfaces=[muscle_surfaces[mi]],
                    margin=0.002,
                    use_gpu=use_gpu_solve,
                    warm_F=warm_F, activation=activation)
                accepted_F = _deformation_gradients_from_q(
                    q, m_data['emu']['tetrahedra'], m_data['emu']['Dm_inv'])
                accepted_J = np.linalg.det(accepted_F)
                needs_q_repair = (
                    not np.all(np.isfinite(accepted_J))
                    or np.min(accepted_J) <= 0.02)
                if (args.repair_invalid_q
                        and (needs_q_repair or args.q_refine_every_frame)):
                    print(
                        f"    {m_data['name']} q refinement: "
                        f"minJ={np.nanmin(accepted_J):.6g}", flush=True)
                    repaired_q, repair_info = relax_positions_with_jacobian_barrier(
                        q, m_data['emu'], fixed_mask, lbs_pos,
                        m_data.get('mu', mu), m_data.get('lam', lam),
                        max_iters=80, activation=activation,
                        soft_attachment_indices=(
                            np.unique(np.concatenate((
                                m_data['origin_attachment_vertices'],
                                m_data['insertion_attachment_vertices'])))
                            if args.soft_cap_weight > 0.0 else None),
                        soft_attachment_targets=(
                            lbs_pos[np.unique(np.concatenate((
                                m_data['origin_attachment_vertices'],
                                m_data['insertion_attachment_vertices'])))]
                            if args.soft_cap_weight > 0.0 else None),
                        soft_attachment_weight=args.soft_cap_weight)
                    print(f"    q repair result: {repair_info}", flush=True)
                    if repaired_q is not None:
                        q = repaired_q
                        accepted_F = _deformation_gradients_from_q(
                            q, m_data['emu']['tetrahedra'],
                            m_data['emu']['Dm_inv'])
                bake_buffers[m_data['name']][frame] = q.astype(np.float32)
                # A discontinuous optimized F can reconstruct to an invalid
                # continuous q. Only the accepted continuous state may seed
                # the next quasistatic frame.
                prev_Fvec[m_data['name']] = accepted_F.ravel().copy()

        dt = time.time() - t0
        print(f"  Frame {frame}: {dt:.2f}s ({n_workers}w)", flush=True)

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
