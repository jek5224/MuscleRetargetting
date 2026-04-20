#!/usr/bin/env python3
"""Post-process ARAP cache with ACAP-based collision correction.

Treats all muscles as one unified mesh. Detects bone-muscle AND muscle-muscle
collisions, then projects colliding vertices using ACAP energy so corrections
distribute smoothly across the mesh.

Much faster than full EMU (~2-5s/frame vs 29s) because:
- No Newton iteration, no per-tet Hessian
- Single sparse linear solve per frame
- ARAP result is the base (already has good shape)

Usage:
    python tools/fix_collisions_acap.py data/motion_cache/walk/_old_cache
    python tools/fix_collisions_acap.py data/motion_cache/walk/_old_cache --output-dir data/motion_cache/walk/arap_coll
"""
import argparse
import gc
import os
import pickle
import re
import sys
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH

SKEL_XML = "data/zygote_skel.xml"
SKEL_MESH_DIR = "Zygote_Meshes_251229/Skeleton"
MESH_SCALE = 0.01

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
# G matrix (reused from bake_emu.py)
# ---------------------------------------------------------------------------
def build_gradient_operator(vertices, tetrahedra):
    """Build gradient operator G (9m × 3n sparse)."""
    n = len(vertices)
    m = len(tetrahedra)
    X = vertices
    tets = tetrahedra
    x4 = X[tets[:, 3]]
    Dm = np.stack([X[tets[:, 0]] - x4, X[tets[:, 1]] - x4, X[tets[:, 2]] - x4], axis=-1)
    Dm_inv = np.linalg.inv(Dm)

    rows, cols, vals = [], [], []
    for c in range(3):
        for a in range(3):
            for b in range(3):
                row_idx = 9 * np.arange(m) + 3 * a + b
                col_idx = 3 * tets[:, c] + a
                val = Dm_inv[:, c, b]
                rows.append(row_idx); cols.append(col_idx); vals.append(val)

    sum_Dm_inv = Dm_inv.sum(axis=1)
    for a in range(3):
        for b in range(3):
            row_idx = 9 * np.arange(m) + 3 * a + b
            col_idx = 3 * tets[:, 3] + a
            val = -sum_Dm_inv[:, b]
            rows.append(row_idx); cols.append(col_idx); vals.append(val)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return sp.csc_matrix((vals, (rows, cols)), shape=(9 * m, 3 * n))


def extract_surface_triangles(tet_elements):
    """Extract boundary faces from tet mesh."""
    face_count = Counter()
    face_orient = {}
    for t in tet_elements:
        v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        for f in [(v0, v2, v1), (v0, v1, v3), (v1, v2, v3), (v0, v3, v2)]:
            key = tuple(sorted(f)); face_count[key] += 1
            if key not in face_orient: face_orient[key] = f
    return np.array([face_orient[k] for k, c in face_count.items() if c == 1], dtype=np.int32)


# ---------------------------------------------------------------------------
# Collision detection on unified mesh
# ---------------------------------------------------------------------------
def detect_collisions(positions, surf_faces, fixed_set, bone_trimeshes,
                       muscle_ranges, margin=0.002):
    """Detect bone-muscle and muscle-muscle collisions on the unified mesh.

    Returns dict {global_vert_idx: target_position} for colliding free vertices.
    """
    import trimesh
    from scipy.spatial import cKDTree

    targets = {}
    n = len(positions)
    surf_verts = sorted(set(np.unique(surf_faces).tolist()))

    # ── Bone-muscle collision ──────────────────────────────────────
    if bone_trimeshes:
        bone_kdtree = cKDTree(np.vstack([bm.vertices for bm in bone_trimeshes]))
        sv_pos = positions[surf_verts]
        dists, _ = bone_kdtree.query(sv_pos)
        near_mask = dists < 0.03

        if np.any(near_mask):
            near_sv = np.array(surf_verts)[near_mask]
            near_pos = positions[near_sv]
            for bone_mesh in bone_trimeshes:
                try:
                    bmin = bone_mesh.bounds[0] - 0.03
                    bmax = bone_mesh.bounds[1] + 0.03
                    in_bbox = np.all((near_pos >= bmin) & (near_pos <= bmax), axis=1)
                    if not np.any(in_bbox):
                        continue
                    bbox_sv = near_sv[in_bbox]
                    bbox_pos = near_pos[in_bbox]
                    inside = bone_mesh.contains(bbox_pos)
                    if not np.any(inside):
                        continue
                    inside_sv = bbox_sv[inside]
                    inside_pos = bbox_pos[inside]
                    closest, _, face_ids = trimesh.proximity.closest_point(
                        bone_mesh, inside_pos)
                    normals = bone_mesh.face_normals[face_ids]
                    for k in range(len(inside_sv)):
                        vi = int(inside_sv[k])
                        if vi in fixed_set:
                            continue
                        targets[vi] = closest[k] + normals[k] * margin
                except Exception:
                    continue

    # ── Muscle-muscle collision ────────────────────────────────────
    # Build per-muscle trimesh and check cross-muscle penetration
    import trimesh
    muscle_meshes = {}
    for mname, (v_start, v_end) in muscle_ranges.items():
        # Find surface faces within this muscle's vertex range
        mask = np.all((surf_faces >= v_start) & (surf_faces < v_end), axis=1)
        if not np.any(mask):
            continue
        local_faces = surf_faces[mask] - v_start
        local_pos = positions[v_start:v_end]
        try:
            tm = trimesh.Trimesh(vertices=local_pos, faces=local_faces, process=False)
            muscle_meshes[mname] = (tm, v_start, v_end)
        except Exception:
            continue

    mnames = list(muscle_meshes.keys())
    for i in range(len(mnames)):
        for j in range(i + 1, len(mnames)):
            ma, mb = mnames[i], mnames[j]
            tm_a, vs_a, ve_a = muscle_meshes[ma]
            tm_b, vs_b, ve_b = muscle_meshes[mb]

            # AABB pre-filter
            try:
                if np.any(tm_a.bounds[0] > tm_b.bounds[1] + margin) or \
                   np.any(tm_b.bounds[0] > tm_a.bounds[1] + margin):
                    continue
            except Exception:
                continue

            # Check A verts inside B
            sv_a = [v for v in surf_verts if vs_a <= v < ve_a]
            if sv_a:
                pos_a = positions[sv_a]
                local_a = pos_a  # B's trimesh is in local coords
                try:
                    inside = tm_b.contains(pos_a)
                    if np.any(inside):
                        idx = np.array(sv_a)[inside]
                        closest, _, fids = trimesh.proximity.closest_point(tm_b, pos_a[inside])
                        normals = tm_b.face_normals[fids]
                        for k in range(len(idx)):
                            vi = int(idx[k])
                            if vi in fixed_set:
                                continue
                            targets[vi] = closest[k] + vs_b * 0 + normals[k] * margin
                            # Note: closest is already in world coords since tm_b uses world positions
                except Exception:
                    pass

            # Check B verts inside A
            sv_b = [v for v in surf_verts if vs_b <= v < ve_b]
            if sv_b:
                pos_b = positions[sv_b]
                try:
                    inside = tm_a.contains(pos_b)
                    if np.any(inside):
                        idx = np.array(sv_b)[inside]
                        closest, _, fids = trimesh.proximity.closest_point(tm_a, pos_b[inside])
                        normals = tm_a.face_normals[fids]
                        for k in range(len(idx)):
                            vi = int(idx[k])
                            if vi in fixed_set:
                                continue
                            targets[vi] = closest[k] + normals[k] * margin
                except Exception:
                    pass

    return targets


# ---------------------------------------------------------------------------
# ACAP collision projection
# ---------------------------------------------------------------------------
def precompute_acap_solver(GtG, free_dofs, fixed_dofs, n):
    """Pre-factorize the base ACAP system (done once, reused every frame)."""
    from scipy.sparse.linalg import splu
    reg = 1e-6
    GtG_reg = GtG + reg * sp.eye(3 * n)
    GtG_csc = GtG_reg.tocsc()

    free_arr = np.array(sorted(free_dofs), dtype=np.int32)
    fixed_arr = np.array(sorted(fixed_dofs), dtype=np.int32)

    L_ff = GtG_csc[np.ix_(free_arr, free_arr)].tocsc()
    L_fc = GtG_csc[np.ix_(free_arr, fixed_arr)].tocsc()
    solver = splu(L_ff)

    return {
        'solver': solver,
        'L_fc': L_fc,
        'free_arr': free_arr,
        'fixed_arr': fixed_arr,
        'GtG_csc': GtG_csc,
    }


def acap_collision_project(positions, G, acap_data, collision_targets,
                            collision_weight=100.0, n_inner=5):
    """Project colliding vertices using iterative ACAP + collision penalty.

    Instead of rebuilding the factorization each frame, uses the
    pre-factorized base ACAP system and iteratively applies collision
    corrections:
      1. Solve base ACAP: q = (G^TG)^{-1} G^T F  (pre-factorized, fast)
      2. For colliding verts: blend toward target
      3. Recompute F from corrected q, repeat

    This converges quickly (3-5 iterations) because most vertices
    don't collide, so the base ACAP solution is already close.
    """
    n = len(positions)
    q_flat = positions.ravel()

    solver = acap_data['solver']
    L_fc = acap_data['L_fc']
    free_arr = acap_data['free_arr']
    fixed_arr = acap_data['fixed_arr']
    free_set = set(free_arr.tolist())

    q_fixed = q_flat[fixed_arr]

    # Build collision target array
    coll_verts = {}  # free vertex global idx → target
    for vi, target in collision_targets.items():
        if any(3 * vi + d in free_set for d in range(3)):
            coll_verts[vi] = target

    if not coll_verts:
        return positions

    blend = min(collision_weight / (collision_weight + 1.0), 0.9)

    q_current = q_flat.copy()
    for it in range(n_inner):
        # Compute F from current q
        Fvec = G @ q_current
        rhs_full = G.T @ Fvec

        # Solve base ACAP (pre-factorized)
        rhs_free = rhs_full[free_arr] - L_fc @ q_fixed
        q_free = solver.solve(rhs_free)

        q_new = q_current.copy()
        q_new[free_arr] = q_free
        q_new[fixed_arr] = q_fixed

        # Blend colliding vertices toward targets
        n_blended = 0
        for vi, target in coll_verts.items():
            q_vi = q_new[3*vi:3*vi+3]
            q_new[3*vi:3*vi+3] = (1 - blend) * q_vi + blend * target
            n_blended += 1

        q_current = q_new
        if n_blended == 0:
            break

    return q_current.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Post-process ARAP cache with ACAP collision correction")
    parser.add_argument('cache_dir', help='Input cache directory')
    parser.add_argument('--bvh', default='data/motion/walk.bvh')
    parser.add_argument('--tet-dir', default='tet')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: <cache_dir>_coll)')
    parser.add_argument('--collision-weight', type=float, default=100.0)
    parser.add_argument('--margin', type=float, default=0.002)
    parser.add_argument('--max-iters', type=int, default=3,
                        help='Collision detection + ACAP iterations per frame')
    parser.add_argument('--sides', default='L')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.cache_dir.rstrip('/') + '_coll'

    # ── Load skeleton ────────────────────────────────────────────
    print("[1] Loading skeleton...")
    skel_info, root_name, bvh_info, _, mesh_info, _ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)

    def _detect_bvh_tframe(bvh_path):
        with open(bvh_path, 'r') as f:
            content = f.read()
        for joint in ['LeftLeg', 'RightLeg']:
            pattern = rf'JOINT\s+{joint}\s*\{{[^}}]*?OFFSET\s+([\d.\-e]+)\s+([\d.\-e]+)\s+([\d.\-e]+)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                x, y, z = abs(float(match.group(1))), abs(float(match.group(2))), abs(float(match.group(3)))
                max_axis = max(x, y, z)
                if max_axis < 1e-6: continue
                if y / max_axis > 0.8: return None
                return 0
        return None

    print(f"[2] Loading BVH: {args.bvh}")
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]

    # ── Load muscle tet data ──────────────────────────────────────
    print("[3] Loading muscles...")
    import trimesh
    muscles = []
    for side in args.sides:
        for mname in UPLEG_MUSCLES:
            name = f"{side}_{mname}"
            tet_path = os.path.join(args.tet_dir, f'{name}_tet.npz')
            if not os.path.exists(tet_path):
                continue
            with open(tet_path, 'rb') as f:
                td = pickle.load(f)
            verts = td['vertices'].astype(np.float64)
            tets = td['tetrahedra'].astype(np.int32)
            # Fix orientation
            v0 = verts[tets[:, 0]]
            cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
            vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
            neg = vol < 0
            if np.any(neg):
                tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
            # Remove slivers
            v0 = verts[tets[:, 0]]
            cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
            vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
            edge_lens = [np.linalg.norm(verts[tets[:, i]] - verts[tets[:, j]], axis=1)
                         for i in range(4) for j in range(i+1, 4)]
            avg_edge = np.mean(edge_lens, axis=0)
            quality = vol / (avg_edge ** 3 + 1e-30)
            tets = tets[quality > 1e-4]
            # Fixed vertices
            fixed = set()
            cap_faces = td.get('cap_face_indices', [])
            sim_faces = td.get('sim_faces', td.get('faces'))
            if sim_faces is not None:
                for fi in cap_faces:
                    if fi < len(sim_faces):
                        for vi in sim_faces[fi]: fixed.add(int(vi))
            for vi in td.get('anchor_vertices', []): fixed.add(int(vi))

            muscles.append({
                'name': name, 'vertices': verts, 'tetrahedra': tets,
                'n_verts': len(verts), 'fixed_vertices': sorted(fixed),
            })

    print(f"    {len(muscles)} muscles")

    # ── Build unified system ──────────────────────────────────────
    print("[4] Building unified G matrix...")
    vert_offset = {}
    muscle_ranges = {}
    all_verts = []
    all_tets = []
    all_fixed = set()
    v_off = 0
    for m in muscles:
        vert_offset[m['name']] = v_off
        muscle_ranges[m['name']] = (v_off, v_off + m['n_verts'])
        all_verts.append(m['vertices'])
        all_tets.append(m['tetrahedra'] + v_off)
        for vi in m['fixed_vertices']:
            all_fixed.add(vi + v_off)
        v_off += m['n_verts']

    combined_verts = np.concatenate(all_verts)
    combined_tets = np.concatenate(all_tets)
    total_n = len(combined_verts)
    total_m = len(combined_tets)
    print(f"    Combined: {total_n} verts, {total_m} tets, {len(all_fixed)} fixed")

    G = build_gradient_operator(combined_verts, combined_tets)
    GtG = G.T @ G
    print(f"    G: {G.shape}")

    # Pre-factorize ACAP system (done once, reused every frame)
    print("    Pre-factorizing ACAP system...")
    t_fac = time.time()
    free_dofs_set = set(range(3 * total_n)) - set(3 * vi + d for vi in all_fixed for d in range(3))
    fixed_dofs_set = set(3 * vi + d for vi in all_fixed for d in range(3))
    acap_data = precompute_acap_solver(GtG, free_dofs_set, fixed_dofs_set, total_n)
    print(f"    Factorized in {time.time() - t_fac:.2f}s")

    # Surface faces
    all_surf = extract_surface_triangles(combined_tets)
    print(f"    Surface: {len(all_surf)} faces")

    # Free/fixed DOF sets
    free_dofs = np.array(sorted(
        set(range(3 * total_n)) -
        set(3 * vi + d for vi in all_fixed for d in range(3))
    ), dtype=np.int32)

    # ── Load bone meshes ──────────────────────────────────────────
    print("[5] Loading bone meshes...")
    bone_data = {}
    for bn in ['L_Os_Coxae', 'L_Femur', 'L_Tibia_Fibula', 'L_Patella', 'Saccrum_Coccyx']:
        path = os.path.join(SKEL_MESH_DIR, f'{bn}.obj')
        if os.path.exists(path):
            m_obj = trimesh.load(path, process=False)
            bone_data[bn] = {
                'vertices': np.array(m_obj.vertices, dtype=np.float64),
                'faces': np.array(m_obj.faces, dtype=np.int32),
            }
    skel.setPositions(np.zeros(skel.getNumDofs()))
    bone_rest = {}
    for bn in bone_data:
        body_name = BONE_NAME_TO_BODY.get(bn, bn + '0')
        body_node = skel.getBodyNode(body_name)
        if body_node:
            wt = body_node.getWorldTransform()
            bone_rest[body_name] = (wt.rotation().copy(), wt.translation().copy())
    print(f"    {len(bone_data)} bones")

    # ── Load ARAP cache ──────────────────────────────────────────
    print("[6] Loading ARAP cache...")
    import glob as glob_mod
    arap_cache = {}
    for m in muscles:
        chunks = sorted(glob_mod.glob(os.path.join(args.cache_dir, f"{m['name']}_chunk_*.npz")))
        if not chunks: continue
        frame_data = {}
        for cp in chunks:
            d = np.load(cp)
            for fi, fn in enumerate(d['frames']):
                frame_data[int(fn)] = d['positions'][fi]
        arap_cache[m['name']] = frame_data
    all_frames = sorted(set(f for fd in arap_cache.values() for f in fd.keys()))
    print(f"    {len(arap_cache)} muscles, {len(all_frames)} frames")

    # ── Output ───────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    bake_buffers = {m['name']: {} for m in muscles}

    # ── Process frames ────────────────────────────────────────────
    print(f"\n[7] Processing {len(all_frames)} frames...")
    t_total = time.time()

    for frame in all_frames:
        t0 = time.time()
        skel.setPositions(motion_bvh.mocap_refs[frame])

        # Build bone trimeshes at current pose
        bone_tms = []
        for bn, bm in bone_data.items():
            body_name = BONE_NAME_TO_BODY.get(bn, bn + '0')
            body_node = skel.getBodyNode(body_name)
            if body_node is None or body_name not in bone_rest: continue
            R_r, t_r = bone_rest[body_name]
            R_c = body_node.getWorldTransform().rotation()
            t_c = body_node.getWorldTransform().translation()
            v_m = bm['vertices'] * MESH_SCALE
            local = (R_r.T @ (v_m - t_r).T).T
            posed = (R_c @ local.T).T + t_c
            bone_tms.append(trimesh.Trimesh(vertices=posed, faces=bm['faces'].copy(), process=True))

        # Assemble unified positions from ARAP cache
        positions = np.zeros((total_n, 3), dtype=np.float64)
        for m in muscles:
            off = vert_offset[m['name']]
            if m['name'] in arap_cache and frame in arap_cache[m['name']]:
                positions[off:off + m['n_verts']] = arap_cache[m['name']][frame]
            else:
                positions[off:off + m['n_verts']] = m['vertices']

        # Iterate: detect collisions → ACAP project → repeat
        total_coll = 0
        for it in range(args.max_iters):
            targets = detect_collisions(
                positions, all_surf, all_fixed, bone_tms,
                muscle_ranges, margin=args.margin)
            n_coll = len(targets)
            total_coll = n_coll
            if n_coll == 0:
                break
            positions = acap_collision_project(
                positions, G, acap_data, targets,
                collision_weight=args.collision_weight)

        # Split back to per-muscle
        for m in muscles:
            off = vert_offset[m['name']]
            bake_buffers[m['name']][frame] = positions[off:off + m['n_verts']].astype(np.float32)

        dt = time.time() - t0
        print(f"  Frame {frame}: {dt:.2f}s, coll={total_coll}", flush=True)

    # ── Save ─────────────────────────────────────────────────────
    print(f"\nSaving to {args.output_dir}...")
    for mname, frames_dict in bake_buffers.items():
        if not frames_dict: continue
        frame_nums = sorted(frames_dict.keys())
        # Split into chunks of 20
        for chunk_start in range(0, len(frame_nums), 20):
            chunk_frames = frame_nums[chunk_start:chunk_start + 20]
            chunk_idx = chunk_start // 20
            positions = np.stack([frames_dict[f] for f in chunk_frames], axis=0)
            path = os.path.join(args.output_dir, f"{mname}_chunk_{chunk_idx:04d}.npz")
            np.savez_compressed(path,
                                frames=np.array(chunk_frames, dtype=np.int32),
                                positions=positions.astype(np.float32))

    elapsed = time.time() - t_total
    print(f"\nDone. {len(all_frames)} frames in {elapsed:.1f}s "
          f"({elapsed/max(len(all_frames),1):.2f}s/frame)")


if __name__ == '__main__':
    main()
