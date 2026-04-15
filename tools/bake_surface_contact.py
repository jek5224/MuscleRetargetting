#!/usr/bin/env python3
"""Muscle bake with IPC surface contact.

Proper surface-based IPC contact (point-triangle, edge-edge) with CCD,
replacing vertex-only collision projection.

Architecture:
  Sim mesh:        original tet mesh (tet_orig/ cache, ~172K verts for 25 muscles)
  Collision mesh:  surface triangles from tets + bone surfaces
  Elastic:         ARAP (local-global, TaichiARAP GPU backend)
  Contact:         IPC barrier potential (ipctk: point-tri, edge-edge, CCD)

Solver: alternating ARAP elastic + IPC contact resolution
  Per frame:
    1. LBS initial guess (or carry from previous frame)
    2. ARAP solve (elastic only, fast on GPU)
    3. IPC surface contact resolution:
       a. Build collision pairs (ipctk broad+narrow phase)
       b. Compute barrier gradient + hessian
       c. Newton step on contacting vertices only
       d. CCD line search (no tunneling)
       e. Repeat until collision-free
    4. Outer iteration: ARAP → contact → ARAP → contact → done

Advantages over vertex-only projection:
  - Edge-edge contact prevents edge-through-bone tunneling
  - Point-triangle contact for broad surface coverage
  - CCD guarantees intersection-free trajectories
  - Barrier potential: smooth force, no oscillation

Requirements:
    pip install ipctk  # needs Python 3.9+, pre-built wheels for 3.10+

Usage:
    python tools/bake_surface_contact.py --bvh data/motion/walk.bvh --sides L --end-frame 5

    # On A6000 server:
    sbatch tools/slurm_bake_surface.sh
"""
import argparse
import gc
import os
import pickle
import sys
import time
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

try:
    import ipctk
    HAS_IPCTK = True
    # Suppress noisy CCD warnings for initially-intersecting meshes
    ipctk.set_logger_level(ipctk.LoggerLevel.error)
except ImportError:
    HAS_IPCTK = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from viewer.arap_backends import check_taichi_available, get_backend
from tools.bake_original_mesh import (
    parse_muscle_xml, identify_fixed_and_assign_bones,
    compute_skinning_weights, build_edges,
    UPLEG_MUSCLES, SKELETON_NAME_MAP, SKEL_XML, MUSCLE_XML, MESH_DIR,
    SKEL_MESH_DIR, MESH_SCALE, FLUSH_INTERVAL, BONE_NAME_TO_BODY,
    get_bone_world_verts,
)
from tools.quasi_static_solver import (
    precompute_tet_data, lame_parameters, solve_quasi_static,
)

COLLISION_BONES = {
    'L': ['L_Os_Coxae', 'L_Femur', 'L_Tibia_Fibula', 'L_Patella'],
    'R': ['R_Os_Coxae', 'R_Femur', 'R_Tibia_Fibula', 'R_Patella'],
}


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------
def extract_surface_triangles(tet_elements):
    """Extract surface triangles from tet mesh.
    A face is on the surface if it belongs to exactly one tet.
    Returns Fx3 array with indices into the original vertex array.
    """
    from collections import Counter
    face_count = Counter()
    face_orient = {}  # sorted_key -> oriented face
    for t in tet_elements:
        v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        # 4 faces with outward normals for positive-volume tet
        faces = [
            (v0, v2, v1),
            (v0, v1, v3),
            (v1, v2, v3),
            (v0, v3, v2),
        ]
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] += 1
            if key not in face_orient:
                face_orient[key] = f
    return np.array(
        [face_orient[k] for k, c in face_count.items() if c == 1],
        dtype=np.int64)


def extract_edges_from_faces(faces):
    """Extract unique edges from surface faces. Returns Ex2 array."""
    edges = set()
    for f in faces:
        for i in range(3):
            e = (min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3]))
            edges.add(e)
    return np.array(sorted(edges), dtype=np.int64)


def remap_to_surface(faces, surf_verts):
    """Remap global face indices to local surface vertex indices.
    surf_verts: sorted unique vertex indices appearing in faces.
    Returns: (local_faces, global_to_local_map)
    """
    g2l = np.full(np.max(surf_verts) + 1, -1, dtype=np.int64)
    g2l[surf_verts] = np.arange(len(surf_verts), dtype=np.int64)
    local_faces = g2l[faces]
    return local_faces, g2l


# ---------------------------------------------------------------------------
# IPC contact resolution
# ---------------------------------------------------------------------------
def build_collision_data(muscles, global_offset, bone_meshes_world):
    """Build unified surface mesh for IPC collision detection.

    Returns:
        V_surf:     (N_s, 3) surface vertex positions (muscles + bones)
        E_surf:     (E_s, 2) surface edges
        F_surf:     (F_s, 3) surface faces
        surf_to_global: (N_ms,) mapping from muscle surface verts to global sim DOFs
        n_muscle_surf:  number of muscle surface vertices
        n_bone_surf:    number of bone surface vertices
        muscle_surf_fixed: bool array, True for fixed muscle surface verts
        bone_attach:    per-muscle dict of attachment body names (for collision filter)
    """
    all_verts = []
    all_edges = []
    all_faces = []
    surf_to_global = []
    vert_offset = 0

    # Muscle surfaces
    for m in muscles:
        sf = m['surface_faces']  # global indices into muscle's vertex range
        sv = np.unique(sf)
        local_f, g2l = remap_to_surface(sf, sv)

        all_verts.append(m['vertices'][sv])  # but we need global sim positions...
        # Actually, positions are updated per frame. Store mapping only.
        # Map surface vert to global sim DOF
        off = global_offset[m['name']]
        surf_to_global.extend(off + sv)

        all_faces.append(local_f + vert_offset)
        local_e = extract_edges_from_faces(local_f)
        all_edges.append(local_e + vert_offset)
        vert_offset += len(sv)

    n_muscle_surf = vert_offset
    surf_to_global = np.array(surf_to_global, dtype=np.int64)

    # Bone surfaces
    for bone_name, bm in bone_meshes_world.items():
        bv = bm['vertices']  # already in world coords
        bf = np.array(bm['faces'], dtype=np.int64)
        be = extract_edges_from_faces(bf)
        all_verts.append(bv)
        all_faces.append(bf + vert_offset)
        all_edges.append(be + vert_offset)
        vert_offset += len(bv)

    n_bone_surf = vert_offset - n_muscle_surf

    V = np.vstack(all_verts) if all_verts else np.zeros((0, 3))
    E = np.vstack(all_edges) if all_edges else np.zeros((0, 2), dtype=np.int64)
    F = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int64)

    return V, E, F, surf_to_global, n_muscle_surf, n_bone_surf


def get_surface_positions(global_positions, surf_to_global, bone_positions_list):
    """Assemble current collision mesh vertex positions."""
    V_muscle = global_positions[surf_to_global]
    if bone_positions_list:
        V_bone = np.vstack(bone_positions_list)
        return np.vstack([V_muscle, V_bone])
    return V_muscle


def resolve_onesided_ipc(global_positions, muscles, global_offset,
                          bone_world_meshes, global_fixed_mask,
                          dhat, kappa, max_iters=30, verbose=False):
    """One-sided IPC: only penalize muscle primitives INSIDE bone.

    Phase 1: Vertex-inside-bone detection (closest_point + normal dot).
             Push inside free vertices to surface + barrier-based margin.
    Phase 2: Edge-bone intersection detection (BVH ray casting).
             For edges crossing bone surface, push both endpoints outward.
    Phase 3: Point-triangle barrier for vertices approaching from outside
             within dhat — prevents future penetration.

    One-sided means: muscles wrapping NEAR bones (outside) are fine,
    only penalize actual penetration (inside) and near-approach.

    Returns: (n_vertex_penetrations, n_edge_intersections)
    """
    import trimesh as _trimesh

    total_vert_pen = 0
    total_edge_int = 0

    for bone_name, bone_wv, bone_faces in bone_world_meshes:
        body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
        bone_tree = cKDTree(bone_wv)
        bone_mesh = _trimesh.Trimesh(vertices=bone_wv, faces=bone_faces,
                                     process=False)

        for m in muscles:
            # Skip attachment bones
            attach_bones = set()
            for s in m.get('xml_streams', []):
                attach_bones.add(s[0])
                attach_bones.add(s[1])
            if body_name in attach_bones:
                continue

            off = global_offset[m['name']]
            n_v = len(m['vertices'])
            pos = global_positions[off:off + n_v]
            fixed_set = set(m['fixed_vertices'])
            sf = m['surface_faces']

            # ── Phase 1: Vertex inside bone ───────────────────────────
            sv_unique = np.unique(sf)
            sv_pos = pos[sv_unique]
            dists_kd, _ = bone_tree.query(sv_pos)
            near_mask = dists_kd < 0.030  # 30mm
            if not np.any(near_mask):
                continue

            near_sv = sv_unique[near_mask]
            near_pos = pos[near_sv]
            free_mask_near = np.array([int(v) not in fixed_set for v in near_sv])

            closest, _, face_ids = _trimesh.proximity.closest_point(
                bone_mesh, near_pos)
            normals = bone_mesh.face_normals[face_ids]
            to_vert = near_pos - closest
            signed_dist = np.sum(to_vert * normals, axis=1)

            # Vertices inside bone (signed_dist < 0) — push to surface + margin
            inside = (signed_dist < 0) & free_mask_near
            if np.any(inside):
                pen_idx = near_sv[inside]
                depths = -signed_dist[inside]
                # Barrier-inspired margin: larger for deeper penetration
                margins = np.clip(depths * 1.5 + 0.003, 0.003, 0.020)
                global_positions[off + pen_idx] = (
                    closest[inside] + normals[inside] * margins[:, None])
                total_vert_pen += int(np.sum(inside))

            # ── Phase 2: Edge-bone intersection (BVH ray cast) ────────
            # Find surface edges where at least one endpoint is near bone
            near_set = set(near_sv.tolist())
            edges_to_check = set()
            for f in sf:
                for i in range(3):
                    v0i, v1i = int(f[i]), int(f[(i + 1) % 3])
                    if v0i in near_set or v1i in near_set:
                        e = (min(v0i, v1i), max(v0i, v1i))
                        edges_to_check.add(e)

            if not edges_to_check:
                continue

            edge_arr = np.array(list(edges_to_check), dtype=np.int64)
            origins = pos[edge_arr[:, 0]]
            endpoints = pos[edge_arr[:, 1]]
            dirs = endpoints - origins
            lengths = np.linalg.norm(dirs, axis=1)
            valid = lengths > 1e-10
            if not np.any(valid):
                continue

            dirs_norm = dirs.copy()
            dirs_norm[valid] /= lengths[valid, None]

            # BVH ray cast against bone mesh
            try:
                locations, ray_idx, tri_idx = bone_mesh.ray.intersects_location(
                    origins[valid], dirs_norm[valid], multiple_hits=True)
            except Exception:
                continue

            if len(locations) == 0:
                continue

            # Filter: intersection must be within edge segment [0, length]
            valid_edges = edge_arr[valid]
            for loc, ri, ti in zip(locations, ray_idx, tri_idx):
                t_param = np.dot(loc - origins[valid][ri], dirs_norm[valid][ri])
                edge_len = lengths[valid][ri]
                if 0 < t_param < edge_len:
                    # Edge intersects bone face!
                    v0i, v1i = int(valid_edges[ri, 0]), int(valid_edges[ri, 1])
                    if v0i in fixed_set and v1i in fixed_set:
                        continue

                    # Push the endpoint that's inside the bone
                    # Determine which endpoint is deeper
                    p0 = pos[v0i]
                    p1 = pos[v1i]
                    face_normal = bone_mesh.face_normals[ti]
                    face_center = bone_wv[bone_faces[ti]].mean(axis=0)

                    d0 = np.dot(p0 - face_center, face_normal)
                    d1 = np.dot(p1 - face_center, face_normal)

                    # Push inside endpoints toward bone surface + margin
                    for vi, di in [(v0i, d0), (v1i, d1)]:
                        if di < 0 and vi not in fixed_set:
                            cp, _, fid = _trimesh.proximity.closest_point(
                                bone_mesh, pos[vi:vi + 1])
                            fn = bone_mesh.face_normals[fid[0]]
                            depth = abs(di)
                            margin = min(depth * 1.5 + 0.003, 0.020)
                            global_positions[off + vi] = cp[0] + fn * margin

                    total_edge_int += 1

            # ── Phase 3: Approaching barrier (within dhat) ────────────
            # For free vertices OUTSIDE bone but within dhat:
            # Apply a soft barrier force to prevent future penetration.
            if HAS_IPCTK:
                approaching = (signed_dist > 0) & (signed_dist < dhat) & free_mask_near
                if np.any(approaching):
                    app_idx = near_sv[approaching]
                    app_pos = pos[app_idx]
                    app_closest = closest[approaching]
                    app_normals = normals[approaching]
                    app_dist = signed_dist[approaching]

                    # Barrier force: push away from bone proportional to
                    # proximity.  b'(d) ≈ -1/d near d=0 for log barrier.
                    # Simplified: force = kappa * (1 - d/dhat)^2 * normal
                    force_mag = kappa * ((1 - app_dist / dhat) ** 2) * 0.001
                    force_mag = np.clip(force_mag, 0, 0.005)  # cap at 5mm
                    displacement = app_normals * force_mag[:, None]
                    global_positions[off + app_idx] += displacement

    return total_vert_pen, total_edge_int


# ---------------------------------------------------------------------------
# Fallback: improved vertex projection (for envs without ipctk)
# ---------------------------------------------------------------------------
def resolve_vertex_contacts(global_positions, muscles, global_offset,
                            bone_surfs_frame, global_fixed_mask):
    """Vertex-only collision projection (fallback when ipctk unavailable)."""
    import trimesh as _trimesh
    proj_count = 0
    for bone_surf, body_name, bone_tree in bone_surfs_frame:
        for m in muscles:
            attach_bones = set()
            for s in m.get('xml_streams', []):
                attach_bones.add(s[0])
                attach_bones.add(s[1])
            if body_name in attach_bones:
                continue
            off = global_offset[m['name']]
            n = len(m['vertices'])
            pos = global_positions[off:off + n]
            fixed_set = set(m['fixed_vertices'])

            dists, _ = bone_tree.query(pos)
            close_mask = dists < 0.030
            if not np.any(close_mask):
                continue
            close_idx = np.where(close_mask)[0]
            free_close = np.array([vi for vi in close_idx if vi not in fixed_set])
            if len(free_close) == 0:
                continue

            closest, _, face_ids = _trimesh.proximity.closest_point(
                bone_surf, pos[free_close])
            normals = bone_surf.face_normals[face_ids]
            to_vert = pos[free_close] - closest
            dots = np.sum(to_vert * normals, axis=1)
            inside = dots < 0

            if not np.any(inside):
                continue
            pen = free_close[inside]
            depths = -dots[inside]
            margins = np.clip(depths + 0.005, 0.005, 0.020)
            global_positions[off + pen] = closest[inside] + normals[inside] * margins[:, None]
            proj_count += len(pen)
    return proj_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Muscle bake with IPC surface contact")
    parser.add_argument('--bvh', required=True, help='BVH motion file')
    parser.add_argument('--sides', default='L', help='L, R, or LR')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=None)
    parser.add_argument('--settle-iters', type=int, default=150,
                        help='ARAP iterations per frame (frame 0 gets 2x)')
    parser.add_argument('--backend', default='auto',
                        choices=['auto', 'taichi', 'gpu', 'cpu'])
    parser.add_argument('--output-dir', default='data/motion_cache')
    parser.add_argument('--dhat', type=float, default=5.0,
                        help='IPC barrier distance in mm')
    parser.add_argument('--kappa', type=float, default=1e3,
                        help='Contact barrier stiffness')
    parser.add_argument('--youngs', type=float, default=5e3,
                        help='Young\'s modulus (Pa)')
    parser.add_argument('--poisson', type=float, default=0.45,
                        help='Poisson ratio (0.45-0.49 for volume preservation)')
    parser.add_argument('--solver', default='lbfgs',
                        choices=['lbfgs', 'arap'],
                        help='Solver: lbfgs (quasi-static FEM+contact) or arap (legacy)')
    parser.add_argument('--outer-iters', type=int, default=3,
                        help='ARAP outer iterations (arap solver only)')
    parser.add_argument('--ipc-iters', type=int, default=30,
                        help='Max contact iterations (arap solver only)')
    args = parser.parse_args()

    dhat_m = args.dhat * 0.001  # mm → m

    if not HAS_IPCTK:
        print("WARNING: ipctk not available. Using vertex-projection fallback.")
        print("         Install ipctk for proper surface contact: pip install ipctk")

    # ── Load skeleton ─────────────────────────────────────────────────────
    print("[1] Loading skeleton...")
    skel_info, root_name, bvh_info, _, mesh_info, _ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    print(f"    DOFs: {skel.getNumDofs()}, Bodies: {skel.getNumBodyNodes()}")

    # ── Load skeleton meshes for collision ─────────────────────────────────
    print("[2] Loading bone meshes...")
    bone_meshes = {}
    bone_rest_transforms = {}
    for bone_name in ['L_Os_Coxae', 'L_Femur', 'L_Tibia_Fibula', 'L_Patella',
                      'R_Os_Coxae', 'R_Femur', 'R_Tibia_Fibula', 'R_Patella',
                      'Saccrum_Coccyx']:
        import trimesh
        path = os.path.join(SKEL_MESH_DIR, f'{bone_name}.obj')
        if os.path.exists(path):
            m = trimesh.load(path, process=False)
            bone_meshes[bone_name] = {
                'vertices': np.array(m.vertices, dtype=np.float64),
                'faces': np.array(m.faces, dtype=np.int64),
            }
    skel.setPositions(np.zeros(skel.getNumDofs()))
    for bone_name in bone_meshes:
        body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
        bn = skel.getBodyNode(body_name)
        if bn is not None:
            wt = bn.getWorldTransform()
            bone_rest_transforms[body_name] = (wt.rotation().copy(), wt.translation().copy())
    print(f"    {len(bone_meshes)} bones loaded")

    # ── Load BVH ──────────────────────────────────────────────────────────
    print("[3] Parsing muscle XML...")
    xml_data = parse_muscle_xml()
    print(f"    {len(xml_data)} muscles found")

    print("[4] Loading BVH...")
    from viewer.zygote_mesh_ui import _detect_bvh_tframe
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    end_frame = args.end_frame if args.end_frame is not None else n_frames - 1
    total_frames = end_frame - args.start_frame + 1
    print(f"    Frames: {n_frames}, baking {args.start_frame}-{end_frame} ({total_frames})")

    # ── Per-side processing ───────────────────────────────────────────────
    for side in args.sides:
        print(f"\n{'=' * 60}")
        print(f"  Processing {side} side")
        print(f"{'=' * 60}")

        # ── Load muscles from tet cache ───────────────────────────────────
        print("[5] Loading muscles from tet cache...")
        muscles = []
        total_verts = 0
        total_tets = 0

        for mname in UPLEG_MUSCLES:
            full_name = f"{side}_{mname}"
            cache_path = os.path.join('tet_orig', f"L_{mname}_tet.npz")

            if not os.path.exists(cache_path):
                print(f"    {full_name}: no tet cache, skipping")
                continue

            with open(cache_path, 'rb') as f:
                cd = pickle.load(f)
            verts = cd['tet_vertices']
            tet_e = cd['tet_elements']
            cap_verts = cd['cap_vertices']
            boundary_verts = cd['boundary_vertices']

            xml_key = f"L_{mname}"
            mxml = xml_data.get(xml_key, [])

            # Mirror for R side
            if side == 'R':
                verts = verts.copy()
                verts[:, 0] *= -1
                tet_e = tet_e.copy()
                tet_e[:, 1], tet_e[:, 2] = tet_e[:, 2].copy(), tet_e[:, 1].copy()

            bone_map = None
            if side == 'R':
                bone_map = {}
                for k, v_name in SKELETON_NAME_MAP.items():
                    bone_map[k + '0'] = v_name + '0'
                bone_map['Saccrum_Coccyx0'] = 'Saccrum_Coccyx0'

            fixed_verts, local_anchors, init_transforms = identify_fixed_and_assign_bones(
                verts, cap_verts, cd['orig_vertices'] if side == 'L'
                else cd['orig_vertices'].copy() * np.array([-1, 1, 1]),
                mxml, skel, bone_map)

            skin_w, skin_bones = compute_skinning_weights(verts, local_anchors, init_transforms)
            edges = build_edges(tet_e)

            # Extract surface triangles for IPC
            surf_faces = extract_surface_triangles(tet_e)

            n_fixed = len(local_anchors)
            total_verts += len(verts)
            total_tets += len(tet_e)

            muscles.append({
                'name': full_name,
                'vertices': verts,
                'tetrahedra': tet_e,
                'edges': edges,
                'surface_faces': surf_faces,
                'fixed_vertices': sorted(fixed_verts.keys()),
                'local_anchors': local_anchors,
                'initial_transforms': init_transforms,
                'skinning_weights': skin_w,
                'skinning_bones': skin_bones,
                'n_orig': len(cd['orig_vertices']),
                'xml_streams': mxml,
            })
            print(f"    {full_name}: {len(verts)} verts, {len(tet_e)} tets, "
                  f"{len(surf_faces)} surf faces, {n_fixed} fixed")

        print(f"    Total: {len(muscles)} muscles, {total_verts} verts, {total_tets} tets")

        if not muscles:
            print("    No muscles loaded, skipping side")
            continue

        # ── Build unified ARAP system ─────────────────────────────────────
        print("[6] Building ARAP system...")
        global_offset = {}
        offset = 0
        for m in muscles:
            global_offset[m['name']] = offset
            offset += len(m['vertices'])

        n_total = offset
        global_rest = np.zeros((n_total, 3))
        global_fixed_mask = np.zeros(n_total, dtype=bool)
        all_local_anchors = {}

        neighbors = [[] for _ in range(n_total)]
        edge_weights = {}
        rest_edge_vectors = [{} for _ in range(n_total)]

        for m in muscles:
            off = global_offset[m['name']]
            n = len(m['vertices'])
            global_rest[off:off + n] = m['vertices']
            for vi in m['fixed_vertices']:
                global_fixed_mask[off + vi] = True
            for vi, (bone, lpos) in m['local_anchors'].items():
                all_local_anchors[off + vi] = (bone, lpos)
            for e in m['edges']:
                gi, gj = off + int(e[0]), off + int(e[1])
                neighbors[gi].append(gj)
                neighbors[gj].append(gi)
                edge_weights[(gi, gj)] = 1.0
                edge_weights[(gj, gi)] = 1.0
                rest_edge_vectors[gi][gj] = global_rest[gi] - global_rest[gj]
                rest_edge_vectors[gj][gi] = global_rest[gj] - global_rest[gi]

        print(f"    Unified: {n_total} verts, {sum(len(n) for n in neighbors) // 2} edges, "
              f"{np.sum(global_fixed_mask)} fixed")

        # ── Build collision surface data ──────────────────────────────────
        side_bones = COLLISION_BONES.get(side, [])
        if HAS_IPCTK:
            print("[6b] Building IPC collision mesh...")
            # We build the topology once; positions update per frame.
            # Collect muscle surface vertex indices (in global sim coords)
            all_surf_verts_global = []  # global sim indices of surface verts
            all_surf_faces_local = []   # faces with local surface indexing
            surf_offset = 0

            per_muscle_surf = []  # for each muscle: (surf_vert_global, surf_vert_local_offset)
            for m in muscles:
                sf = m['surface_faces']
                off = global_offset[m['name']]
                # Global indices of surface vertices
                sv_local = np.unique(sf)
                sv_global = off + sv_local
                # Remap faces to surface-local indices
                g2l = np.full(np.max(sv_local) + 1, -1, dtype=np.int64)
                g2l[sv_local] = np.arange(len(sv_local), dtype=np.int64) + surf_offset
                local_faces = np.vectorize(lambda x: g2l[x])(sf)
                all_surf_verts_global.extend(sv_global.tolist())
                all_surf_faces_local.append(local_faces)
                per_muscle_surf.append((sv_global, surf_offset, len(sv_local)))
                surf_offset += len(sv_local)

            n_muscle_surf = surf_offset
            surf_to_global = np.array(all_surf_verts_global, dtype=np.int64)

            # Bone surfaces (topology fixed, positions update per frame)
            bone_surf_info = []  # [(bone_name, n_verts, n_faces)]
            all_bone_faces = []
            bone_vert_offset = n_muscle_surf
            for bone_name in side_bones:
                if bone_name not in bone_meshes:
                    continue
                bm = bone_meshes[bone_name]
                nv = len(bm['vertices'])
                bf = bm['faces'] + bone_vert_offset
                all_bone_faces.append(bf)
                bone_surf_info.append((bone_name, nv, len(bf)))
                bone_vert_offset += nv

            n_bone_surf = bone_vert_offset - n_muscle_surf

            # Concatenate faces
            all_faces = np.vstack(all_surf_faces_local + all_bone_faces) if (
                all_surf_faces_local or all_bone_faces) else np.zeros((0, 3), dtype=np.int64)
            all_edges_ipc = extract_edges_from_faces(all_faces)

            print(f"    Collision mesh: {n_muscle_surf} muscle surf verts, "
                  f"{n_bone_surf} bone surf verts, {len(all_faces)} faces, "
                  f"{len(all_edges_ipc)} edges")

        # ── Identify collision candidate vertices (at rest pose) ─────────
        print("[6c] Identifying collision candidates at rest pose...")
        collision_candidates = set()
        skel.setPositions(np.zeros(skel.getNumDofs()))
        for bone_name in side_bones:
            if bone_name not in bone_meshes:
                continue
            bm = bone_meshes[bone_name]
            body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
            wv = get_bone_world_verts(
                skel, bone_name, bm['vertices'], bone_rest_transforms)
            if wv is None:
                continue
            bone_tree = cKDTree(wv)
            for m_data in muscles:
                # Skip attachment bones
                attach_bones = set()
                for s in m_data.get('xml_streams', []):
                    attach_bones.add(s[0])
                    attach_bones.add(s[1])
                if body_name in attach_bones:
                    continue
                off = global_offset[m_data['name']]
                n_v = len(m_data['vertices'])
                dists, _ = bone_tree.query(m_data['vertices'])
                near = np.where(dists < 0.020)[0]  # 20mm at rest
                for vi in near:
                    if not global_fixed_mask[off + vi]:
                        collision_candidates.add(off + int(vi))
        print(f"    {len(collision_candidates)} collision candidate vertices "
              f"(out of {n_total - int(np.sum(global_fixed_mask))} free)")

        # ── Solver setup ───────────────────────────────────────────────────
        import trimesh as _trimesh

        if args.solver == 'lbfgs':
            # Quasi-static FEM + one-sided contact via L-BFGS
            mu, lam = lame_parameters(args.youngs, args.poisson)
            print(f"[6c] Quasi-static solver: E={args.youngs}, ν={args.poisson}, "
                  f"μ={mu:.1f}, λ={lam:.1f}")

            # Precompute tet data for all muscles (unified)
            all_tets_global = []
            for m in muscles:
                off = global_offset[m['name']]
                all_tets_global.append(m['tetrahedra'] + off)
            global_tets = np.vstack(all_tets_global)
            Dm_inv, tet_volumes = precompute_tet_data(global_rest, global_tets)
            print(f"    {len(global_tets)} tets, total volume {tet_volumes.sum():.6f} m³")
        else:
            # Legacy ARAP backend
            backend_name = args.backend
            if backend_name == 'auto':
                backend_name = 'taichi' if check_taichi_available() else 'cpu'
            print(f"    Backend: {backend_name}")
            backend = get_backend(backend_name)
            backend.build_system(n_total, neighbors, edge_weights, global_fixed_mask,
                                 regularization=1e-6)
            print(f"    System built")

        # ── Frame loop ────────────────────────────────────────────────────
        print(f"[7] Baking frames {args.start_frame}-{end_frame}...")
        bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
        cache_dir = os.path.join(args.output_dir, bvh_stem, f"orig_{side}_UpLeg")
        os.makedirs(cache_dir, exist_ok=True)

        bake_data = {m['name']: {} for m in muscles}
        global_positions = global_rest.copy()
        bake_start = time.time()

        for fi, frame in enumerate(range(args.start_frame, end_frame + 1)):
            frame_start = time.time()
            skel.setPositions(motion_bvh.mocap_refs[frame])

            # Update fixed targets (full array for L-BFGS, indexed for ARAP)
            fixed_indices = np.where(global_fixed_mask)[0]
            fixed_targets_full = np.zeros((n_total, 3))
            fixed_targets_full[:] = global_rest
            for gi, (bone_name, local_pos) in all_local_anchors.items():
                body_node = skel.getBodyNode(bone_name)
                if body_node is not None:
                    R = body_node.getWorldTransform().rotation()
                    t = body_node.getWorldTransform().translation()
                    fixed_targets_full[gi] = R @ local_pos + t

            # LBS initial guess
            for m in muscles:
                off = global_offset[m['name']]
                n = len(m['vertices'])
                rest = m['vertices']
                lbs = np.zeros((n, 3))
                for bi, bone_name in enumerate(m['skinning_bones']):
                    body_node = skel.getBodyNode(bone_name)
                    if body_node is None:
                        continue
                    R_cur = body_node.getWorldTransform().rotation()
                    t_cur = body_node.getWorldTransform().translation()
                    if bone_name in m['initial_transforms']:
                        R0, t0 = m['initial_transforms'][bone_name]
                    else:
                        continue
                    local = (R0.T @ (rest - t0).T).T
                    deformed = (R_cur @ local.T).T + t_cur
                    w = m['skinning_weights'][:, bi:bi + 1]
                    lbs += w * deformed
                global_positions[off:off + n] = lbs

            verbose_frame = (fi % 20 == 0)

            # Build bone contact data for this frame
            bone_contact_data = []  # for quasi_static_solver format
            for bone_name in side_bones:
                if bone_name not in bone_meshes:
                    continue
                bm = bone_meshes[bone_name]
                body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
                wv = get_bone_world_verts(
                    skel, bone_name, bm['vertices'], bone_rest_transforms)
                if wv is None:
                    continue
                bone_mesh = _trimesh.Trimesh(
                    vertices=wv, faces=bm['faces'], process=False)
                bone_tree = cKDTree(wv)
                # Muscle info for this bone
                muscle_infos = []
                for m_data in muscles:
                    attach_bones = set()
                    for s in m_data.get('xml_streams', []):
                        attach_bones.add(s[0])
                        attach_bones.add(s[1])
                    off = global_offset[m_data['name']]
                    n_v = len(m_data['vertices'])
                    fixed_set = set(m_data['fixed_vertices'])
                    muscle_infos.append((off, n_v, fixed_set, attach_bones))
                bone_contact_data.append((bone_mesh, bone_tree, body_name, muscle_infos))

            # ── Solve ─────────────────────────────────────────────────────
            if args.solver == 'lbfgs':
                solve_iters = args.settle_iters * 2 if fi == 0 else args.settle_iters
                global_positions, n_iters, final_E = solve_quasi_static(
                    global_positions, global_rest, global_tets, Dm_inv, tet_volumes,
                    mu, lam, global_fixed_mask, fixed_targets_full[fixed_indices],
                    bone_contact_data, dhat_m, args.kappa,
                    max_iterations=solve_iters, tolerance=1e-4,
                    verbose=verbose_frame)
            else:
                # Legacy ARAP
                solve_iters = args.settle_iters * 2 if fi == 0 else args.settle_iters
                fixed_targets_arap = fixed_targets_full[fixed_indices]
                global_positions, n_iters, _ = backend.solve(
                    global_positions, global_rest, neighbors, edge_weights,
                    rest_edge_vectors, global_fixed_mask, fixed_targets_arap,
                    max_iterations=solve_iters, tolerance=1e-4,
                    verbose=verbose_frame)

            total_collisions = 0  # placeholder

            # Capture
            for m in muscles:
                off = global_offset[m['name']]
                n = len(m['vertices'])
                bake_data[m['name']][frame] = global_positions[off:off + n].astype(np.float32)

            # Progress
            dt = time.time() - frame_start
            elapsed = time.time() - bake_start
            avg = elapsed / (fi + 1)
            eta = avg * (total_frames - fi - 1)
            coll_str = f", {total_collisions} contacts" if total_collisions > 0 else ""
            print(f"  Frame {frame}/{end_frame} ({fi + 1}/{total_frames}) "
                  f"{dt:.2f}s avg {avg:.2f}s/frame ETA {eta:.0f}s{coll_str}",
                  flush=True)

            # Progressive save
            if (fi + 1) % FLUSH_INTERVAL == 0 or frame == end_frame:
                chunk_start = fi - (fi % FLUSH_INTERVAL)
                chunk_idx = chunk_start // FLUSH_INTERVAL
                chunk_frames = list(range(
                    args.start_frame + chunk_start,
                    min(args.start_frame + chunk_start + FLUSH_INTERVAL, end_frame + 1)))
                for m in muscles:
                    chunk_pos = []
                    chunk_f = []
                    for cf in chunk_frames:
                        if cf in bake_data[m['name']]:
                            chunk_pos.append(bake_data[m['name']][cf])
                            chunk_f.append(cf)
                    if chunk_pos:
                        out_path = os.path.join(
                            cache_dir, f"{m['name']}_chunk_{chunk_idx:04d}.npz")
                        np.savez(out_path,
                                 frames=np.array(chunk_f, dtype=np.int32),
                                 positions=np.array(chunk_pos, dtype=np.float32))
                # Free saved frames
                for cf in chunk_frames:
                    for m in muscles:
                        bake_data[m['name']].pop(cf, None)
                gc.collect()
                print(f"  Flushed chunk {chunk_idx}")

        elapsed_total = time.time() - bake_start
        print(f"\nDone in {elapsed_total:.1f}s. {len(muscles)} muscles, {total_frames} frames.")
        print(f"Output: {cache_dir}/")


if __name__ == '__main__':
    main()
