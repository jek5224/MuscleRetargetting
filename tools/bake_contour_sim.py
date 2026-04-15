#!/usr/bin/env python3
"""Contour tet-mesh muscle bake with surface-based collision.

Simulates directly on contour mesh tets (480-857 verts/muscle) — no mapping
to/from fine-grained original meshes needed.

Elastic solver:  XPBD Neo-Hookean (UnifiedFEMSolver, GPU via Taichi)
Collision:       XPBD built-in vertex-bone projection
               + edge-bone ray cast post-processing (catches edge tunneling)
               + muscle-muscle surface push-apart
Mesh source:     Contour tet caches from tet/ directory

The key advantage over vertex-only collision: contour meshes have long edges
that can tunnel through bones. The edge-bone ray cast detects and resolves
these after each XPBD solve.

Usage:
    python tools/bake_contour_sim.py --bvh data/motion/walk.bvh --sides L
    python tools/bake_contour_sim.py --bvh data/motion/dance.bvh --sides LR --end-frame 50
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
from scipy.spatial import cKDTree
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from viewer.fem_sim import UnifiedFEMSolver

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKEL_XML = "data/zygote_skel.xml"
SKEL_MESH_DIR = "Zygote_Meshes_251229/Skeleton"
MESH_SCALE = 0.01  # Zygote OBJ cm → meters (for bone meshes only)
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

COLLISION_BONES = {
    'L': ['L_Os_Coxae', 'L_Femur', 'L_Tibia_Fibula', 'L_Patella'],
    'R': ['R_Os_Coxae', 'R_Femur', 'R_Tibia_Fibula', 'R_Patella'],
}

BONE_NAME_TO_BODY = {
    'L_Os_Coxae': 'L_Os_Coxae0', 'L_Femur': 'L_Femur0',
    'L_Tibia_Fibula': 'L_Tibia_Fibula0', 'L_Patella': 'L_Patella0',
    'R_Os_Coxae': 'R_Os_Coxae0', 'R_Femur': 'R_Femur0',
    'R_Tibia_Fibula': 'R_Tibia_Fibula0', 'R_Patella': 'R_Patella0',
    'Saccrum_Coccyx': 'Saccrum_Coccyx0',
}


# ---------------------------------------------------------------------------
# Skeleton / BVH helpers
# ---------------------------------------------------------------------------
def _find_body(skel, name):
    """Find DART body node by name, trying exact match then fuzzy."""
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
    """Detect if BVH needs T_frame=0 (non-upright rest pose)."""
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
            print(f"[Motion] Non-upright rest pose detected, using T_frame=0")
            return 0
    return None


# ---------------------------------------------------------------------------
# Muscle tet loading
# ---------------------------------------------------------------------------
def load_muscle(tet_dir, name):
    """Load contour tet mesh. Vertices are already in meters."""
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

    # Remove degenerate and near-degenerate tets.
    # Contour meshes have many thin tets (24% with quality<0.001 in some muscles)
    # that flip instantly under LBS deformation, causing cascading inversions.
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    # Quality metric: vol / avg_edge^3 — thin slivers have near-zero quality
    edge_lens = []
    for i in range(4):
        for j in range(i + 1, 4):
            edge_lens.append(np.linalg.norm(verts[tets[:, i]] - verts[tets[:, j]], axis=1))
    avg_edge = np.mean(edge_lens, axis=0)
    quality = vol / (avg_edge ** 3 + 1e-30)
    good = quality > 1e-4  # remove worst slivers
    if not np.all(good):
        n_removed = int(np.sum(~good))
        tets = tets[good]

    # Collect fixed vertices from caps + anchors
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

    # Keep ALL vertices (no compaction). Orphan vertices (not in any tet)
    # get zero inverse mass in XPBD = fixed. This ensures output vertex
    # count matches the viewer's tet_vertices and LBS covers all vertices.

    cap_attachments = data.get('cap_attachments', [])
    attach_skel_names = data.get('attach_skeleton_names', [])

    return {
        'name': name,
        'vertices': verts,
        'rest_vertices': verts.copy(),
        'tetrahedra': tets,
        'fixed_vertices': sorted(fixed_verts),
        'cap_attachments': cap_attachments,
        'attach_skeleton_names': attach_skel_names,
        'remap': None,
    }


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------
def extract_surface_triangles(tet_elements):
    """Extract boundary faces from tet mesh (faces belonging to exactly 1 tet)."""
    face_count = Counter()
    face_orient = {}
    for t in tet_elements:
        v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
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
            e = (min(int(f[i]), int(f[(i + 1) % 3])),
                 max(int(f[i]), int(f[(i + 1) % 3])))
            edges.add(e)
    return np.array(sorted(edges), dtype=np.int64)


# ---------------------------------------------------------------------------
# Multi-bone LBS
# ---------------------------------------------------------------------------
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
            t0 = node.getWorldTransform().translation()  # meters
            bone_info[raw_name] = {'node_name': resolved, 'R0': R0, 't0': t0}

    if len(bone_info) == 0:
        return None

    bone_names = list(bone_info.keys())

    # Assign cap anchor vertices to their bone
    per_vertex = {}
    anchor_bone = {}

    if len(cap_att) > 0:
        for row in cap_att:
            orig_vi = int(row[0])
            stream_idx = int(row[1])
            end_idx = int(row[2])
            if stream_idx < len(attach_names) and end_idx < len(attach_names[stream_idx]):
                bone_name = attach_names[stream_idx][end_idx]
                vi = orig_vi
                if remap is not None:
                    vi = int(remap[orig_vi]) if orig_vi < len(remap) else -1
                if vi >= 0 and vi < n_verts and bone_name in bone_info:
                    per_vertex[vi] = [(bone_name, 1.0)]
                    anchor_bone[vi] = bone_name

    # Heat diffusion from anchors
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
    n_iters = 100
    bone_heat = {}
    for bname in bone_names:
        if bname not in bone_cap_verts:
            continue
        heat = np.zeros(n_verts)
        for vi in bone_cap_verts[bname]:
            heat[vi] = 1.0
        bone_heat[bname] = heat

    for _ in range(n_iters):
        for bname, heat in bone_heat.items():
            new_heat = heat.copy()
            for vi in range(n_verts):
                if vi in fixed or len(adj[vi]) == 0:
                    continue
                new_heat[vi] = np.mean([heat[ni] for ni in adj[vi]])
            bone_heat[bname] = new_heat

    # Assign normalized weights
    for vi in range(n_verts):
        if vi in per_vertex:
            continue
        weights = []
        for bname, heat in bone_heat.items():
            if heat[vi] > 1e-8:
                weights.append((bname, heat[vi]))
        if len(weights) == 0:
            best_dist = float('inf')
            best_bone = bone_names[0]
            for bname in bone_names:
                d = np.linalg.norm(rest_verts[vi] - bone_info[bname]['t0'])
                if d < best_dist:
                    best_dist = d
                    best_bone = bname
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
    """Compute multi-bone LBS-deformed positions at current skeleton pose."""
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
            t1 = node.getWorldTransform().translation()  # meters
            pos += w * (R1 @ (R0.T @ (rest_pos - t0)) + t1)
        positions[vi] = pos

    return positions


# ---------------------------------------------------------------------------
# Bone mesh loading
# ---------------------------------------------------------------------------
def load_bone_meshes(sides):
    """Load skeleton OBJ meshes for collision."""
    bone_names = set()
    for side in sides:
        bone_names.update(COLLISION_BONES.get(side, []))
    bone_names.add('Saccrum_Coccyx')

    bone_meshes = {}
    for bone_name in bone_names:
        path = os.path.join(SKEL_MESH_DIR, f'{bone_name}.obj')
        if os.path.exists(path):
            m = trimesh.load(path, process=False)
            bone_meshes[bone_name] = {
                'vertices': np.array(m.vertices, dtype=np.float64),
                'faces': np.array(m.faces, dtype=np.int32),
            }
    return bone_meshes


def compute_bone_rest_transforms(skel, bone_meshes):
    """Compute rest-pose transforms for all bones."""
    saved_pos = skel.getPositions().copy()
    skel.setPositions(np.zeros(skel.getNumDofs()))
    rest_transforms = {}
    for bone_name in bone_meshes:
        body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
        bn = skel.getBodyNode(body_name)
        if bn is not None:
            wt = bn.getWorldTransform()
            rest_transforms[body_name] = (wt.rotation().copy(), wt.translation().copy())
    skel.setPositions(saved_pos)
    return rest_transforms


def build_bone_trimeshes(skel, bone_meshes, bone_rest_transforms):
    """Build posed bone trimesh objects for current skeleton pose."""
    result = []
    for bone_name, bm in bone_meshes.items():
        body_name = BONE_NAME_TO_BODY.get(bone_name, bone_name + '0')
        body_node = skel.getBodyNode(body_name)
        if body_node is None:
            continue
        if body_name not in bone_rest_transforms:
            continue

        R_rest, t_rest = bone_rest_transforms[body_name]
        R_cur = body_node.getWorldTransform().rotation()
        t_cur = body_node.getWorldTransform().translation()

        # Bone OBJ verts are in cm, in rest-pose world coords
        v_m = bm['vertices'] * MESH_SCALE  # cm → m
        local = (R_rest.T @ (v_m - t_rest).T).T
        posed = (R_cur @ local.T).T + t_cur

        tm = trimesh.Trimesh(vertices=posed, faces=bm['faces'].copy(), process=True)
        result.append(tm)
    return result


# ---------------------------------------------------------------------------
# Edge-bone surface collision (post-XPBD)
# ---------------------------------------------------------------------------
def resolve_surface_collisions(solver, muscles, bone_trimeshes,
                                margin=0.002, max_iters=3, verbose=False):
    """Surface collision: vertex-bone + edge-bone resolution.

    Phase 1: Push surface vertices inside ANY bone to surface + margin.
             Uses closest_point + signed distance (works for non-watertight).
    Phase 2: Ray-cast along surface edges to detect edge-bone tunneling.
             Push inside endpoint(s) to bone surface + margin.

    Returns (n_vert_pushed, n_edge_resolved).
    """
    if not bone_trimeshes:
        return 0, 0

    # Merge all bones for KDTree pre-filter
    all_bone_verts = np.vstack([bm.vertices for bm in bone_trimeshes])
    bone_kdtree = cKDTree(all_bone_verts)

    total_vert_pushed = 0
    total_edge_resolved = 0

    for iteration in range(max_iters):
        n_vert = 0
        n_edge = 0

        for m in muscles:
            name = m['name']
            pos = solver.get_muscle_positions(name).copy()
            fixed_set = set(m['fixed_vertices'])
            modified = False

            # ── Phase 1: Vertex-bone collision ──────────────────────
            surf_verts = np.array(m['_surface_vert_set'], dtype=np.int64)
            sv_pos = pos[surf_verts]
            dists_kd, _ = bone_kdtree.query(sv_pos)
            near_mask = dists_kd < 0.03  # 3cm
            if np.any(near_mask):
                near_sv = surf_verts[near_mask]
                near_pos = pos[near_sv]

                for bone_mesh in bone_trimeshes:
                    try:
                        # AABB pre-filter
                        bmin = bone_mesh.bounds[0] - 0.03
                        bmax = bone_mesh.bounds[1] + 0.03
                        in_bbox = np.all((near_pos >= bmin) & (near_pos <= bmax), axis=1)
                        if not np.any(in_bbox):
                            continue

                        bbox_pos = near_pos[in_bbox]
                        bbox_sv = near_sv[in_bbox]

                        # Use contains() for reliable inside/outside detection
                        inside = bone_mesh.contains(bbox_pos)

                        if not np.any(inside):
                            continue

                        # Project inside vertices to surface + margin
                        inside_pos = bbox_pos[inside]
                        inside_sv = bbox_sv[inside]
                        closest, _, face_ids = trimesh.proximity.closest_point(
                            bone_mesh, inside_pos)
                        normals = bone_mesh.face_normals[face_ids]

                        for k in range(len(inside_sv)):
                            vi = int(inside_sv[k])
                            if vi in fixed_set:
                                continue
                            pos[vi] = closest[k] + normals[k] * margin
                            modified = True
                            n_vert += 1
                    except Exception:
                        continue

                near_pos = pos[near_sv]

            # ── Phase 2: Edge-bone collision ────────────────────────
            surface_edges = m['_surface_edges']
            edge_v0 = pos[surface_edges[:, 0]]
            edge_v1 = pos[surface_edges[:, 1]]

            d0, _ = bone_kdtree.query(edge_v0)
            d1, _ = bone_kdtree.query(edge_v1)
            edge_near = (d0 < 0.02) | (d1 < 0.02)
            if np.any(edge_near):
                candidate_edges = surface_edges[edge_near]
                origins = pos[candidate_edges[:, 0]]
                endpoints = pos[candidate_edges[:, 1]]
                dirs = endpoints - origins
                lengths = np.linalg.norm(dirs, axis=1)
                valid = lengths > 1e-10
                if np.any(valid):
                    dirs_norm = dirs.copy()
                    dirs_norm[valid] /= lengths[valid, None]

                    for bone_mesh in bone_trimeshes:
                        try:
                            locations, ray_idx, tri_idx = bone_mesh.ray.intersects_location(
                                origins[valid], dirs_norm[valid], multiple_hits=True)
                        except Exception:
                            continue
                        if len(locations) == 0:
                            continue

                        valid_edges = candidate_edges[valid]
                        for loc, ri, ti in zip(locations, ray_idx, tri_idx):
                            t_param = np.dot(loc - origins[valid][ri], dirs_norm[valid][ri])
                            edge_len = lengths[valid][ri]
                            if 0 < t_param < edge_len:
                                v0i, v1i = int(valid_edges[ri, 0]), int(valid_edges[ri, 1])
                                if v0i in fixed_set and v1i in fixed_set:
                                    continue
                                face_normal = bone_mesh.face_normals[ti]
                                face_center = bone_mesh.triangles[ti].mean(axis=0)
                                d0s = np.dot(pos[v0i] - face_center, face_normal)
                                d1s = np.dot(pos[v1i] - face_center, face_normal)
                                for vi, di in [(v0i, d0s), (v1i, d1s)]:
                                    if di < 0 and vi not in fixed_set:
                                        cp, _, fid = trimesh.proximity.closest_point(
                                            bone_mesh, pos[vi:vi + 1])
                                        fn = bone_mesh.face_normals[fid[0]]
                                        depth = abs(di)
                                        push_margin = min(depth * 1.5 + margin, 0.020)
                                        pos[vi] = cp[0] + fn * push_margin
                                        modified = True
                                        n_edge += 1

            if modified:
                vs, ve, _, _ = solver._muscle_ranges[name]
                solver.positions[vs:ve] = pos

        if verbose and (n_vert > 0 or n_edge > 0):
            print(f"    surface coll iter {iteration}: {n_vert} verts, {n_edge} edges pushed")

        total_vert_pushed += n_vert
        total_edge_resolved += n_edge
        if n_vert == 0 and n_edge == 0:
            break

    return total_vert_pushed, total_edge_resolved


def resolve_muscle_muscle_collisions(solver, muscles, d_min=0.002):
    """Push apart overlapping surface vertices between different muscles."""
    global_pts = []
    global_muscle_id = []
    global_info = []  # (muscle_name, local_vertex_idx)

    for mi, m in enumerate(muscles):
        name = m['name']
        pos = solver.get_muscle_positions(name)
        surf_verts = m['_surface_vert_set']
        for vi in surf_verts:
            if vi < len(pos):
                global_pts.append(pos[vi])
                global_muscle_id.append(mi)
                global_info.append((name, vi))

    if len(global_pts) == 0:
        return 0

    global_pts = np.array(global_pts, dtype=np.float64)
    global_muscle_id = np.array(global_muscle_id)

    n_pushed = 0
    for _ in range(3):
        tree = cKDTree(global_pts)
        pairs = tree.query_pairs(r=d_min)
        iter_pushed = 0
        for i, j in pairs:
            if global_muscle_id[i] == global_muscle_id[j]:
                continue
            diff = global_pts[j] - global_pts[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                diff = np.array([0.0, 1.0, 0.0])
                dist = 1.0
            push = (d_min - dist) * 0.5 * diff / dist
            global_pts[i] -= push
            global_pts[j] += push
            iter_pushed += 1
        n_pushed += iter_pushed
        if iter_pushed == 0:
            break

    # Write back
    for gi, (name, vi) in enumerate(global_info):
        vs, ve, _, _ = solver._muscle_ranges[name]
        solver.positions[vs + vi] = global_pts[gi]

    return n_pushed


def smooth_output_positions(solver, muscles, iterations=5):
    """Taubin smoothing on output surface using surface-only adjacency.

    Uses surface triangle edges (not tet edges) so smoothing stays on the
    surface instead of pulling vertices inward toward interior.
    """
    lam = 0.5
    mu = -0.52

    for m in muscles:
        name = m['name']
        vs, ve, _, _ = solver._muscle_ranges[name]
        pos = solver.positions[vs:ve].copy()
        fixed_set = set(m['fixed_vertices'])
        surf_adj = m['_surf_adj']

        for _ in range(iterations):
            new_pos = pos.copy()
            for vi, nbrs in surf_adj.items():
                if vi in fixed_set or len(nbrs) == 0:
                    continue
                centroid = np.mean(pos[list(nbrs)], axis=0)
                new_pos[vi] = pos[vi] + lam * (centroid - pos[vi])
            pos = new_pos
            new_pos = pos.copy()
            for vi, nbrs in surf_adj.items():
                if vi in fixed_set or len(nbrs) == 0:
                    continue
                centroid = np.mean(pos[list(nbrs)], axis=0)
                new_pos[vi] = pos[vi] + mu * (centroid - pos[vi])
            pos = new_pos

        solver.positions[vs:ve] = pos


# ---------------------------------------------------------------------------
# Inter-muscle constraint discovery
# ---------------------------------------------------------------------------
def find_inter_muscle_constraints(muscles, threshold=0.015):
    """Find distance constraints between surface vertices of different muscles."""
    constraints = []
    for i in range(len(muscles)):
        m1 = muscles[i]
        pos1 = m1['rest_vertices']
        fixed1 = set(m1['fixed_vertices'])
        sv1 = sorted(m1['_surface_vert_set'])

        for j in range(i + 1, len(muscles)):
            m2 = muscles[j]
            pos2 = m2['rest_vertices']
            fixed2 = set(m2['fixed_vertices'])
            sv2 = sorted(m2['_surface_vert_set'])

            tree2 = cKDTree(pos2[sv2])
            for si, vi in enumerate(sv1):
                nearby = tree2.query_ball_point(pos1[vi], threshold)
                for ni in nearby:
                    vj = sv2[ni]
                    is_fixed1 = vi in fixed1
                    is_fixed2 = vj in fixed2
                    if is_fixed1 != is_fixed2:
                        continue
                    dist = np.linalg.norm(pos1[vi] - pos2[vj])
                    constraints.append((
                        m1['name'], vi, is_fixed1,
                        m2['name'], vj, is_fixed2,
                        dist
                    ))
    return constraints


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def flush_bake_data(bake_buffers, output_dir, chunk_counters):
    """Write buffered frames to NPZ chunk files."""
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
    parser = argparse.ArgumentParser(
        description="Contour tet-mesh bake with surface collision")
    parser.add_argument('--bvh', required=True, help='BVH motion file')
    parser.add_argument('--sides', default='L', help='L, R, or LR')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=None)
    parser.add_argument('--tet-dir', default='tet', help='Tet mesh directory')
    parser.add_argument('--output-dir', default='data/motion_cache')
    parser.add_argument('--youngs', type=float, default=500.0,
                        help='Young\'s modulus (Pa)')
    parser.add_argument('--poisson', type=float, default=0.40,
                        help='Poisson ratio')
    parser.add_argument('--vol-penalty', type=float, default=5000.0,
                        help='Volume stiffness')
    parser.add_argument('--kappa', type=float, default=1e4,
                        help='Collision stiffness')
    parser.add_argument('--settle-iters', type=int, default=150,
                        help='XPBD iterations per frame')
    parser.add_argument('--margin', type=float, default=0.002,
                        help='Collision margin (m)')
    parser.add_argument('--shape-weight', type=float, default=0.0,
                        help='LBS shape-following weight (0=pure XPBD, 1=pure LBS)')
    parser.add_argument('--post-iters', type=int, default=3,
                        help='Edge-bone post-processing iterations')
    parser.add_argument('--constraint-threshold', type=float, default=0.015,
                        help='Inter-muscle constraint distance (m)')
    args = parser.parse_args()

    # ── Load skeleton ────────────────────────────────────────────────────
    print("[1] Loading skeleton...")
    skel, bvh_info, mesh_info = load_skeleton()
    print(f"    DOFs: {skel.getNumDofs()}, Bodies: {skel.getNumBodyNodes()}")

    # ── Load BVH ─────────────────────────────────────────────────────────
    print(f"[2] Loading BVH: {args.bvh}")
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    end_frame = args.end_frame if args.end_frame is not None else n_frames - 1
    end_frame = min(end_frame, n_frames - 1)
    total_frames = end_frame - args.start_frame + 1
    print(f"    Frames: {n_frames}, baking {args.start_frame}-{end_frame} ({total_frames})")

    # ── Load bone meshes ─────────────────────────────────────────────────
    print("[3] Loading bone meshes...")
    bone_meshes = load_bone_meshes(args.sides)
    bone_rest_transforms = compute_bone_rest_transforms(skel, bone_meshes)
    print(f"    {len(bone_meshes)} bones, {len(bone_rest_transforms)} transforms")

    # ── Load muscles ─────────────────────────────────────────────────────
    print(f"[4] Loading muscles from {args.tet_dir}/...")
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

    # ── Precompute surface data ──────────────────────────────────────────
    print("[5] Extracting surface triangles and edges...")
    for m in muscles:
        surf_tri = extract_surface_triangles(m['tetrahedra'])
        surf_edges = extract_edges_from_faces(surf_tri)
        surf_vert_set = sorted(set(np.unique(surf_tri).tolist()))
        m['_surface_faces'] = surf_tri
        m['_surface_edges'] = surf_edges
        m['_surface_vert_set'] = surf_vert_set
        # Surface-only adjacency (triangle edges, not tet edges)
        surf_adj = {}
        for f in surf_tri:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                surf_adj.setdefault(a, set()).add(b)
                surf_adj.setdefault(b, set()).add(a)
        m['_surf_adj'] = surf_adj
    total_surf_faces = sum(len(m['_surface_faces']) for m in muscles)
    total_surf_edges = sum(len(m['_surface_edges']) for m in muscles)
    print(f"    {total_surf_faces} surface faces, {total_surf_edges} surface edges")

    # ── Compute LBS bindings ─────────────────────────────────────────────
    print("[6] Computing multi-bone LBS bindings...")
    for m in muscles:
        m['lbs_bindings'] = compute_multibone_lbs(m, skel)
        if m['lbs_bindings'] is not None:
            bones_used = set()
            for _, bw in m['lbs_bindings']:
                for bname, *_ in bw:
                    bones_used.add(bname)
            print(f"    {m['name']}: {len(m['vertices'])}v, bones={bones_used}")
        else:
            print(f"    {m['name']}: WARNING no LBS bindings!")

    # ── Pre-position at frame-0 LBS ─────────────────────────────────────
    print("[7] Pre-positioning muscles at frame-0 LBS...")
    pose0 = motion_bvh.mocap_refs[args.start_frame].copy()
    skel.setPositions(pose0)
    for m in muscles:
        if m['lbs_bindings'] is not None:
            lbs0 = compute_lbs_positions(m['lbs_bindings'], skel, len(m['vertices']))
            m['vertices'] = lbs0
            m['rest_vertices'] = lbs0.copy()
            # Re-orient tets for deformed config
            tets = m['tetrahedra']
            v0 = lbs0[tets[:, 0]]
            cross = np.cross(lbs0[tets[:, 1]] - v0, lbs0[tets[:, 2]] - v0)
            vol = np.einsum('ij,ij->i', cross, lbs0[tets[:, 3]] - v0) / 6.0
            neg = vol < 0
            if np.any(neg):
                tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
            v0 = lbs0[tets[:, 0]]
            cross = np.cross(lbs0[tets[:, 1]] - v0, lbs0[tets[:, 2]] - v0)
            vol = np.einsum('ij,ij->i', cross, lbs0[tets[:, 3]] - v0) / 6.0
            good = vol > 1e-12
            if not np.all(good):
                n_removed = int(np.sum(~good))
                m['tetrahedra'] = tets[good]
                print(f"    {m['name']}: removed {n_removed} degenerate tets after LBS")
                # Recompute surface data
                m['_surface_faces'] = extract_surface_triangles(m['tetrahedra'])
                m['_surface_edges'] = extract_edges_from_faces(m['_surface_faces'])
                m['_surface_vert_set'] = sorted(set(np.unique(m['_surface_faces']).tolist()))
                surf_adj = {}
                for f in m['_surface_faces']:
                    for i in range(3):
                        a, b = int(f[i]), int(f[(i + 1) % 3])
                        surf_adj.setdefault(a, set()).add(b)
                        surf_adj.setdefault(b, set()).add(a)
                m['_surf_adj'] = surf_adj

    # ── Build XPBD solver ────────────────────────────────────────────────
    print("[8] Building XPBD solver...")
    solver = UnifiedFEMSolver()

    muscles_data = {}
    for m in muscles:
        n_v = len(m['vertices'])
        fixed_mask = np.zeros(n_v, dtype=bool)
        for vi in m['fixed_vertices']:
            if vi < n_v:
                fixed_mask[vi] = True

        # Surface faces need global indexing — solver.build handles offsets
        surf_faces = m['_surface_faces'].astype(np.int32)

        muscles_data[m['name']] = {
            'rest_positions': m['rest_vertices'],
            'tetrahedra': m['tetrahedra'],
            'fixed_mask': fixed_mask,
            'surface_faces': surf_faces,
        }

    solver.build(muscles_data)

    # ── Inter-muscle constraints ─────────────────────────────────────────
    print("[9] Finding inter-muscle constraints...")
    constraints = find_inter_muscle_constraints(muscles, args.constraint_threshold)
    if constraints:
        solver.set_inter_muscle_constraints(constraints)
    print(f"    {len(constraints)} constraints")

    # Build tet adjacency for Laplacian smoothing of LBS targets
    for m in muscles:
        adj = [set() for _ in range(len(m['vertices']))]
        for t in m['tetrahedra']:
            for i in range(4):
                for j in range(i + 1, 4):
                    adj[t[i]].add(t[j])
                    adj[t[j]].add(t[i])
        m['_adj'] = adj

    # ── Compute Lamé parameters ──────────────────────────────────────────
    mu = args.youngs / (2.0 * (1.0 + args.poisson))
    lam = args.youngs * args.poisson / ((1.0 + args.poisson) * (1.0 - 2.0 * args.poisson))

    # ── Setup output ─────────────────────────────────────────────────────
    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    side_tag = ''.join(args.sides)
    out_dir = os.path.join(args.output_dir, bvh_stem, f"contour_{side_tag}_UpLeg")
    os.makedirs(out_dir, exist_ok=True)
    print(f"    Output: {out_dir}")

    bake_buffers = {m['name']: {} for m in muscles}
    chunk_counters = {m['name']: 0 for m in muscles}

    # ── Bake loop ────────────────────────────────────────────────────────
    print(f"\n[10] Baking frames {args.start_frame}-{end_frame}...")
    t_total = time.time()

    for frame in range(args.start_frame, end_frame + 1):
        t0 = time.time()

        # Set skeleton pose
        pose = motion_bvh.mocap_refs[frame].copy()
        skel.setPositions(pose)

        # Compute LBS positions for all muscles
        muscles_update = {}
        for m in muscles:
            if m['lbs_bindings'] is None:
                continue
            lbs_pos = compute_lbs_positions(m['lbs_bindings'], skel, len(m['vertices']))

            # Laplacian smooth LBS targets (3 iterations)
            adj = m['_adj']
            fixed_set = set(m['fixed_vertices'])
            for _ in range(3):
                smoothed = lbs_pos.copy()
                for vi in range(len(lbs_pos)):
                    if vi in fixed_set or len(adj[vi]) == 0:
                        continue
                    smoothed[vi] = 0.5 * lbs_pos[vi] + 0.5 * np.mean(
                        lbs_pos[list(adj[vi])], axis=0)
                lbs_pos = smoothed

            # Split into positions (all) and fixed_targets (fixed only)
            n_v = len(m['vertices'])
            fixed_mask = np.zeros(n_v, dtype=bool)
            for vi in m['fixed_vertices']:
                if vi < n_v:
                    fixed_mask[vi] = True
            fixed_targets = lbs_pos[fixed_mask]

            muscles_update[m['name']] = {
                'positions': lbs_pos,
                'fixed_targets': fixed_targets,
            }

        # Update rest shape to current LBS each frame.
        # This makes muscles follow bones rigidly (F≈I when pos≈LBS).
        # Elastic energy only appears where collisions or attachment distance
        # changes force deviation from LBS — matching real muscle behavior
        # (nearly rigid body that deforms when origin/insertion move).
        for m in muscles:
            if m['name'] not in muscles_update:
                continue
            vs, ve, _, _ = solver._muscle_ranges[m['name']]
            solver.rest_positions[vs:ve] = muscles_update[m['name']]['positions']
        solver._precompute_rest_state()
        solver._tet_mass = 1060.0 * solver.rest_volume
        solver.ti_Bm_inv.from_numpy(solver.Bm_inv)
        solver.ti_rest_volume.from_numpy(solver.rest_volume)

        # LBS-init positions
        solver._has_previous_solution = False
        solver.update_targets_and_positions(muscles_update)

        # Build bone trimeshes at current pose
        bone_tms = build_bone_trimeshes(skel, bone_meshes, bone_rest_transforms)

        # XPBD solve — starts at LBS with rest=LBS, so F≈I initially.
        # Only collision projection and constraint violations create F≠I,
        # which the elastic solve distributes smoothly.
        fevals, residual = solver.solve(
            mu=mu, lam=lam,
            vol_penalty=args.vol_penalty,
            kappa=args.kappa,
            margin=args.margin,
            max_lbfgs_iters=args.settle_iters,
            bone_meshes=bone_tms,
            verbose=(frame == args.start_frame),
        )

        # Snapshot XPBD-converged positions (clean state for next frame)
        xpbd_positions = {m['name']: solver.get_muscle_positions(m['name']).copy()
                          for m in muscles}

        # Post-solve: surface collision on a COPY (output only,
        # don't pollute solver state which would cause cascading inversions)
        n_vert_pushed, n_edge_resolved = resolve_surface_collisions(
            solver, muscles, bone_tms,
            margin=args.margin,
            max_iters=args.post_iters,
            verbose=(frame == args.start_frame),
        )

        # Post-solve: muscle-muscle push-apart (also on output copy)
        n_mm_pushed = resolve_muscle_muscle_collisions(
            solver, muscles, d_min=args.margin)

        # Blend XPBD result toward LBS for bone-following (like ARAP targets).
        # XPBD only constrains fixed vertices; this pulls free vertices toward
        # their LBS positions so muscles follow bones, not float in air.
        if args.shape_weight > 0:
            for m in muscles:
                name = m['name']
                vs, ve, _, _ = solver._muscle_ranges[name]
                xpbd_pos = solver.positions[vs:ve]
                lbs_pos = muscles_update[name]['positions']
                fixed_set = set(m['fixed_vertices'])
                w = args.shape_weight
                blended = (1.0 - w) * xpbd_pos + w * lbs_pos
                # Keep fixed vertices exactly at XPBD result (bone-locked)
                for vi in fixed_set:
                    if vi < len(blended):
                        blended[vi] = xpbd_pos[vi]
                solver.positions[vs:ve] = blended

        # Taubin smoothing to reduce wrinkling on output
        smooth_output_positions(solver, muscles, iterations=3)

        # Capture output positions (with blending + collision + smoothing)
        output_positions = {m['name']: solver.get_muscle_positions(m['name']).astype(np.float32)
                            for m in muscles}

        # Restore solver to XPBD-converged state (clean carry-forward)
        for m in muscles:
            vs, ve, _, _ = solver._muscle_ranges[m['name']]
            solver.positions[vs:ve] = xpbd_positions[m['name']]

        dt = time.time() - t0

        # Count inversions (on XPBD-converged positions)
        total_inv = 0
        for m in muscles:
            pos = xpbd_positions[m['name']]
            tets = m['tetrahedra']
            if len(tets) > 0:
                v0 = pos[tets[:, 0]]
                cr = np.cross(pos[tets[:, 1]] - v0, pos[tets[:, 2]] - v0)
                vol = np.einsum('ij,ij->i', cr, pos[tets[:, 3]] - v0) / 6.0
                total_inv += int(np.sum(vol <= 0))

        print(f"  Frame {frame}: {dt:.2f}s, iters={fevals}, "
              f"||dx||={residual:.2e}, inv={total_inv}/{total_tets}, "
              f"vert_coll={n_vert_pushed}, edge_coll={n_edge_resolved}, mm={n_mm_pushed}",
              flush=True)

        # Buffer output positions (no remap needed — all vertices kept)
        for m in muscles:
            bake_buffers[m['name']][frame] = output_positions[m['name']]

        # Flush periodically
        if (frame - args.start_frame + 1) % FLUSH_INTERVAL == 0:
            flush_bake_data(bake_buffers, out_dir, chunk_counters)
            gc.collect()

    # Final flush
    flush_bake_data(bake_buffers, out_dir, chunk_counters)

    elapsed = time.time() - t_total
    print(f"\nDone. {len(muscles)} muscles, {total_frames} frames, "
          f"{elapsed:.1f}s total ({elapsed/max(total_frames,1):.2f}s/frame)")
    print(f"Output: {out_dir}")


if __name__ == '__main__':
    main()
