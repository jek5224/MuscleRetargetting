#!/usr/bin/env python3
"""Layered ARAP bake with surface collision projection (Iron Man approach).

Muscles start OUTSIDE bones and are pulled in by ARAP toward attachment points.
Collision projection (every 5 ARAP iterations) prevents vertices from crossing
bone/obstacle surfaces. Direction is always unambiguous — vertices approach from
outside, never start inside.

Deep muscles settle first → become rigid obstacles for mid/superficial layers.

Usage:
    python tools/bake_layered.py --bvh data/motion/walk.bvh --sides L --start-frame 60 --end-frame 70
"""
import argparse
import gc
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace
from core.dartHelper import saveSkeletonInfo, buildFromInfo
from core.bvhparser import MyBVH
from viewer.mesh_loader import MeshLoader
from viewer.zygote_mesh_ui import (
    find_inter_muscle_constraints,
    _detect_bvh_tframe,
    _flatten_waypoints,
)
from viewer.arap_backends import check_taichi_available, check_gpu_available, get_backend


# ---------------------------------------------------------------------------
# Warp GPU collision detection
# ---------------------------------------------------------------------------
_warp_initialized = [False]
_warp_ok = [False]
_warp_kernel_compiled = [False]


def _warp_available():
    if _warp_initialized[0]:
        return _warp_ok[0]
    _warp_initialized[0] = True
    try:
        import warp as wp
        wp.init()
        if wp.is_cuda_available():
            _warp_ok[0] = True
            _compile_warp_kernel()
    except Exception:
        pass
    return _warp_ok[0]


def _compile_warp_kernel():
    """Compile the warp mesh query kernel (once)."""
    import warp as wp
    global _warp_query_kernel

    @wp.kernel
    def mesh_query_kernel(
        points: wp.array(dtype=wp.vec3),
        mesh: wp.uint64,
        max_dist: float,
        closest: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        signed_dists: wp.array(dtype=float),
    ):
        i = wp.tid()
        p = points[i]

        face = int(0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)

        found = wp.mesh_query_point_sign_normal(mesh, p, max_dist, sign, face, u, v)

        if found:
            i0 = wp.mesh_get_index(mesh, face * 3)
            i1 = wp.mesh_get_index(mesh, face * 3 + 1)
            i2 = wp.mesh_get_index(mesh, face * 3 + 2)
            v0 = wp.mesh_get_point(mesh, i0)
            v1 = wp.mesh_get_point(mesh, i1)
            v2 = wp.mesh_get_point(mesh, i2)

            cp = v0 * (1.0 - u - v) + v1 * u + v2 * v
            closest[i] = cp

            e1 = v1 - v0
            e2 = v2 - v0
            n = wp.normalize(wp.cross(e1, e2))
            normals[i] = n

            dist = wp.length(p - cp)
            signed_dists[i] = sign * dist
        else:
            closest[i] = p
            normals[i] = wp.vec3(0.0, 1.0, 0.0)
            signed_dists[i] = max_dist

    _warp_query_kernel = mesh_query_kernel
    _warp_kernel_compiled[0] = True


_warp_mesh_cache = {}  # id(trimesh) -> wp.Mesh


def _warp_mesh_query(obs_mesh, query_points):
    """GPU-accelerated closest point + signed distance using warp.

    Caches wp.Mesh objects by trimesh identity for reuse within a frame.
    Returns (closest_points, normals, signed_dists) as numpy arrays.
    """
    import warp as wp

    device = "cuda:0"
    n = len(query_points)

    # Cache warp mesh (avoid rebuilding BVH for same bone within same frame)
    mesh_id = id(obs_mesh)
    if mesh_id not in _warp_mesh_cache:
        verts = obs_mesh.vertices.astype(np.float32)
        faces = obs_mesh.faces.astype(np.int32).flatten()
        _warp_mesh_cache[mesh_id] = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device=device),
            indices=wp.array(faces, dtype=wp.int32, device=device),
        )
    wp_mesh = _warp_mesh_cache[mesh_id]

    wp_query = wp.array(query_points.astype(np.float32), dtype=wp.vec3, device=device)
    wp_closest = wp.zeros(n, dtype=wp.vec3, device=device)
    wp_normals = wp.zeros(n, dtype=wp.vec3, device=device)
    wp_signed = wp.zeros(n, dtype=float, device=device)

    wp.launch(
        _warp_query_kernel, dim=n, device=device,
        inputs=[wp_query, wp_mesh.id, float(0.01), wp_closest, wp_normals, wp_signed],
    )
    wp.synchronize()

    closest = wp_closest.numpy()
    normals = wp_normals.numpy()
    signed_dists = wp_signed.numpy()

    return closest, normals, signed_dists


def _warp_clear_cache():
    """Clear warp mesh cache (call at start of each frame)."""
    _warp_mesh_cache.clear()


SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229/"
ZYGOTE_HIRES_DIR = "Zygote_Meshes/"  # Fine-grained meshes for collision
MESH_SCALE = 0.01
FLUSH_INTERVAL = 20

# Muscle depth layers
LAYERS = {
    0: [  # Deep
        "Iliacus", "Obturator_Internus", "Obturator_Externus",
        "Inferior_Gemellus", "Superior_Gemellus", "Quadratus_Femoris",
        "Piriformis", "Popliteus", "Vastus_Intermedius", "Gluteus_Minimus",
    ],
    1: [  # Mid
        "Adductor_Brevis", "Adductor_Longus", "Adductor_Magnus",
        "Pectineus", "Vastus_Medialis", "Vastus_Lateralis",
    ],
    2: [  # Superficial
        "Gluteus_Maximus", "Gluteus_Medius", "Rectus_Femoris",
        "Sartorius", "Gracilis", "Biceps_Femoris",
        "Semimembranosus", "Semitendinosus", "Tensor_Fascia_Lata",
    ],
}


# ---------------------------------------------------------------------------
# Surface / collision helpers
# ---------------------------------------------------------------------------
def extract_surface_triangles(tet_elements):
    """Extract boundary faces from tet mesh (faces belonging to exactly 1 tet)."""
    face_count = Counter()
    face_orient = {}
    for t in tet_elements:
        v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        faces = [(v0, v2, v1), (v0, v1, v3), (v1, v2, v3), (v0, v3, v2)]
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] += 1
            if key not in face_orient:
                face_orient[key] = f
    return np.array([face_orient[k] for k, c in face_count.items() if c == 1],
                    dtype=np.int64)


def extract_edges_from_faces(faces):
    """Extract unique edges from surface faces."""
    edges = set()
    for f in faces:
        for i in range(3):
            e = (min(int(f[i]), int(f[(i + 1) % 3])),
                 max(int(f[i]), int(f[(i + 1) % 3])))
            edges.add(e)
    return np.array(sorted(edges), dtype=np.int64)


def precompute_surface_data(active_muscles):
    """Precompute surface triangles and edges for each muscle."""
    for mname, mobj in active_muscles.items():
        tets = getattr(mobj, 'tet_tetrahedra', None)
        if tets is None and hasattr(mobj, 'soft_body'):
            tets = getattr(mobj.soft_body, 'tetrahedra', None)
        if tets is None:
            continue
        sf = extract_surface_triangles(tets)
        se = extract_edges_from_faces(sf)
        sv = sorted(set(np.unique(sf).tolist()))
        mobj._surf_faces = sf
        mobj._surf_edges = se
        mobj._surf_verts = sv
        fixed_set = set()
        if hasattr(mobj, 'soft_body') and mobj.soft_body is not None:
            fi = mobj.soft_body.fixed_indices
            if fi is not None:
                fixed_set = set(int(i) for i in fi)
        mobj._surf_fixed = fixed_set


def subdivide_long_edges_near_bones(mobj, bone_trimeshes, max_edge_len=0.010, bone_dist=0.015):
    """Subdivide long surface edges near bones in the tet mesh.

    For each surface edge > max_edge_len that is within bone_dist of any bone:
    - Add midpoint vertex
    - Split all tets containing the edge into 2 tets each
    - Split surface faces containing the edge into 2 faces each
    - Interpolate per-vertex data (vertex_contour_level)

    Modifies mobj.tet_vertices, mobj.tet_tetrahedra, etc. in place.
    Returns number of edges subdivided.
    """
    from scipy.spatial import cKDTree

    verts = mobj.tet_vertices.copy()  # (N, 3) float32
    tets = mobj.tet_tetrahedra.copy()  # (T, 4) int32
    if verts is None or tets is None:
        return 0

    # Get surface faces and edges
    sf = extract_surface_triangles(tets)
    se = extract_edges_from_faces(sf)

    # Tet vertices are already in meters
    verts_m = verts

    # Find long edges near bones
    v0_pos = verts_m[se[:, 0]]
    v1_pos = verts_m[se[:, 1]]
    edge_lens = np.linalg.norm(v1_pos - v0_pos, axis=1)
    long_mask = edge_lens > max_edge_len

    if not np.any(long_mask):
        return 0

    # KDTree from all bone vertices for proximity check
    if not bone_trimeshes:
        return 0
    all_bone_verts = np.vstack([bm.vertices for bm in bone_trimeshes])
    bone_kdtree = cKDTree(all_bone_verts)

    long_edges = se[long_mask]
    long_mids = 0.5 * (verts_m[long_edges[:, 0]] + verts_m[long_edges[:, 1]])
    d_mid, _ = bone_kdtree.query(long_mids)
    near_bone = d_mid < bone_dist

    edges_to_split = long_edges[near_bone]
    if len(edges_to_split) == 0:
        return 0

    # Build edge → tet mapping
    edge_set = set()
    for e in edges_to_split:
        edge_set.add((min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1]))))

    # Per-vertex data to interpolate
    vcl = getattr(mobj, '_tet_vertex_contour_level', None)

    new_verts = list(verts)
    new_tets = []
    new_vcl = list(vcl) if vcl is not None else None
    midpoint_cache = {}  # (v0, v1) → new vertex index

    def get_midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid_pos = 0.5 * (verts[a] + verts[b])
        new_idx = len(new_verts)
        new_verts.append(mid_pos)
        if new_vcl is not None and vcl is not None:
            new_vcl.append(0.5 * (vcl[a] + vcl[b]))
        midpoint_cache[key] = new_idx
        return new_idx

    # Process each tet
    for t in tets:
        t = [int(x) for x in t]
        # Find which edges of this tet need splitting
        tet_edges = [(t[i], t[j]) for i in range(4) for j in range(i+1, 4)]
        splits = []
        for a, b in tet_edges:
            key = (min(a, b), max(a, b))
            if key in edge_set:
                splits.append((a, b))

        if not splits:
            new_tets.append(t)
            continue

        if len(splits) == 1:
            # Split one edge → tet becomes 2 tets
            a, b = splits[0]
            m = get_midpoint(a, b)
            # Other two vertices
            others = [v for v in t if v != a and v != b]
            c, d = others
            new_tets.append([a, m, c, d])
            new_tets.append([m, b, c, d])
        else:
            # Multiple edges split — keep tet as-is for simplicity
            # (multi-edge splits produce complex configurations)
            new_tets.append(t)

    # Update surface faces
    new_faces = []
    for f in sf:
        f = [int(x) for x in f]
        face_edges = [(f[i], f[(i+1) % 3]) for i in range(3)]
        face_splits = []
        for a, b in face_edges:
            key = (min(a, b), max(a, b))
            if key in midpoint_cache:
                face_splits.append((a, b, midpoint_cache[key]))

        if not face_splits:
            new_faces.append(f)
        elif len(face_splits) == 1:
            a, b, m = face_splits[0]
            c = [v for v in f if v != a and v != b][0]
            new_faces.append([a, m, c])
            new_faces.append([m, b, c])
        else:
            new_faces.append(f)  # Multi-split face: keep as-is

    # Write back
    mobj.tet_vertices = np.array(new_verts, dtype=np.float32)
    mobj.tet_tetrahedra = np.array(new_tets, dtype=np.int32)

    # Update render_faces (same as surface faces for contour meshes)
    new_faces_arr = np.array(new_faces, dtype=np.int32)
    mobj.tet_render_faces = new_faces_arr
    mobj.tet_sim_faces = new_faces_arr

    if new_vcl is not None:
        mobj._tet_vertex_contour_level = np.array(new_vcl, dtype=np.float32)

    return len(midpoint_cache)


def cache_bone_rest_transforms(skel):
    """Cache rest-pose transforms for all body nodes."""
    saved_pos = skel.getPositions().copy()
    skel.setPositions(np.zeros(skel.getNumDofs()))
    rest_transforms = {}
    for i in range(skel.getNumBodyNodes()):
        bn = skel.getBodyNode(i)
        wt = bn.getWorldTransform()
        rest_transforms[bn.getName()] = (wt.rotation().copy(), wt.translation().copy())
    skel.setPositions(saved_pos)
    return rest_transforms


def load_hires_skeleton_meshes():
    """Load fine-grained skeleton meshes for collision detection."""
    hires_dir = os.path.join(ZYGOTE_HIRES_DIR, "Skeleton")
    if not os.path.isdir(hires_dir):
        return {}
    # Naming: "Legs_L_Femur.obj" → body name "L_Femur0"
    # Also: "Legs_L_Tibia.obj" + "Legs_L_Fibula.obj" (separate in hires)
    meshes = {}
    for fname in sorted(os.listdir(hires_dir)):
        if not fname.endswith(".obj"):
            continue
        # Strip prefix: "Legs_L_Femur.obj" → "L_Femur"
        name = fname.replace(".obj", "")
        if name.startswith("Legs_"):
            name = name[5:]  # "L_Femur"
        elif name.startswith("Hips_"):
            name = name[5:]
        elif name.startswith("Spine_"):
            name = name[6:]
        path = os.path.join(hires_dir, fname)
        skel_tri = trimesh.load_mesh(path)
        skel_tri.vertices *= MESH_SCALE
        meshes[name] = SimpleNamespace(trimesh=skel_tri)
    if meshes:
        print(f"    Loaded {len(meshes)} hi-res skeleton meshes for collision")
        # Show vertex counts for key bones
        for key in ['L_Femur', 'L_Tibia', 'L_Fibula', 'L_Os_Coxae', 'L_Patella']:
            if key in meshes:
                print(f"      {key}: {meshes[key].trimesh.vertices.shape[0]} verts")
    return meshes


def build_bone_collision_meshes(skeleton_meshes, skel, rest_transforms):
    """Build bone trimeshes at current skeleton pose."""
    # Name mapping: hi-res mesh name → DART body name
    BODY_NAME_MAP = {
        'L_Tibia': 'L_Tibia_Fibula0', 'L_Fibula': 'L_Tibia_Fibula0',
        'R_Tibia': 'R_Tibia_Fibula0', 'R_Fibula': 'R_Tibia_Fibula0',
        'Saccrum_Coccyx': 'Saccrum_Coccyx0', 'Saccrum': 'Saccrum_Coccyx0',
    }
    bone_meshes = []
    for mesh_name, mesh_obj in skeleton_meshes.items():
        if not (hasattr(mesh_obj, 'trimesh') and mesh_obj.trimesh is not None):
            continue
        body_node = None
        body_name = None
        # Try direct name, name+0, and explicit mapping
        for candidate in [mesh_name, mesh_name + '0', BODY_NAME_MAP.get(mesh_name, '')]:
            if not candidate:
                continue
            body_node = skel.getBodyNode(candidate)
            if body_node is not None:
                body_name = candidate
                break
        if body_node is None:
            continue
        R_rest, t_rest = rest_transforms.get(body_name, (np.eye(3), np.zeros(3)))
        R_posed = body_node.getWorldTransform().rotation()
        t_posed = body_node.getWorldTransform().translation()
        verts = mesh_obj.trimesh.vertices.copy()
        local_verts = (R_rest.T @ (verts - t_rest).T).T
        posed_verts = (R_posed @ local_verts.T).T + t_posed
        tm = trimesh.Trimesh(vertices=posed_verts,
                             faces=mesh_obj.trimesh.faces.copy(), process=True)
        bone_meshes.append(tm)
    return bone_meshes


def collision_project(positions, obstacle_meshes, collision_vertex_set,
                      surface_edges, fixed_mask, margin=0.002):
    """Project penetrating vertices/edges to bone surface + margin.

    Phase 1: Vertex-bone via contains() (reliable inside/outside).
    Phase 2: Edge-bone via ray-cast (detects edge tunneling through bones).
    Does NOT modify ARAP system — operates on positions only.
    Returns (corrected_positions, n_projected).
    """
    from scipy.spatial import cKDTree

    sv_arr = np.array(sorted(collision_vertex_set), dtype=np.int64)
    sv_pos = positions[sv_arr]
    n_projected = 0

    # Phase 1: Vertex-bone collision via contains()
    for obs_mesh in obstacle_meshes:
        bmin = obs_mesh.bounds[0] - 0.005
        bmax = obs_mesh.bounds[1] + 0.005
        in_bbox = np.all((sv_pos >= bmin) & (sv_pos <= bmax), axis=1)
        if not np.any(in_bbox):
            continue

        bbox_sv = sv_arr[in_bbox]
        bbox_pos = sv_pos[in_bbox]

        try:
            inside = obs_mesh.contains(bbox_pos)
        except Exception:
            continue
        if not np.any(inside):
            continue

        inside_sv = bbox_sv[inside]
        inside_pos = bbox_pos[inside]
        closest, _, face_ids = trimesh.proximity.closest_point(obs_mesh, inside_pos)
        normals = obs_mesh.face_normals[face_ids]

        for k in range(len(inside_sv)):
            vi = int(inside_sv[k])
            if fixed_mask[vi]:
                continue
            positions[vi] = closest[k] + normals[k] * margin
            local_idx = np.searchsorted(sv_arr, vi)
            if local_idx < len(sv_pos) and sv_arr[local_idx] == vi:
                sv_pos[local_idx] = positions[vi]
            n_projected += 1

    return positions, n_projected


def _detect_collisions(positions, obstacle_meshes, collision_vertex_set,
                       surface_edges, fixed_mask, margin, out_targets,
                       depth_threshold=0.0005):
    """Detect muscle-vertex-in-bone collisions.

    out_targets: {global_vertex_idx: target_position (surface + margin)}
    """
    from scipy.spatial import cKDTree

    sv_arr = np.array(sorted(collision_vertex_set), dtype=np.int64)
    sv_pos = positions[sv_arr]

    # Phase 1: Muscle vertices inside bones → push to surface + margin
    for obs_mesh in obstacle_meshes:
        bmin = obs_mesh.bounds[0]
        bmax = obs_mesh.bounds[1]
        in_bbox = np.all((sv_pos >= bmin) & (sv_pos <= bmax), axis=1)
        if not np.any(in_bbox):
            continue
        bbox_sv = sv_arr[in_bbox]
        bbox_pos = sv_pos[in_bbox]

        bone_kdtree = cKDTree(obs_mesh.vertices)
        kd_dists, _ = bone_kdtree.query(bbox_pos)
        near_surf = kd_dists < 0.008
        if not np.any(near_surf):
            continue
        near_sv = bbox_sv[near_surf]
        near_pos = bbox_pos[near_surf]

        try:
            inside = obs_mesh.contains(near_pos)
        except Exception:
            continue
        if not np.any(inside):
            continue

        inside_sv = near_sv[inside]
        inside_pos = near_pos[inside]
        closest, _, face_ids = trimesh.proximity.closest_point(obs_mesh, inside_pos)
        normals = obs_mesh.face_normals[face_ids]
        depths = np.linalg.norm(inside_pos - closest, axis=1)

        deep = depths > depth_threshold
        if not np.any(deep):
            continue

        inside_sv = inside_sv[deep]
        inside_closest = closest[deep]
        inside_normals = normals[deep]
        for k in range(len(inside_sv)):
            vi = int(inside_sv[k])
            if fixed_mask[vi]:
                continue
            out_targets[vi] = inside_closest[k] + inside_normals[k] * margin



def _local_arap_resolve(positions, rest_positions, neighbors, edge_weights,
                         rest_edge_vectors, fixed_mask, collision_targets,
                         n_rings, collision_weight=5.0, max_iterations=50, tolerance=1e-4):
    """Re-solve local ARAP patches around collision vertices.

    Grows collision vertices by n_rings to form a patch. Boundary of patch
    is fixed at current (ARAP) positions. Collision vertices get a soft
    pull toward their targets. Interior vertices are free to adjust smoothly.
    """
    # Grow collision vertices by n_rings
    collision_verts = set(collision_targets.keys())
    patch_verts = set(collision_verts)
    frontier = set(collision_verts)

    for _ in range(n_rings):
        new_frontier = set()
        for vi in frontier:
            if vi < len(neighbors):
                for nj in neighbors[vi]:
                    if nj not in patch_verts:
                        patch_verts.add(nj)
                        new_frontier.add(nj)
        frontier = new_frontier

    if not patch_verts:
        return

    # Classify patch vertices
    # Boundary: vertices at the edge of the patch (in patch but have neighbor outside)
    # Interior: collision verts + their interior neighbors
    patch_list = sorted(patch_verts)
    patch_set = set(patch_list)

    boundary = set()
    for vi in patch_list:
        if fixed_mask[vi]:
            boundary.add(vi)
            continue
        if vi < len(neighbors):
            for nj in neighbors[vi]:
                if nj not in patch_set:
                    boundary.add(vi)
                    break

    interior = patch_set - boundary

    if not interior:
        return

    # Build local index mapping
    local_idx = {vi: li for li, vi in enumerate(patch_list)}
    n_local = len(patch_list)

    # Local fixed mask: boundary vertices are fixed
    local_fixed = np.array([vi in boundary for vi in patch_list])

    # Local positions
    local_pos = positions[patch_list].copy()
    local_rest = rest_positions[patch_list].copy()
    local_fixed_targets = local_pos[local_fixed].copy()

    # Build local edges
    local_neighbors = [[] for _ in range(n_local)]
    local_edge_weights = {}
    local_rest_edges = [{} for _ in range(n_local)]

    for vi in patch_list:
        li = local_idx[vi]
        if vi >= len(neighbors):
            continue
        for nj in neighbors[vi]:
            if nj in patch_set:
                lj = local_idx[nj]
                if lj not in local_neighbors[li]:
                    local_neighbors[li].append(lj)
                w = edge_weights.get((vi, nj), 1.0)
                local_edge_weights[(li, lj)] = w
                re = rest_edge_vectors[vi].get(nj, local_rest[lj] - local_rest[li])
                local_rest_edges[li][lj] = re

    # Collision vertices get soft target via penalty
    coll_local_set = set()
    for vi in collision_targets:
        if vi in local_idx:
            coll_local_set.add(local_idx[vi])

    # Solve local ARAP with collision as soft penalty
    local_backend = get_backend('taichi')
    local_backend.build_system(n_local, local_neighbors, local_edge_weights,
                                local_fixed, regularization=1e-6,
                                collision_vertices=coll_local_set,
                                collision_weight=collision_weight)

    # Collision target function for local system
    def local_coll_fn(pos):
        targets = {}
        for vi, target in collision_targets.items():
            if vi in local_idx:
                li = local_idx[vi]
                if not local_fixed[li]:
                    targets[li] = target
        # Free collision candidates default to current pos (zero force)
        for li in coll_local_set:
            if li not in targets:
                targets[li] = pos[li].copy()
        return targets

    fixed_indices_local = np.where(local_fixed)[0]
    local_pos, _, _ = local_backend.solve(
        local_pos, local_rest, local_neighbors, local_edge_weights, local_rest_edges,
        local_fixed, local_fixed_targets,
        max_iterations=max_iterations, tolerance=tolerance,
        collision_target_fn=local_coll_fn,
    )

    # Write back only interior vertices
    for vi in interior:
        li = local_idx[vi]
        positions[vi] = local_pos[li]


# ---------------------------------------------------------------------------
# Per-layer unified ARAP solve with collision projection
# ---------------------------------------------------------------------------
def run_layer_sim_with_collision(layer_muscles, frozen_muscles, skeleton_meshes, skel,
                                  obstacle_meshes, inter_muscle_constraints,
                                  layer_cache, backend_name,
                                  max_iterations=100, tolerance=1e-4,
                                  collision_margin=0.002, frame_independent=False,
                                  verbose=False):
    """Run unified ARAP for one layer with collision projection.

    Muscles start outside obstacles. ARAP pulls toward attachments.
    Collision projection (every 5 iters) prevents surface crossing.

    frozen_muscles: dict of earlier-layer muscles (already settled). Included in
    system as fixed vertices so cross-layer distance constraints work.
    """
    muscle_names = list(layer_muscles.keys())
    frozen_names = list(frozen_muscles.keys())
    all_names = muscle_names + frozen_names
    total_verts = (sum(layer_muscles[n].soft_body.num_vertices for n in muscle_names)
                   + sum(frozen_muscles[n].soft_body.num_vertices for n in frozen_names))

    # Step 1: Update positions and fixed targets from skeleton
    for name, mobj in layer_muscles.items():
        if hasattr(mobj, '_update_tet_positions_from_skeleton'):
            mobj._update_tet_positions_from_skeleton(skel)
        if hasattr(mobj, '_update_fixed_targets_from_skeleton'):
            mobj._update_fixed_targets_from_skeleton(skeleton_meshes, skel)

    # Check if cached topology is still valid
    cache_valid = (layer_cache.get('muscle_names') == muscle_names
                   and layer_cache.get('frozen_names') == frozen_names
                   and layer_cache.get('total_verts') == total_verts)

    if not cache_valid:
        global_offset = {}
        offset_accum = 0
        # Current layer muscles first
        for name in muscle_names:
            global_offset[name] = offset_accum
            offset_accum += layer_muscles[name].soft_body.num_vertices
        # Frozen (earlier layer) muscles after
        for name in frozen_names:
            global_offset[name] = offset_accum
            offset_accum += frozen_muscles[name].soft_body.num_vertices

        global_rest_positions = np.zeros((total_verts, 3))
        global_fixed_mask = np.zeros(total_verts, dtype=bool)

        # Current layer: use their own fixed_mask
        for name, mobj in layer_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            global_rest_positions[offset:offset+n] = mobj.soft_body.rest_positions
            global_fixed_mask[offset:offset+n] = mobj.soft_body.fixed_mask

        # Frozen muscles: ALL vertices are fixed
        for name, mobj in frozen_muscles.items():
            offset = global_offset[name]
            n = mobj.soft_body.num_vertices
            global_rest_positions[offset:offset+n] = mobj.soft_body.rest_positions
            global_fixed_mask[offset:offset+n] = True  # All frozen

        # Build edges from tet connectivity (current layer only)
        all_edges = []
        for name, mobj in layer_muscles.items():
            offset = global_offset[name]
            sb = mobj.soft_body
            for edge_idx, (i, j) in enumerate(zip(sb.edge_i, sb.edge_j)):
                if hasattr(sb, 'rest_lengths') and sb.rest_lengths is not None and edge_idx < len(sb.rest_lengths):
                    rest_len = sb.rest_lengths[edge_idx]
                else:
                    rest_len = np.linalg.norm(sb.rest_positions[j] - sb.rest_positions[i])
                all_edges.append((offset + i, offset + j, rest_len, 1.0))

        # Inter-muscle constraints as edges (within-layer AND cross-layer)
        all_muscle_set = set(all_names)
        n_within = 0
        n_cross = 0
        for constraint in inter_muscle_constraints:
            name1, v1_idx, v1_fixed, name2, v2_idx, v2_fixed, rest_dist = constraint
            if name1 in all_muscle_set and name2 in all_muscle_set:
                gi = global_offset[name1] + v1_idx
                gj = global_offset[name2] + v2_idx
                all_edges.append((gi, gj, rest_dist, 1.0))
                if name1 in set(muscle_names) and name2 in set(muscle_names):
                    n_within += 1
                else:
                    n_cross += 1

        neighbors = [[] for _ in range(total_verts)]
        edge_weights = {}
        rest_edge_vectors = [{} for _ in range(total_verts)]
        for gi, gj, rest_len, weight in all_edges:
            neighbors[gi].append(gj)
            neighbors[gj].append(gi)
            edge_weights[(gi, gj)] = weight
            edge_weights[(gj, gi)] = weight
            rest_edge_vectors[gi][gj] = global_rest_positions[gj] - global_rest_positions[gi]
            rest_edge_vectors[gj][gi] = global_rest_positions[gi] - global_rest_positions[gj]

        # Build collision vertex set and surface edges (current layer only)
        collision_vertex_set = set()
        global_surf_edges_list = []
        for name, mobj in layer_muscles.items():
            offset = global_offset[name]
            if hasattr(mobj, '_surf_verts'):
                for lv in mobj._surf_verts:
                    gv = offset + lv
                    if not global_fixed_mask[gv]:
                        collision_vertex_set.add(gv)
            if hasattr(mobj, '_surf_edges'):
                for e in mobj._surf_edges:
                    global_surf_edges_list.append([offset + int(e[0]), offset + int(e[1])])
        global_surf_edges = np.array(global_surf_edges_list, dtype=np.int64) if global_surf_edges_list else np.zeros((0, 2), dtype=np.int64)

        layer_cache.update({
            'global_offset': global_offset,
            'total_verts': total_verts,
            'global_rest_positions': global_rest_positions,
            'global_fixed_mask': global_fixed_mask,
            'neighbors': neighbors,
            'edge_weights': edge_weights,
            'rest_edge_vectors': rest_edge_vectors,
            'muscle_names': muscle_names,
            'frozen_names': frozen_names,
            'collision_vertex_set': collision_vertex_set,
            'global_surf_edges': global_surf_edges,
        })
        if verbose:
            print(f"    Built topology: {total_verts} verts, {len(all_edges)} edges, "
                  f"{len(collision_vertex_set)} collision candidates, "
                  f"{n_within} within-layer + {n_cross} cross-layer constraints")

    # Unpack cache
    global_offset = layer_cache['global_offset']
    global_rest_positions = layer_cache['global_rest_positions']
    global_fixed_mask = layer_cache['global_fixed_mask']
    neighbors = layer_cache['neighbors']
    edge_weights = layer_cache['edge_weights']
    rest_edge_vectors = layer_cache['rest_edge_vectors']
    collision_vertex_set = layer_cache['collision_vertex_set']
    global_surf_edges = layer_cache['global_surf_edges']

    # Compute LBS positions (skeleton-following) for current layer
    global_lbs = np.zeros((total_verts, 3))
    for name, mobj in layer_muscles.items():
        offset = global_offset[name]
        n = mobj.soft_body.num_vertices
        rest = mobj.soft_body.rest_positions
        if hasattr(mobj, 'skinning_weights') and mobj.skinning_weights is not None and len(mobj.skinning_bones) > 0:
            lbs = np.zeros((n, 3))
            for bone_idx, bone_name in enumerate(mobj.skinning_bones):
                body_node = skel.getBodyNode(bone_name)
                if body_node is None:
                    continue
                R = body_node.getWorldTransform().rotation()
                t = body_node.getWorldTransform().translation()
                if bone_name in mobj.soft_body_initial_transforms:
                    R0, t0 = mobj.soft_body_initial_transforms[bone_name]
                else:
                    continue
                w = mobj.skinning_weights[:, bone_idx:bone_idx+1]
                local = (R0.T @ (rest - t0).T).T
                deformed = (R @ local.T).T + t
                lbs += w * deformed
            global_lbs[offset:offset+n] = lbs
        else:
            global_lbs[offset:offset+n] = mobj.soft_body.positions

    # Frozen muscles: use their current settled positions
    for name, mobj in frozen_muscles.items():
        offset = global_offset[name]
        n = mobj.soft_body.num_vertices
        global_lbs[offset:offset+n] = mobj.soft_body.get_positions()

    # Warm-start: 70% LBS + 30% previous solution (skip if frame_independent)
    prev_solution = layer_cache.get('prev_solution', None)
    if not frame_independent and prev_solution is not None and prev_solution.shape[0] == total_verts:
        global_positions = 0.7 * global_lbs + 0.3 * prev_solution
        fixed_idx = np.where(global_fixed_mask)[0]
        global_positions[fixed_idx] = global_lbs[fixed_idx]
    else:
        global_positions = global_lbs.copy()

    # Fixed targets: bone attachments for current layer + all positions for frozen
    fixed_indices = np.where(global_fixed_mask)[0]
    global_fixed_targets = {}
    # Current layer: origin/insertion targets from skeleton
    for name, mobj in layer_muscles.items():
        offset = global_offset[name]
        if mobj.soft_body.fixed_targets is not None and len(mobj.soft_body.fixed_indices) > 0:
            for local_idx, target in zip(mobj.soft_body.fixed_indices, mobj.soft_body.fixed_targets):
                global_fixed_targets[offset + local_idx] = target
    # Frozen muscles: all vertices are fixed at their settled positions
    for name, mobj in frozen_muscles.items():
        offset = global_offset[name]
        settled_pos = mobj.soft_body.get_positions()
        for local_idx in range(mobj.soft_body.num_vertices):
            global_fixed_targets[offset + local_idx] = settled_pos[local_idx]
    fixed_targets_array = np.array([global_fixed_targets.get(i, global_rest_positions[i])
                                     for i in fixed_indices])

    # Iron Man offset: detach muscles outward from bones
    # In frame_independent mode: every frame. Otherwise: first frame only.
    is_first_frame = prev_solution is None
    do_offset = (frame_independent or is_first_frame) and len(obstacle_meshes) > 0
    if do_offset:
        n_offset = 0
        offset_amount = 0.005  # 5mm detachment
        for name, mobj in layer_muscles.items():
            off = global_offset[name]
            n = mobj.soft_body.num_vertices

            # Compute direction: muscle centroid relative to bone midpoint at rest
            if hasattr(mobj, 'skinning_bones') and len(mobj.skinning_bones) >= 2:
                bone_positions = []
                for bname in mobj.skinning_bones:
                    bn = skel.getBodyNode(bname)
                    if bn is not None:
                        bone_positions.append(bn.getWorldTransform().translation())
                if len(bone_positions) >= 2:
                    bone_mid = np.mean(bone_positions, axis=0)
                    muscle_centroid = global_positions[off:off+n].mean(axis=0)
                    direction = muscle_centroid - bone_mid
                    dist = np.linalg.norm(direction)
                    if dist > 1e-6:
                        direction /= dist
                        # Offset all FREE vertices of this muscle outward
                        for vi in range(off, off + n):
                            if not global_fixed_mask[vi]:
                                global_positions[vi] += direction * offset_amount
                                n_offset += 1
        if verbose and n_offset > 0:
            print(f"    First frame: offset {n_offset} vertices outward (Iron Man start)")

    # Get or create backend
    backend = layer_cache.get('backend', None)
    if backend is None or getattr(backend, '_backend_name', None) != backend_name:
        backend = get_backend(backend_name)
        backend._backend_name = backend_name
        layer_cache['backend'] = backend

    # Build system WITHOUT collision penalty (pure ARAP)
    need_build = not cache_valid or getattr(backend, '_splu', None) is None
    if need_build:
        backend.build_system(total_verts, neighbors, edge_weights, global_fixed_mask,
                             regularization=1e-6)

    # Step 1: Pure ARAP solve
    global_positions, iterations, max_disp = backend.solve(
        global_positions, global_rest_positions, neighbors, edge_weights, rest_edge_vectors,
        global_fixed_mask, fixed_targets_array,
        max_iterations=max_iterations, tolerance=tolerance,
        verbose=verbose,
    )

    # Inflate bone meshes for collision — keeps muscles 3mm from bone surface
    # Only used for detection; ARAP system is unmodified
    bone_inflate = 0.005  # 5mm inflation
    inflated_obstacles = []
    for om in obstacle_meshes:
        try:
            vn = om.vertex_normals
            inflated_verts = om.vertices + vn * bone_inflate
            inflated = trimesh.Trimesh(vertices=inflated_verts, faces=om.faces.copy(), process=True)
            inflated_obstacles.append(inflated)
        except Exception:
            inflated_obstacles.append(om)

    # Pre-filter obstacle meshes
    current_layer_verts = sum(layer_muscles[n].soft_body.num_vertices for n in muscle_names)
    layer_pos = global_positions[:current_layer_verts]
    layer_min = layer_pos.min(0)
    layer_max = layer_pos.max(0)
    nearby_obstacles = []
    for om in inflated_obstacles:
        ob_min, ob_max = om.bounds[0], om.bounds[1]
        if np.all(ob_min <= layer_max) and np.all(ob_max >= layer_min):
            nearby_obstacles.append(om)

    # Step 2: Detect + local ARAP re-solve
    n_rings = 3
    total_targets = 0
    for coll_round in range(2):
        collision_targets = {}
        _detect_collisions(global_positions, nearby_obstacles, collision_vertex_set,
                           global_surf_edges, global_fixed_mask, collision_margin,
                           collision_targets)
        if not collision_targets:
            break
        total_targets += len(collision_targets)

        shallow_targets = {}
        deep_targets = {}
        for vi, target in collision_targets.items():
            depth = np.linalg.norm(global_positions[vi] - target)
            if depth < 0.004:
                shallow_targets[vi] = target
            else:
                deep_targets[vi] = target

        for vi, target in shallow_targets.items():
            global_positions[vi] = target

        if deep_targets:
            _local_arap_resolve(global_positions, global_rest_positions, neighbors,
                                edge_weights, rest_edge_vectors, global_fixed_mask,
                                deep_targets, n_rings, collision_weight=5.0,
                                max_iterations=50, tolerance=1e-4)

    if verbose and total_targets > 0:
        print(f"    Collision: {total_targets} targets ({len(nearby_obstacles)} bones), "
              f"{coll_round+1} rounds")

    # Stash for warm-start
    layer_cache['prev_solution'] = global_positions.copy()

    # Write back to current layer muscles only (frozen are unchanged)
    for name, mobj in layer_muscles.items():
        offset = global_offset[name]
        n = mobj.soft_body.num_vertices
        mobj.soft_body.positions = global_positions[offset:offset+n].copy()
        mobj.tet_vertices = mobj.soft_body.get_positions().astype(np.float32)

    return iterations, max_disp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Layered ARAP bake with collision projection")
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--muscles", default=".last_loaded_muscles.json")
    parser.add_argument("--settle-iters", type=int, default=150)
    parser.add_argument("--constraint-threshold", type=float, default=0.015)
    parser.add_argument("--collision-margin", type=float, default=0.002)
    parser.add_argument("--frame-independent", action="store_true",
                        help="No warm-start between frames. Each frame starts from Iron Man offset. "
                             "Enables frame-level parallelization.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers (subprocesses). Implies --frame-independent.")
    parser.add_argument("--backend", choices=["auto","taichi","gpu","cpu"], default="auto")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--sides", default="L")
    parser.add_argument("--region-tag", default="layered")
    parser.add_argument("--tet-dir", default="tet",
                        help="Directory for tet mesh files (default: tet, use tet_subdiv for subdivided)")
    args = parser.parse_args()

    # Multi-worker: split frames into chunks, launch subprocesses
    if args.num_workers > 1:
        import subprocess
        args.frame_independent = True

        # Determine frame range
        if args.end_frame is None:
            # Need to load BVH to get frame count
            skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
            skel = buildFromInfo(skel_info, root_name)
            t_frame = _detect_bvh_tframe(args.bvh)
            motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
            end_frame = motion_bvh.mocap_refs.shape[0] - 1
        else:
            end_frame = args.end_frame

        total_frames = end_frame - args.start_frame + 1
        chunk_size = (total_frames + args.num_workers - 1) // args.num_workers

        procs = []
        for wi in range(args.num_workers):
            sf = args.start_frame + wi * chunk_size
            ef = min(sf + chunk_size - 1, end_frame)
            if sf > end_frame:
                break
            cmd = [sys.executable, __file__,
                   "--bvh", args.bvh,
                   "--muscles", args.muscles,
                   "--settle-iters", str(args.settle_iters),
                   "--constraint-threshold", str(args.constraint_threshold),
                   "--collision-margin", str(args.collision_margin),
                   "--frame-independent",
                   "--backend", args.backend,
                   "--start-frame", str(sf),
                   "--end-frame", str(ef),
                   "--sides", args.sides,
                   "--region-tag", args.region_tag,
                   ]
            # On multi-GPU: assign one GPU per worker
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(wi % max(1, int(os.environ.get("NUM_GPUS", "1"))))
            print(f"Worker {wi}: frames {sf}-{ef} (GPU {env['CUDA_VISIBLE_DEVICES']})")
            procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env))

        # Wait for all workers to complete (they run in parallel)
        t0 = time.time()
        outputs = [None] * len(procs)
        for wi, proc in enumerate(procs):
            out, _ = proc.communicate()
            outputs[wi] = out.decode()

        wall_time = time.time() - t0
        for wi, out in enumerate(outputs):
            for line in out.strip().split('\n'):
                if 'Frame' in line or 'Done' in line:
                    print(f"  [W{wi}] {line.strip()}")

        print(f"\nAll {args.num_workers} workers done. Wall time: {wall_time:.1f}s "
              f"({wall_time/total_frames:.1f}s/frame effective)")
        return

    # Resolve backend
    backend_name = args.backend
    if backend_name == "auto":
        if check_taichi_available(): backend_name = "taichi"
        elif check_gpu_available(): backend_name = "gpu"
        else: backend_name = "cpu"
    print(f"ARAP backend: {backend_name.upper()}")

    # ── Load everything ──────────────────────────────────────────────────
    print("[1] Loading skeleton...")
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)

    print("[2] Loading skeleton meshes...")
    skeleton_meshes = {}
    skel_dir = os.path.join(ZYGOTE_DIR, "Skeleton")
    for fname in sorted(os.listdir(skel_dir)):
        if not fname.endswith(".obj"): continue
        name = fname.split(".")[0]
        path = os.path.join(skel_dir, fname)
        skeleton_meshes[name] = MeshLoader()
        skeleton_meshes[name].load(path)
        skeleton_meshes[name].color = np.array([0.9, 0.9, 0.9])
        skel_tri = trimesh.load_mesh(path)
        skel_tri.vertices *= MESH_SCALE
        skeleton_meshes[name].trimesh = skel_tri

    print("[3] Loading muscle meshes...")
    with open(args.muscles, "r") as f:
        muscle_list = json.load(f)
    all_muscle_meshes = {}
    for entry in muscle_list:
        name, path = entry["name"], entry["path"]
        if not os.path.exists(path): continue
        all_muscle_meshes[name] = MeshLoader()
        all_muscle_meshes[name].load(path)
        all_muscle_meshes[name].color = np.array([0.8, 0.2, 0.2])
        mtri = trimesh.load_mesh(path); mtri.vertices *= MESH_SCALE
        all_muscle_meshes[name].trimesh = mtri
    all_muscle_meshes = dict(sorted(all_muscle_meshes.items()))

    print("[4] Loading tet meshes...")
    orig_vert_counts = {}
    bary_mappings = {}
    orig_anchor_sets = {}
    anchor_bone_maps = {}  # Per-anchor bone assignment from original mesh
    for name, mobj in all_muscle_meshes.items():
        tet_path = os.path.join(args.tet_dir, f"{name}_tet.npz")
        if args.tet_dir != "tet" and os.path.exists(tet_path):
            import pickle as _pkl
            with open(tet_path, 'rb') as _f:
                _tet_data = _pkl.load(_f)
            if 'orig_to_fine_mapping' in _tet_data:
                bary_mappings[name] = _tet_data['orig_to_fine_mapping']
                orig_vert_counts[name] = _tet_data.get('orig_n_verts', 0)
            elif 'cap_vertex_types' in _tet_data:
                # Original mesh: save all verts + also contour-mapped version
                if 'contour_mapping' in _tet_data:
                    bary_mappings[name] = _tet_data['contour_mapping']
                    orig_vert_counts[name] = _tet_data.get('contour_n_verts', 0)
            elif os.path.exists(os.path.join("tet", f"{name}_tet.npz")):
                with open(os.path.join("tet", f"{name}_tet.npz"), 'rb') as _f:
                    _orig = _pkl.load(_f)
                orig_vert_counts[name] = len(_orig['vertices'])
            # Load CONTOUR anchors (for position mapping back to contour mesh)
            # Try contour backup first, then tet/
            contour_tet_path = os.path.join("tet", f"{name}_tet.npz.contour_backup")
            if not os.path.exists(contour_tet_path):
                contour_tet_path = os.path.join("tet", f"{name}_tet.npz")
            if os.path.exists(contour_tet_path):
                with open(contour_tet_path, 'rb') as _f:
                    _orig_data = _pkl.load(_f)
                # Only use if it's actually the contour mesh (not original)
                if 'cap_vertex_types' not in _orig_data:
                    orig_anchor_sets[name] = set(int(v) for v in _orig_data.get('anchor_vertices', []))
            if 'anchor_bone_map' in _tet_data:
                anchor_bone_maps[name] = _tet_data['anchor_bone_map']
        mobj.load_tetrahedron_mesh(name, filepath=tet_path)

    print("[5] Initializing soft bodies...")
    skel.setPositions(np.zeros(skel.getNumDofs()))
    for name, mobj in all_muscle_meshes.items():
        if mobj.tet_vertices is None: continue

        # Check if this is an original mesh (has cap_vertex_types)
        tet_path = os.path.join(args.tet_dir, f"{name}_tet.npz")
        _has_orig_format = False
        if os.path.exists(tet_path):
            import pickle as _pkl
            with open(tet_path, 'rb') as _f:
                _td = _pkl.load(_f)
            _has_orig_format = 'cap_vertex_types' in _td

        if False and _has_orig_format:  # Disabled: init_soft_body handles original mesh
            # Original mesh: manual soft body setup (init_soft_body doesn't handle it)
            from viewer.muscle_mesh import SoftBodySimulation
            cap_types = _td['cap_vertex_types']  # {vi: 'origin'/'insertion'}
            attach_names = _td.get('attach_skeleton_names', [[]])

            # Fixed vertices = all cap vertices (proper boundary from open mesh)
            fixed_indices = sorted(cap_types.keys())
            fixed_mask = np.zeros(len(mobj.tet_vertices), dtype=bool)
            for vi in fixed_indices:
                if vi < len(fixed_mask):
                    fixed_mask[vi] = True

            # Create soft body
            mobj.soft_body = SoftBodySimulation(
                vertices=mobj.tet_vertices,
                tetrahedra=mobj.tet_tetrahedra,
                fixed_vertices=fixed_indices,
            )

            # Skeleton bindings: origin → first bone, insertion → second bone
            bones = attach_names[0] if attach_names else []
            origin_bone = bones[0] if len(bones) > 0 else None
            insertion_bone = bones[1] if len(bones) > 1 else None

            mobj.soft_body_local_anchors = {}
            mobj.soft_body_initial_transforms = {}
            mobj.skinning_bones = []
            mobj.skinning_weights = None

            if origin_bone and insertion_bone:
                mobj.skinning_bones = [origin_bone, insertion_bone]
                # Compute rest transforms
                for bname in mobj.skinning_bones:
                    bn = skel.getBodyNode(bname)
                    if bn:
                        R = bn.getWorldTransform().rotation()
                        t = bn.getWorldTransform().translation()
                        mobj.soft_body_initial_transforms[bname] = (R.copy(), t.copy())

                # Per-anchor local position and bone assignment
                for vi, cap_type in cap_types.items():
                    vi = int(vi)
                    if vi >= len(mobj.tet_vertices):
                        continue
                    bname = origin_bone if cap_type == 'origin' else insertion_bone
                    bn = skel.getBodyNode(bname)
                    if bn:
                        R = bn.getWorldTransform().rotation()
                        t = bn.getWorldTransform().translation()
                        local_pos = R.T @ (mobj.tet_vertices[vi] - t)
                        mobj.soft_body_local_anchors[vi] = (bname, local_pos)

                # Skeleton bindings for _update_tet_positions_from_skeleton
                n_verts = len(mobj.tet_vertices)
                mobj.tet_skeleton_bindings = []
                mobj.tet_initial_bone_transforms = dict(mobj.soft_body_initial_transforms)

                _, t_o = mobj.soft_body_initial_transforms.get(origin_bone, (None, None))
                _, t_i = mobj.soft_body_initial_transforms.get(insertion_bone, (None, None))
                if t_o is not None and t_i is not None:
                    axis = t_i - t_o
                    axis_len = np.linalg.norm(axis)
                    axis_dir = axis / axis_len if axis_len > 1e-6 else np.array([0, 1, 0])
                    weights = np.zeros((n_verts, 2), dtype=np.float32)
                    for vi in range(n_verts):
                        proj = np.dot(mobj.tet_vertices[vi] - t_o, axis_dir) / max(axis_len, 1e-6)
                        proj = np.clip(proj, 0, 1)
                        weights[vi, 0] = 1.0 - proj
                        weights[vi, 1] = proj
                        mobj.tet_skeleton_bindings.append(
                            (origin_bone, insertion_bone, proj, mobj.tet_vertices[vi].copy()))
                    mobj.skinning_weights = weights

            print(f"    {name}: orig mesh {len(mobj.tet_vertices)} verts, "
                  f"{len(fixed_indices)} fixed, {len(mobj.skinning_bones)} bones")
        else:
            mobj.init_soft_body(skeleton_meshes=skeleton_meshes, skeleton=skel, mesh_info=mesh_info)
        # Override bone assignments from original mesh if available
        abm = anchor_bone_maps.get(name)
        if abm and hasattr(mobj, 'soft_body_local_anchors'):
            for fine_vi, bone_name in abm.items():
                fine_vi = int(fine_vi)
                if fine_vi in mobj.soft_body_local_anchors:
                    old_bone, local_pos = mobj.soft_body_local_anchors[fine_vi]
                    if old_bone != bone_name:
                        # Recompute local anchor position for the correct bone
                        body_node = skel.getBodyNode(bone_name)
                        if body_node is not None:
                            R = body_node.getWorldTransform().rotation()
                            t = body_node.getWorldTransform().translation()
                            world_pos = mobj.tet_vertices[fine_vi]
                            new_local = R.T @ (world_pos - t)
                            mobj.soft_body_local_anchors[fine_vi] = (bone_name, new_local)

    # Filter to active muscles on requested side
    side = args.sides[0]
    active_all = {n: m for n, m in all_muscle_meshes.items()
                  if m.soft_body is not None and n.startswith(f"{side}_")}
    print(f"    {len(active_all)} active {side}-side muscles")

    # ── Precompute surface data ───────────────────────────────────────────
    # Pre-compute contour mesh anchor soft bodies for position conversion
    # (needed when ARAP runs on original mesh but viewer uses contour mesh)
    _contour_anchor_cache_init = {}  # {name: {anchor_vi: (bone_name, local_pos)}}
    for name in list(orig_anchor_sets.keys()):
        if name not in bary_mappings:
            continue
        # Use contour backup if available (tet/ may have original mesh)
        contour_tet_path = os.path.join("tet", f"{name}_tet.npz.contour_backup")
        if not os.path.exists(contour_tet_path):
            contour_tet_path = os.path.join("tet", f"{name}_tet.npz")
        if not os.path.exists(contour_tet_path):
            continue
        import json as _json_init
        _entry = None
        with open('.last_loaded_muscles.json') as _mf:
            for _e in _json_init.load(_mf):
                if _e['name'] == name:
                    _entry = _e; break
        if not _entry or not os.path.exists(_entry['path']):
            continue
        _tmp = MeshLoader()
        _tmp.load(_entry['path'])
        _tmp.load_tetrahedron_mesh(name, filepath=contour_tet_path)
        _tmp.init_soft_body(skeleton_meshes=skeleton_meshes, skeleton=skel, mesh_info=mesh_info)
        if _tmp.soft_body and hasattr(_tmp, 'soft_body_local_anchors'):
            _contour_anchor_cache_init[name] = dict(_tmp.soft_body_local_anchors)

    print("[7] Precomputing surface data...")
    precompute_surface_data(active_all)
    bone_rest_transforms = cache_bone_rest_transforms(skel)
    total_sv = sum(len(getattr(m, '_surf_verts', [])) for m in active_all.values())
    print(f"    {total_sv} surface vertices for collision")


    # ── Classify into layers ──────────────────────────────────────────────
    print("[8] Classifying into layers...")
    layer_muscles = [[], [], []]
    for name in active_all:
        short = name.replace(f"{side}_", "")
        assigned = False
        for li, lm in LAYERS.items():
            if short in lm:
                layer_muscles[li].append(name)
                assigned = True
                break
        if not assigned:
            layer_muscles[2].append(name)

    for li in range(3):
        print(f"    Layer {li}: {len(layer_muscles[li])} — {layer_muscles[li]}")

    # ── Find inter-muscle constraints ─────────────────────────────────────
    print("[9] Finding inter-muscle constraints...")
    global_ctx = SimpleNamespace(
        env=SimpleNamespace(skel=skel, mesh_info=mesh_info),
        zygote_muscle_meshes=active_all,
        zygote_skeleton_meshes=skeleton_meshes,
        inter_muscle_constraints=[],
        inter_muscle_constraint_threshold=args.constraint_threshold,
        coupled_as_unified_volume=True,
        use_gpu_arap=False,
        use_taichi_arap=False,
        use_muscle_aware_arap=True,
        use_fem_sim=False,
        use_vbd_sim=False,
        use_pn_sim=False,
        _unified_arap_backend=None,
        _unified_sim_cache=None,
    )
    n_global = find_inter_muscle_constraints(global_ctx)
    all_constraints = global_ctx.inter_muscle_constraints
    print(f"    {n_global} global constraints")

    # Filter constraints per-layer
    cumulative_muscles = set()
    layer_constraints = [[], [], []]
    for li in range(3):
        cumulative_muscles.update(layer_muscles[li])
        layer_set = set(layer_muscles[li])
        for c in all_constraints:
            name1, v1, f1, name2, v2, f2, dist = c
            if name1 in cumulative_muscles and name2 in cumulative_muscles:
                if name1 in layer_set or name2 in layer_set:
                    layer_constraints[li].append(c)
        print(f"    Layer {li}: {len(layer_constraints[li])} constraints")

    # ── Load BVH ──────────────────────────────────────────────────────────
    print("[10] Loading BVH...")
    t_frame = _detect_bvh_tframe(args.bvh)
    motion_bvh = MyBVH(args.bvh, bvh_info, skel, T_frame=t_frame)
    n_frames = motion_bvh.mocap_refs.shape[0]
    end_frame = args.end_frame if args.end_frame is not None else n_frames - 1
    end_frame = min(end_frame, n_frames - 1)
    total_frames = end_frame - args.start_frame + 1

    # ── Output ────────────────────────────────────────────────────────────
    bvh_stem = os.path.splitext(os.path.basename(args.bvh))[0]
    cache_dir = os.path.join("data", "motion_cache", bvh_stem, args.region_tag)
    os.makedirs(cache_dir, exist_ok=True)

    import glob as glob_mod
    for old in glob_mod.glob(os.path.join(cache_dir, "*_chunk_*.npz")):
        os.remove(old)

    bake_data = {n: {} for n in active_all}
    contour_bake_data = {n: {} for n in active_all}  # Contour-mapped positions
    contour_cache_dir = os.path.join("data", "motion_cache", bvh_stem, "layered_contour")
    if bary_mappings:
        os.makedirs(contour_cache_dir, exist_ok=True)
        for old in glob_mod.glob(os.path.join(contour_cache_dir, "*_chunk_*.npz")):
            os.remove(old)
    flush_count = 0
    bake_start = time.time()

    # Per-layer caches
    layer_caches = [{}, {}, {}]

    # ── Frame loop ────────────────────────────────────────────────────────
    print(f"\n[11] Baking frames {args.start_frame}-{end_frame}...")

    for frame in range(args.start_frame, end_frame + 1):
        frame_start = time.time()
        skel.setPositions(motion_bvh.mocap_refs[frame])

        # Build bone collision meshes at current pose
        # Use hi-res meshes if available (10x finer → better inverse collision detection)
        if _warp_ok[0]:
            _warp_clear_cache()
        bone_tms = build_bone_collision_meshes(skeleton_meshes, skel, bone_rest_transforms)
        obstacle_meshes = list(bone_tms)  # Bones always in obstacle set
        settled_muscles = {}  # Accumulate settled earlier-layer muscles

        # Process each layer in depth order
        for li in range(3):
            if not layer_muscles[li]:
                continue

            layer_active = {n: active_all[n] for n in layer_muscles[li]}

            # Disable waypoint updates during baking
            for mname, mobj in layer_active.items():
                mobj.waypoints_from_tet_sim = False
                mobj._baking_mode = True

            # Run ARAP with penalty-in-diagonal collision
            # frozen_muscles = all muscles from earlier layers (already settled)
            iters, max_disp = run_layer_sim_with_collision(
                layer_active, settled_muscles, skeleton_meshes, skel,
                obstacle_meshes, layer_constraints[li],
                layer_caches[li], backend_name,
                max_iterations=args.settle_iters, tolerance=1e-4,
                collision_margin=args.collision_margin,
                frame_independent=args.frame_independent,
                verbose=(frame == args.start_frame and li == 0))

            # Restore flags and capture positions
            for mname, mobj in layer_active.items():
                mobj.waypoints_from_tet_sim = True
                mobj._baking_mode = False
                fine_pos = mobj.soft_body.get_positions().astype(np.float32)
                # Save original mesh positions (all verts)
                bake_data[mname][frame] = fine_pos.copy()

                # Also save contour-mapped positions for contour display
                mapping = bary_mappings.get(mname)
                n_orig = orig_vert_counts.get(mname)
                anchor_set = orig_anchor_sets.get(mname, set())
                if mapping is not None and n_orig:
                    contour_pos = np.zeros((n_orig, 3), dtype=np.float32)
                    fine_tets = mobj.tet_tetrahedra
                    anchor_targets = {}
                    if anchor_set and mname in _contour_anchor_cache_init:
                        for oa_idx, (bname, local_pos) in _contour_anchor_cache_init[mname].items():
                            if int(oa_idx) not in anchor_set:
                                continue
                            bn = skel.getBodyNode(bname)
                            if bn is not None:
                                R = bn.getWorldTransform().rotation()
                                t = bn.getWorldTransform().translation()
                                anchor_targets[int(oa_idx)] = (R @ local_pos) + t
                    for ci, (tet_idx, bary) in enumerate(mapping):
                        if ci >= n_orig:
                            break
                        if ci in anchor_targets:
                            contour_pos[ci] = anchor_targets[ci]
                            continue
                        if tet_idx < 0 or tet_idx >= len(fine_tets):
                            contour_pos[ci] = fine_pos[ci] if ci < len(fine_pos) else 0
                            continue
                        tv = fine_tets[tet_idx]
                        contour_pos[ci] = (bary[0] * fine_pos[tv[0]] +
                                           bary[1] * fine_pos[tv[1]] +
                                           bary[2] * fine_pos[tv[2]] +
                                           bary[3] * fine_pos[tv[3]])
                    contour_bake_data[mname][frame] = contour_pos

            # Settled muscles become obstacles AND frozen constraints for next layer
            # Inflate obstacle slightly so outer muscles can't reach bone through gaps
            obstacle_margin = 0.002  # 2mm buffer
            for mname, mobj in layer_active.items():
                settled_muscles[mname] = mobj
                if hasattr(mobj, '_surf_faces'):
                    verts = mobj.soft_body.get_positions().copy()
                    tm = trimesh.Trimesh(vertices=verts,
                                         faces=mobj._surf_faces.copy(), process=True)
                    # Inflate: push each vertex outward along its vertex normal
                    try:
                        vnormals = tm.vertex_normals
                        verts_inflated = verts + vnormals * obstacle_margin
                        tm_inflated = trimesh.Trimesh(vertices=verts_inflated,
                                                       faces=mobj._surf_faces.copy(), process=True)
                        obstacle_meshes.append(tm_inflated)
                    except Exception:
                        obstacle_meshes.append(tm)

        frame_dt = time.time() - frame_start
        frames_done = frame - args.start_frame + 1
        elapsed = time.time() - bake_start
        avg = elapsed / frames_done
        remaining = avg * (total_frames - frames_done)
        print(f"  Frame {frame}: {frame_dt:.2f}s  avg {avg:.2f}s/frame  ETA {remaining:.0f}s",
              flush=True)

        # Flush
        n_acc = sum(len(fd) for fd in bake_data.values())
        if n_acc >= FLUSH_INTERVAL * len(active_all):
            for mname, fd in bake_data.items():
                if not fd: continue
                sf = sorted(fd.keys())
                fp = os.path.join(cache_dir, f"{mname}_chunk_{flush_count:04d}.npz")
                np.savez(fp, frames=np.array(sf, dtype=np.int32),
                         positions=np.array([fd[f] for f in sf], dtype=np.float32))
                fd.clear()
            # Also flush contour-mapped data
            for mname, fd in contour_bake_data.items():
                if not fd: continue
                sf = sorted(fd.keys())
                fp = os.path.join(contour_cache_dir, f"{mname}_chunk_{flush_count:04d}.npz")
                np.savez(fp, frames=np.array(sf, dtype=np.int32),
                         positions=np.array([fd[f] for f in sf], dtype=np.float32))
                fd.clear()
            flush_count += 1; gc.collect()

    # Final flush
    for mname, fd in bake_data.items():
        if not fd: continue
        sf = sorted(fd.keys())
        fp = os.path.join(cache_dir, f"{mname}_chunk_{flush_count:04d}.npz")
        np.savez(fp, frames=np.array(sf, dtype=np.int32),
                 positions=np.array([fd[f] for f in sf], dtype=np.float32))
        fd.clear()
    for mname, fd in contour_bake_data.items():
        if not fd: continue
        sf = sorted(fd.keys())
        fp = os.path.join(contour_cache_dir, f"{mname}_chunk_{flush_count:04d}.npz")
        np.savez(fp, frames=np.array(sf, dtype=np.int32),
                 positions=np.array([fd[f] for f in sf], dtype=np.float32))
        fd.clear()

    elapsed = time.time() - bake_start
    print(f"\nDone. {total_frames} frames in {elapsed:.1f}s ({elapsed/max(total_frames,1):.2f}s/frame)")
    print(f"Output: {cache_dir}")


if __name__ == "__main__":
    main()
