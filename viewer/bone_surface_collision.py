"""Post-ARAP bone collision resolution for muscle tet meshes.

Adapted from `tools/bake_contour_sim.py:resolve_surface_collisions`. Runs once
per frame after the elastic solve has converged so the push-out is not undone
by ARAP elasticity in a subsequent iteration.

Two phases:
  Phase 1 — vertex-bone: bone_mesh.contains() picks verts inside any bone,
            project to closest_point + face_normal * margin.
  Phase 2 — edge-bone:   ray-cast along surface edges; if the ray hits a bone
            triangle inside the segment, push the bone-side endpoint(s) out.

`positions` is mutated in place. Verts in `fixed_indices` are never moved.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def resolve_bone_collisions(positions, fixed_indices, surface_verts, surface_edges,
                            bone_trimeshes, margin=0.002, max_iters=3):
    """Resolve vertex- and edge-level penetrations of `bone_trimeshes`.

    Returns (n_vert_pushed, n_edge_resolved) summed across iterations.
    """
    if not bone_trimeshes or len(positions) == 0:
        return 0, 0

    fixed_set = set(int(i) for i in (fixed_indices if fixed_indices is not None else []))
    surface_verts = np.asarray(surface_verts, dtype=np.int64)
    if surface_edges is None or len(surface_edges) == 0:
        surface_edges = np.empty((0, 2), dtype=np.int64)
    else:
        surface_edges = np.asarray(surface_edges, dtype=np.int64)

    all_bone_verts = np.vstack([bm.vertices for bm in bone_trimeshes])
    bone_kdtree = cKDTree(all_bone_verts)

    total_vert = 0
    total_edge = 0

    for _ in range(max_iters):
        n_vert = 0
        n_edge = 0

        # Phase 1: vertex-bone -------------------------------------------------
        if len(surface_verts) > 0:
            sv_pos = positions[surface_verts]
            dists, _ = bone_kdtree.query(sv_pos)
            near_mask = dists < 0.03  # 3 cm
            if np.any(near_mask):
                near_sv = surface_verts[near_mask]
                near_pos = positions[near_sv]

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
                        normals = bone_mesh.face_normals[face_ids]
                        for k in range(len(inside_sv)):
                            vi = int(inside_sv[k])
                            if vi in fixed_set:
                                continue
                            positions[vi] = closest[k] + normals[k] * margin
                            n_vert += 1
                    except Exception:
                        continue

        # Phase 2: edge-bone tunneling ----------------------------------------
        if len(surface_edges) > 0:
            edge_v0 = positions[surface_edges[:, 0]]
            edge_v1 = positions[surface_edges[:, 1]]
            d0, _ = bone_kdtree.query(edge_v0)
            d1, _ = bone_kdtree.query(edge_v1)
            edge_near = (d0 < 0.02) | (d1 < 0.02)
            if np.any(edge_near):
                cand = surface_edges[edge_near]
                origins = positions[cand[:, 0]]
                endpoints = positions[cand[:, 1]]
                dirs = endpoints - origins
                lengths = np.linalg.norm(dirs, axis=1)
                valid = lengths > 1e-10
                if np.any(valid):
                    dirs_norm = dirs.copy()
                    dirs_norm[valid] /= lengths[valid, None]
                    valid_edges = cand[valid]
                    valid_origins = origins[valid]
                    valid_dirs = dirs_norm[valid]
                    valid_lengths = lengths[valid]

                    for bone_mesh in bone_trimeshes:
                        try:
                            locations, ray_idx, tri_idx = bone_mesh.ray.intersects_location(
                                valid_origins, valid_dirs, multiple_hits=True)
                        except Exception:
                            continue
                        if len(locations) == 0:
                            continue
                        for loc, ri, ti in zip(locations, ray_idx, tri_idx):
                            t_param = float(np.dot(loc - valid_origins[ri], valid_dirs[ri]))
                            edge_len = float(valid_lengths[ri])
                            if not (0.0 < t_param < edge_len):
                                continue
                            v0i = int(valid_edges[ri, 0])
                            v1i = int(valid_edges[ri, 1])
                            if v0i in fixed_set and v1i in fixed_set:
                                continue
                            face_normal = bone_mesh.face_normals[ti]
                            face_center = bone_mesh.triangles[ti].mean(axis=0)
                            d0s = float(np.dot(positions[v0i] - face_center, face_normal))
                            d1s = float(np.dot(positions[v1i] - face_center, face_normal))
                            for vi, di in ((v0i, d0s), (v1i, d1s)):
                                if di >= 0 or vi in fixed_set:
                                    continue
                                cp, _, fid = trimesh.proximity.closest_point(
                                    bone_mesh, positions[vi:vi + 1])
                                fn = bone_mesh.face_normals[fid[0]]
                                depth = abs(di)
                                push_margin = min(depth * 1.5 + margin, 0.020)
                                positions[vi] = cp[0] + fn * push_margin
                                n_edge += 1

        total_vert += n_vert
        total_edge += n_edge
        if n_vert == 0 and n_edge == 0:
            break

    return total_vert, total_edge


def compute_surface_topology(tetrahedra=None, tet_faces=None):
    """Surface = triangular faces shared by exactly one tet.

    Pass either pre-computed boundary `tet_faces` (F, 3) — used directly —
    or `tetrahedra` (T, 4) — boundary faces are derived. Returns
    (surface_vidx, surface_edges).
    """
    if tet_faces is not None and len(tet_faces) > 0:
        tf = np.asarray(tet_faces, dtype=np.int64)
    elif tetrahedra is not None and len(tetrahedra) > 0:
        tets = np.asarray(tetrahedra, dtype=np.int64)
        # Each tet has 4 faces (vertex opposite each tet corner).
        faces = np.concatenate([
            tets[:, [1, 2, 3]], tets[:, [0, 3, 2]],
            tets[:, [0, 1, 3]], tets[:, [0, 2, 1]],
        ], axis=0)
        sorted_faces = np.sort(faces, axis=1)
        # Boundary faces appear once across the whole tet mesh.
        _, inverse, counts = np.unique(sorted_faces, axis=0,
                                       return_inverse=True, return_counts=True)
        boundary_mask = counts[inverse] == 1
        tf = faces[boundary_mask]
    else:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.int64)

    surface_vidx = np.unique(tf.reshape(-1))
    edges = np.concatenate([
        tf[:, [0, 1]], tf[:, [1, 2]], tf[:, [2, 0]],
    ], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return surface_vidx, edges
