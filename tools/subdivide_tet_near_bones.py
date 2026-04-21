#!/usr/bin/env python3
"""Subdivide long surface edges near bones in tet meshes.

One-time preprocessing: splits edges > max_edge_len within bone_dist
of bone surfaces. Adds midpoint vertices, splits incident tets/faces.
Properly updates all metadata (cap_face_indices, vertex_contour_level, etc.).

Usage:
    python tools/subdivide_tet_near_bones.py --input-dir tet --output-dir tet_subdiv
"""
import argparse
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229/"
MESH_SCALE = 0.01


def extract_surface_edges(tets):
    """Extract surface edges from tet mesh."""
    face_count = Counter()
    face_orient = {}
    for t in tets:
        v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        faces = [(v0, v2, v1), (v0, v1, v3), (v1, v2, v3), (v0, v3, v2)]
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] += 1
            if key not in face_orient:
                face_orient[key] = f
    surf_faces = [face_orient[k] for k, c in face_count.items() if c == 1]
    edges = set()
    for f in surf_faces:
        for i in range(3):
            e = (min(int(f[i]), int(f[(i+1) % 3])),
                 max(int(f[i]), int(f[(i+1) % 3])))
            edges.add(e)
    return np.array(sorted(edges), dtype=np.int64)


def subdivide_tet_file(input_path, output_path, bone_trimeshes,
                        max_edge_len=0.010, bone_dist=0.015):
    """Subdivide long surface edges near bones in a tet file.

    Returns number of edges subdivided.
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    verts = data['vertices']  # Already in meters
    tets = data['tetrahedra']
    if verts is None or tets is None:
        # Copy as-is
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        return 0

    # Find surface edges
    se = extract_surface_edges(tets)
    if len(se) == 0:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        return 0

    # Find long edges near bones
    v0_pos = verts[se[:, 0]]
    v1_pos = verts[se[:, 1]]
    edge_lens = np.linalg.norm(v1_pos - v0_pos, axis=1)
    long_mask = edge_lens > max_edge_len

    if not np.any(long_mask) or not bone_trimeshes:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        return 0

    all_bone_verts = np.vstack([bm.vertices for bm in bone_trimeshes])
    bone_kdtree = cKDTree(all_bone_verts)

    long_edges = se[long_mask]
    long_mids = 0.5 * (verts[long_edges[:, 0]] + verts[long_edges[:, 1]])
    d_mid, _ = bone_kdtree.query(long_mids)
    near_bone = d_mid < bone_dist

    edges_to_split = long_edges[near_bone]
    if len(edges_to_split) == 0:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        return 0

    edge_set = set()
    for e in edges_to_split:
        edge_set.add((min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1]))))

    # --- Subdivide ---
    new_verts = list(verts)
    midpoint_cache = {}  # (v0, v1) → new vertex index

    def get_midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid_pos = 0.5 * (verts[a] + verts[b])
        new_idx = len(new_verts)
        new_verts.append(mid_pos)
        midpoint_cache[key] = new_idx
        return new_idx

    # Split tets
    new_tets = []
    for t in tets:
        t = [int(x) for x in t]
        tet_edges = [(t[i], t[j]) for i in range(4) for j in range(i+1, 4)]
        splits = []
        for a, b in tet_edges:
            key = (min(a, b), max(a, b))
            if key in edge_set:
                splits.append((a, b))

        if not splits:
            new_tets.append(t)
        elif len(splits) == 1:
            a, b = splits[0]
            m = get_midpoint(a, b)
            others = [v for v in t if v != a and v != b]
            c, d = others
            new_tets.append([a, m, c, d])
            new_tets.append([m, b, c, d])
        else:
            # Multi-edge split: keep as-is (complex subdivision)
            new_tets.append(t)

    # Split render_faces
    old_faces = data.get('render_faces', data.get('faces'))
    new_faces = []
    old_to_new_face = {}  # old_face_idx → [new_face_idx, ...]
    for fi, f in enumerate(old_faces):
        f = [int(x) for x in f]
        face_edges = [(f[i], f[(i+1) % 3]) for i in range(3)]
        face_splits = []
        for a, b in face_edges:
            key = (min(a, b), max(a, b))
            if key in midpoint_cache:
                face_splits.append((a, b, midpoint_cache[key]))

        if not face_splits:
            old_to_new_face[fi] = [len(new_faces)]
            new_faces.append(f)
        elif len(face_splits) == 1:
            a, b, m = face_splits[0]
            c = [v for v in f if v != a and v != b][0]
            old_to_new_face[fi] = [len(new_faces), len(new_faces) + 1]
            new_faces.append([a, m, c])
            new_faces.append([m, b, c])
        else:
            old_to_new_face[fi] = [len(new_faces)]
            new_faces.append(f)

    # --- Update metadata ---
    n_orig = len(verts)
    n_new = len(new_verts)
    n_added = n_new - n_orig

    # vertices
    data['vertices'] = np.array(new_verts, dtype=np.float32)

    # tetrahedra
    data['tetrahedra'] = np.array(new_tets, dtype=np.int32)

    # faces / render_faces
    new_faces_arr = np.array(new_faces, dtype=np.int32)
    data['faces'] = new_faces_arr
    data['render_faces'] = new_faces_arr

    # sim_faces: regenerate from tets (boundary faces)
    data['sim_faces'] = None  # Will be regenerated by init_soft_body

    # cap_face_indices: re-identify using anchor_vertices
    anchor_set = set(int(v) for v in data.get('anchor_vertices', []))
    new_cap_faces = []
    for fi, face in enumerate(new_faces):
        if all(int(v) in anchor_set for v in face):
            new_cap_faces.append(fi)
    data['cap_face_indices'] = np.array(new_cap_faces, dtype=np.int32)

    # vertex_contour_level: extend with -1 for new midpoints
    vcl = data.get('vertex_contour_level')
    if vcl is not None and isinstance(vcl, np.ndarray):
        ext = np.full(n_added, -1, dtype=vcl.dtype)
        data['vertex_contour_level'] = np.concatenate([vcl, ext])

    # contour_to_tet_mapping: extend with None for new midpoints
    ctm = data.get('contour_to_tet_mapping')
    if ctm is not None and isinstance(ctm, list):
        data['contour_to_tet_mapping'] = ctm + [None] * n_added

    # mvc_weights: extend with interpolated weights
    mvc = data.get('mvc_weights')
    if mvc is not None and isinstance(mvc, np.ndarray) and len(mvc) == n_orig:
        new_mvc = np.zeros((n_added, mvc.shape[1]), dtype=mvc.dtype)
        for (a, b), new_idx in midpoint_cache.items():
            local_idx = new_idx - n_orig
            new_mvc[local_idx] = 0.5 * (mvc[a] + mvc[b])
        data['mvc_weights'] = np.concatenate([mvc, new_mvc])

    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    return len(midpoint_cache)


def main():
    parser = argparse.ArgumentParser(description="Subdivide tet mesh edges near bones")
    parser.add_argument("--input-dir", default="tet")
    parser.add_argument("--output-dir", default="tet_subdiv")
    parser.add_argument("--max-edge-len", type=float, default=0.010,
                        help="Maximum edge length in meters (default: 10mm)")
    parser.add_argument("--bone-dist", type=float, default=0.015,
                        help="Distance threshold from bone surface (default: 15mm)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build bone trimeshes at rest pose
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    skel.setPositions(np.zeros(skel.getNumDofs()))

    bone_trimeshes = []
    skel_dir = os.path.join(ZYGOTE_DIR, "Skeleton")
    for fname in sorted(os.listdir(skel_dir)):
        if not fname.endswith(".obj"):
            continue
        skel_tri = trimesh.load_mesh(os.path.join(skel_dir, fname))
        skel_tri.vertices *= MESH_SCALE
        bone_trimeshes.append(skel_tri)
    print(f"Loaded {len(bone_trimeshes)} bone meshes")

    # Process each tet file
    total_subdiv = 0
    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith("_tet.npz"):
            continue
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)
        t0 = time.time()
        n = subdivide_tet_file(input_path, output_path, bone_trimeshes,
                                max_edge_len=args.max_edge_len,
                                bone_dist=args.bone_dist)
        dt = time.time() - t0
        if n > 0:
            with open(output_path, 'rb') as f:
                d = pickle.load(f)
            print(f"  {fname}: +{n} midpoints → {d['vertices'].shape[0]} verts, "
                  f"{d['tetrahedra'].shape[0]} tets, {dt:.2f}s")
            total_subdiv += n
        else:
            print(f"  {fname}: no subdivision needed")

    print(f"\nTotal: {total_subdiv} edges subdivided")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
