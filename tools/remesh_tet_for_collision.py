#!/usr/bin/env python3
"""Create collision-ready tet meshes by subdividing contour surfaces near bones.

Takes existing contour tet meshes, subdivides their surface to shorter edges
near bone surfaces, re-tetrahedralizes with TetGen, and saves with barycentric
mapping from original contour vertices to new tet elements.

Usage:
    python tools/remesh_tet_for_collision.py --input-dir tet --output-dir tet_fine
    python tools/bake_layered.py --tet-dir tet_fine --frame-independent ...
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229/"
MESH_SCALE = 0.01


def compute_barycentric(point, tet_verts):
    """Compute barycentric coordinates of point in tetrahedron."""
    v0, v1, v2, v3 = tet_verts
    T = np.column_stack([v0 - v3, v1 - v3, v2 - v3])
    try:
        bary3 = np.linalg.solve(T, point - v3)
    except np.linalg.LinAlgError:
        return np.array([-1, -1, -1, -1])
    return np.array([bary3[0], bary3[1], bary3[2], 1.0 - bary3.sum()])


def build_vert_to_tet_map(tets, n_verts):
    """Build mapping from vertex index to list of incident tet indices."""
    v2t = [[] for _ in range(n_verts)]
    for ti, t in enumerate(tets):
        for vi in t:
            v2t[int(vi)].append(ti)
    return v2t


def build_mapping(query_verts, tet_verts, tet_elems):
    """Build barycentric mapping from query vertices to tet elements.

    For each query vertex, find the containing tet and compute barycentric coords.
    Falls back to nearest tet with clamped barycentrics for surface vertices.
    """
    kdtree = cKDTree(tet_verts)
    v2t = build_vert_to_tet_map(tet_elems, len(tet_verts))
    mapping = []

    for qi in range(len(query_verts)):
        point = query_verts[qi]
        _, nearby_idx = kdtree.query(point, k=min(30, len(tet_verts)))
        candidate_tets = set()
        for vi in nearby_idx:
            candidate_tets.update(v2t[int(vi)])

        best_tet = -1
        best_bary = None
        best_min_bary = -float('inf')

        for ti in candidate_tets:
            bary = compute_barycentric(point, tet_verts[tet_elems[ti]])
            min_bary = bary.min()
            if min_bary >= -1e-6:
                best_tet = ti
                best_bary = bary
                break
            elif min_bary > best_min_bary:
                best_tet = ti
                best_bary = bary
                best_min_bary = min_bary

        if best_bary is not None:
            best_bary = np.maximum(best_bary, 0.0)
            s = best_bary.sum()
            if s > 1e-10:
                best_bary /= s
            mapping.append((best_tet, best_bary))
        else:
            mapping.append((0, np.array([0.25, 0.25, 0.25, 0.25])))

    return mapping


def tetrahedralize_surface(vertices, faces):
    """Tetrahedralize a closed surface mesh using TetGen."""
    import pymeshfix

    # Repair mesh
    fixer = pymeshfix.MeshFix(vertices.astype(np.float64), faces.astype(np.int32))
    fixer.repair(verbose=False)
    v_fixed, f_fixed = fixer.v, fixer.f

    # TetGen
    try:
        import tetgen
        tg = tetgen.TetGen(v_fixed, f_fixed)
        tg.tetrahedralize(order=1, mindihedral=0, quality=False)
        tet_verts = np.array(tg.node, dtype=np.float64)
        tet_elems = np.array(tg.elem, dtype=np.int32)
    except Exception as e:
        print(f"    TetGen failed: {e}, using Delaunay fallback")
        from scipy.spatial import Delaunay
        mesh_check = trimesh.Trimesh(vertices=v_fixed, faces=f_fixed, process=False)
        dl = Delaunay(v_fixed)
        centroids = v_fixed[dl.simplices].mean(axis=1)
        inside = mesh_check.contains(centroids)
        tet_elems = dl.simplices[inside]
        tet_verts = v_fixed

    # Fix orientation (positive volume)
    v = tet_verts[tet_elems]
    vols = np.einsum('ij,ij->i', v[:, 1] - v[:, 0],
                     np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]))
    neg = vols < 0
    tet_elems[neg, 1], tet_elems[neg, 2] = tet_elems[neg, 2].copy(), tet_elems[neg, 1].copy()

    return tet_verts.astype(np.float32), tet_elems.astype(np.int32), \
           v_fixed.astype(np.float32), f_fixed.astype(np.int32)


def subdivide_surface_near_bones(verts, faces, bone_kdtree,
                                  max_edge_len=0.008, bone_dist=0.025, max_passes=4):
    """Subdivide surface edges near bones to target length.

    Only subdivides edges where the midpoint is within bone_dist of a bone.
    Preserves all original vertex indices (midpoints appended).
    """
    verts = list(verts)
    faces = list([list(f) for f in faces])
    total_added = 0

    for pass_i in range(max_passes):
        verts_arr = np.array(verts)
        edges = set()
        for f in faces:
            for i in range(3):
                e = (min(f[i], f[(i+1) % 3]), max(f[i], f[(i+1) % 3]))
                edges.add(e)
        edges = list(edges)

        # Find long edges near bones
        to_split = set()
        for a, b in edges:
            length = np.linalg.norm(verts_arr[a] - verts_arr[b])
            if length <= max_edge_len:
                continue
            mid = 0.5 * (verts_arr[a] + verts_arr[b])
            d, _ = bone_kdtree.query(mid)
            if d < bone_dist:
                to_split.add((a, b))

        if not to_split:
            break

        # Create midpoints
        midpoint_cache = {}
        for a, b in to_split:
            mid = 0.5 * (np.array(verts[a], dtype=np.float64) +
                         np.array(verts[b], dtype=np.float64))
            midpoint_cache[(a, b)] = len(verts)
            verts.append(mid.astype(np.float32))

        # Split faces
        new_faces = []
        for f in faces:
            f_edges = [(f[i], f[(i+1) % 3]) for i in range(3)]
            splits = []
            for a, b in f_edges:
                key = (min(a, b), max(a, b))
                if key in midpoint_cache:
                    splits.append((a, b, midpoint_cache[key]))

            if not splits:
                new_faces.append(f)
            elif len(splits) == 1:
                a, b, m = splits[0]
                c = [v for v in f if v != a and v != b][0]
                new_faces.append([a, m, c])
                new_faces.append([m, b, c])
            elif len(splits) == 2:
                # Two edges split — create 3 triangles
                s0, s1 = splits[0], splits[1]
                a0, b0, m0 = s0
                a1, b1, m1 = s1
                # Find shared vertex between the two split edges
                shared = set([a0, b0]) & set([a1, b1])
                if shared:
                    sv = shared.pop()
                    other0 = a0 if b0 == sv else b0
                    other1 = a1 if b1 == sv else b1
                    new_faces.append([sv, m0, m1])
                    new_faces.append([m0, other0, m1])
                    new_faces.append([other0, other1, m1])
                else:
                    new_faces.append(f)
            else:
                # All 3 edges split — create 4 triangles
                m01 = midpoint_cache.get((min(f[0], f[1]), max(f[0], f[1])))
                m12 = midpoint_cache.get((min(f[1], f[2]), max(f[1], f[2])))
                m02 = midpoint_cache.get((min(f[0], f[2]), max(f[0], f[2])))
                if m01 is not None and m12 is not None and m02 is not None:
                    new_faces.append([f[0], m01, m02])
                    new_faces.append([m01, f[1], m12])
                    new_faces.append([m02, m12, f[2]])
                    new_faces.append([m01, m12, m02])
                else:
                    new_faces.append(f)

        n_added = len(midpoint_cache)
        total_added += n_added
        faces = new_faces

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32), total_added


def process_muscle(input_path, output_path, bone_kdtree, bone_trimeshes,
                    max_edge_len=0.008, bone_dist=0.025):
    """Process one muscle tet file: subdivide surface, re-tet, build mapping."""
    with open(input_path, 'rb') as f:
        orig_data = pickle.load(f)

    orig_verts = orig_data['vertices']
    orig_tets = orig_data['tetrahedra']
    orig_faces = orig_data.get('render_faces', orig_data.get('faces'))
    if orig_verts is None or orig_tets is None:
        with open(output_path, 'wb') as f:
            pickle.dump(orig_data, f)
        return 0

    n_orig = len(orig_verts)

    # Subdivide surface mesh near bones
    sub_verts, sub_faces, n_added = subdivide_surface_near_bones(
        orig_verts, orig_faces, bone_kdtree,
        max_edge_len=max_edge_len, bone_dist=bone_dist)

    if n_added == 0:
        # No subdivision needed — copy with mapping
        data = dict(orig_data)
        data['orig_to_fine_mapping'] = [(i, np.array([1, 0, 0, 0], dtype=np.float32))
                                         for i in range(n_orig)]  # identity
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        return 0

    # Re-tetrahedralize the subdivided surface
    tet_verts, tet_elems, surf_verts_repaired, surf_faces_repaired = \
        tetrahedralize_surface(sub_verts, sub_faces)

    # Identify anchor vertices in new tet mesh (match by position)
    orig_anchors = orig_data.get('anchor_vertices', [])
    orig_anchor_positions = orig_verts[orig_anchors]
    new_kdtree = cKDTree(tet_verts)
    new_anchors = []
    for ap in orig_anchor_positions:
        d, idx = new_kdtree.query(ap)
        if d < 0.001:  # 1mm tolerance
            new_anchors.append(int(idx))
    new_anchors = sorted(set(new_anchors))

    # Identify cap faces (all vertices in anchor set)
    anchor_set = set(new_anchors)
    new_cap_faces = []
    for fi, face in enumerate(surf_faces_repaired):
        if all(int(v) in anchor_set for v in face):
            new_cap_faces.append(fi)

    # Build barycentric mapping: original contour verts → new tet elements
    mapping = build_mapping(orig_verts, tet_verts, tet_elems)

    # Rebuild cap_attachments by position matching
    orig_cap_attach = orig_data.get('cap_attachments')
    new_cap_attach = None
    if orig_cap_attach is not None and len(orig_cap_attach) > 0:
        new_cap_attach = []
        for ca in orig_cap_attach:
            anchor_idx = int(ca[0])
            anchor_pos = orig_verts[anchor_idx]
            d, new_idx = new_kdtree.query(anchor_pos)
            if d < 0.001:
                new_ca = list(ca)
                new_ca[0] = new_idx
                new_cap_attach.append(new_ca)
        if new_cap_attach:
            new_cap_attach = np.array(new_cap_attach, dtype=np.int32)

    # Build new data dict
    data = {}
    # Copy non-vertex-dependent fields
    for key in orig_data:
        if key not in ('vertices', 'tetrahedra', 'faces', 'render_faces', 'sim_faces',
                        'cap_face_indices', 'anchor_vertices', 'cap_attachments',
                        'vertex_contour_level', 'contour_to_tet_mapping', 'mvc_weights',
                        'surface_face_count'):
            data[key] = orig_data[key]

    # Set new mesh data
    data['vertices'] = tet_verts.astype(np.float32)
    data['tetrahedra'] = tet_elems.astype(np.int32)
    data['faces'] = surf_faces_repaired.astype(np.int32)
    data['render_faces'] = surf_faces_repaired.astype(np.int32)
    data['sim_faces'] = None
    data['cap_face_indices'] = np.array(new_cap_faces, dtype=np.int32)
    data['anchor_vertices'] = np.array(new_anchors, dtype=np.int32)
    data['surface_face_count'] = len(surf_faces_repaired)
    if new_cap_attach is not None:
        data['cap_attachments'] = new_cap_attach

    # Per-vertex arrays: set to defaults for new mesh
    data['vertex_contour_level'] = np.full(len(tet_verts), -1, dtype=np.int32)
    data['contour_to_tet_mapping'] = [None] * len(tet_verts)
    data['mvc_weights'] = None  # Will be recomputed if needed

    # Store mapping for position conversion
    data['orig_to_fine_mapping'] = mapping
    data['orig_n_verts'] = n_orig

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    return n_added


def main():
    parser = argparse.ArgumentParser(description="Remesh contour tets for collision")
    parser.add_argument("--input-dir", default="tet")
    parser.add_argument("--output-dir", default="tet_fine")
    parser.add_argument("--max-edge-len", type=float, default=0.008,
                        help="Max edge length near bones in meters (default: 8mm)")
    parser.add_argument("--bone-dist", type=float, default=0.025,
                        help="Distance threshold from bone surface (default: 25mm)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build bone KDTree at rest pose
    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    skel.setPositions(np.zeros(skel.getNumDofs()))

    bone_trimeshes = []
    all_bone_verts = []
    skel_dir = os.path.join(ZYGOTE_DIR, "Skeleton")
    for fname in sorted(os.listdir(skel_dir)):
        if not fname.endswith(".obj"):
            continue
        skel_tri = trimesh.load_mesh(os.path.join(skel_dir, fname))
        skel_tri.vertices *= MESH_SCALE
        bone_trimeshes.append(skel_tri)
        all_bone_verts.append(skel_tri.vertices)

    bone_kdtree = cKDTree(np.vstack(all_bone_verts))
    print(f"Loaded {len(bone_trimeshes)} bone meshes")

    # Process each tet file
    total_added = 0
    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith("_tet.npz"):
            continue
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)
        t0 = time.time()
        n = process_muscle(input_path, output_path, bone_kdtree, bone_trimeshes,
                           max_edge_len=args.max_edge_len, bone_dist=args.bone_dist)
        dt = time.time() - t0
        if n > 0:
            with open(output_path, 'rb') as f:
                d = pickle.load(f)
            print(f"  {fname}: +{n} surface verts → {d['vertices'].shape[0]} tet verts, "
                  f"{d['tetrahedra'].shape[0]} tets, {len(d['cap_face_indices'])} caps, {dt:.2f}s")
            total_added += n
        else:
            print(f"  {fname}: no subdivision needed")

    print(f"\nTotal: {total_added} surface vertices added")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
