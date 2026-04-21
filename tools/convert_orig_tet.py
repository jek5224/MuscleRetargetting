#!/usr/bin/env python3
"""Convert tet_orig/ format to standard tet/ format for bake_layered.py.

tet_orig/ has: tet_vertices, tet_elements, cap_vertices, boundary_vertices
tet/ needs: vertices, tetrahedra, faces, render_faces, anchor_vertices,
            cap_face_indices, cap_attachments, attach_skeleton_names, etc.

Copies missing metadata (attach_skeleton_names, waypoints) from contour tet.

Usage:
    python tools/convert_orig_tet.py --input-dir tet_orig --output-dir tet_orig_std --contour-dir tet
"""
import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def extract_surface_faces(tets):
    """Extract boundary faces from tet mesh."""
    face_count = Counter()
    face_orient = {}
    for t in tets:
        v0, v1, v2, v3 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        for f in [(v0, v2, v1), (v0, v1, v3), (v1, v2, v3), (v0, v3, v2)]:
            key = tuple(sorted(f))
            face_count[key] += 1
            if key not in face_orient:
                face_orient[key] = f
    return np.array([face_orient[k] for k, c in face_count.items() if c == 1], dtype=np.int32)


def convert_one(input_path, output_path, contour_path):
    """Convert one tet_orig file to standard format."""
    with open(input_path, 'rb') as f:
        orig = pickle.load(f)

    verts = orig['tet_vertices']
    tets = orig['tet_elements']
    cap_verts = orig.get('cap_vertices', {})  # {vi: 'origin'/'insertion'}

    # Extract surface faces
    faces = extract_surface_faces(tets)

    # Anchor vertices = all cap vertices
    anchor_vertices = sorted(cap_verts.keys())

    # Cap face indices: faces where ALL vertices are cap vertices
    anchor_set = set(anchor_vertices)
    cap_face_indices = []
    for fi, face in enumerate(faces):
        if all(int(v) in anchor_set for v in face):
            cap_face_indices.append(fi)

    # Cap attachments: one representative per group (origin, insertion)
    origin_verts = [v for v, t in cap_verts.items() if t == 'origin']
    insertion_verts = [v for v, t in cap_verts.items() if t == 'insertion']
    cap_attachments = []
    if origin_verts:
        cap_attachments.append([origin_verts[0], 0, 0, 0, 0])
    if insertion_verts:
        cap_attachments.append([insertion_verts[0], 0, 1, 0, 0])
    cap_attachments = np.array(cap_attachments, dtype=np.int32) if cap_attachments else np.zeros((0, 5), dtype=np.int32)

    # Copy metadata from contour tet
    contour_data = {}
    if contour_path and os.path.exists(contour_path):
        with open(contour_path, 'rb') as f:
            contour_data = pickle.load(f)

    # Build output
    data = {
        'vertices': verts.astype(np.float32),
        'tetrahedra': tets.astype(np.int32),
        'faces': faces,
        'render_faces': faces,
        'sim_faces': None,
        'cap_face_indices': np.array(cap_face_indices, dtype=np.int32),
        'anchor_vertices': np.array(anchor_vertices, dtype=np.int32),
        'cap_attachments': cap_attachments,
        'surface_face_count': len(faces),
        'vertex_contour_level': np.full(len(verts), -1, dtype=np.int32),
        'contour_to_tet_mapping': [None] * len(verts),
        'mvc_weights': None,
        # Store cap type per vertex for bone assignment override
        'cap_vertex_types': cap_verts,
        'orig_n_verts': len(verts),  # All verts are "original"
    }

    # Copy from contour tet
    for key in ['attach_skeleton_names', 'attach_skeletons', 'attach_skeletons_sub',
                'waypoints', 'waypoint_bary_coords', 'contours', 'fiber_architecture',
                'bounding_planes', 'draw_contour_stream', 'fiber_sampling_seed',
                'stream_contours', 'stream_bounding_planes', 'stream_groups',
                '_stream_endpoints']:
        if key in contour_data:
            data[key] = contour_data[key]

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    return len(verts), len(tets), len(anchor_vertices), len(cap_face_indices)


def main():
    parser = argparse.ArgumentParser(description="Convert tet_orig to standard format")
    parser.add_argument("--input-dir", default="tet_orig")
    parser.add_argument("--output-dir", default="tet_orig_std")
    parser.add_argument("--contour-dir", default="tet")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith("_tet.npz"):
            continue
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)
        contour_path = os.path.join(args.contour_dir, fname)

        try:
            nv, nt, na, nc = convert_one(input_path, output_path, contour_path)
            print(f"  {fname}: {nv} verts, {nt} tets, {na} anchors, {nc} cap faces")
        except Exception as e:
            print(f"  {fname}: ERROR {e}")

    # Copy missing muscles from contour dir
    for fname in sorted(os.listdir(args.contour_dir)):
        if not fname.endswith("_tet.npz"):
            continue
        output_path = os.path.join(args.output_dir, fname)
        if not os.path.exists(output_path):
            import shutil
            shutil.copy(os.path.join(args.contour_dir, fname), output_path)
            print(f"  {fname}: copied from contour (no original mesh)")

    print(f"\nOutput: {args.output_dir}/")


if __name__ == "__main__":
    main()
