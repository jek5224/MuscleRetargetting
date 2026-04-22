#!/usr/bin/env python3
"""Create standard tet files from original Zygote OBJ muscle meshes.

Uses the pipeline from bake_original_mesh.py: boundary loop detection,
cap classification, TetGen tetrahedralization, bone assignment.
Saves in the same format as tet/ for use with bake_layered.py.

Usage:
    python tools/tet_from_original_obj.py --output-dir tet_orig_open
"""
import argparse
import os
import pickle
import sys

import numpy as np
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.bake_original_mesh import (
    find_boundary_loops,
    cap_boundary_loop,
    load_and_identify_boundaries,
    tetrahedralize_mesh,
    identify_fixed_and_assign_bones,
)

SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229/"
OBJ_DIR = "Zygote_Meshes/Muscle/"
MESH_SCALE = 0.01


def process_muscle(muscle_name, obj_path, contour_tet_path, output_path, skel, mesh_info):
    """Create tet file from OBJ with proper boundary/cap handling."""

    # Parse muscle XML data for bone assignment
    from tools.bake_original_mesh import parse_muscle_xml
    all_xml = parse_muscle_xml()
    # Match muscle name (try with and without L_/R_ prefix)
    short_name = muscle_name.replace('L_', '').replace('R_', '')
    xml_data = all_xml.get(muscle_name) or all_xml.get(short_name)
    if xml_data is None:
        print(f"    No XML data for {muscle_name}")
        return 0

    # Load OBJ and identify boundaries
    vertices, faces, boundary_verts = load_and_identify_boundaries(obj_path, xml_data)
    n_boundary = len(boundary_verts)
    if n_boundary == 0:
        print(f"    No boundary vertices found")
        return 0

    # Tetrahedralize (pymeshfix closes caps, TetGen creates tets)
    tet_verts, tet_elems, surf_verts, surf_faces = tetrahedralize_mesh(
        vertices.astype(np.float64), faces.astype(np.int32))

    # Extract tet boundary faces, exclude cap faces (cap faces deform badly)
    from collections import Counter, defaultdict
    face_count = Counter()
    face_orient = {}
    for t in tet_elems:
        for f in [(t[0], t[2], t[1]), (t[0], t[1], t[3]), (t[1], t[2], t[3]), (t[0], t[3], t[2])]:
            key = tuple(sorted(f))
            face_count[key] += 1
            if key not in face_orient:
                face_orient[key] = f
    all_surf = [face_orient[k] for k, c in face_count.items() if c == 1]
    sim_faces = np.array(all_surf, dtype=np.int32)

    # render_faces will be set after fixed_verts is computed (below)

    # Find anchor vertices using XML waypoint positions (not boundary loops)
    # Same approach as contour mesh: vertices near origin/insertion waypoints → fixed
    from scipy.spatial import cKDTree

    origin_pts = np.vstack([s[2] for s in xml_data]) if xml_data else np.zeros((1, 3))
    insertion_pts = np.vstack([s[3] for s in xml_data]) if xml_data else np.zeros((1, 3))
    origin_bone = xml_data[0][0] if xml_data else None
    insertion_bone = xml_data[0][1] if xml_data else None

    # For each XML waypoint, find the N closest tet vertices
    fixed_verts = {}
    tet_kd = cKDTree(tet_verts)
    n_per_waypoint = 3  # Fix ~3 vertices per waypoint

    for wp_pos in origin_pts:
        dists, idxs = tet_kd.query(wp_pos, k=n_per_waypoint)
        for d, vi in zip(dists, idxs):
            if d < 0.02:  # 20mm from waypoint
                fixed_verts[int(vi)] = (origin_bone, 'origin')

    for wp_pos in insertion_pts:
        dists, idxs = tet_kd.query(wp_pos, k=n_per_waypoint)
        for d, vi in zip(dists, idxs):
            if d < 0.02:
                fixed_verts[int(vi)] = (insertion_bone, 'insertion')

    # Compute local anchors at rest pose
    skel.setPositions(np.zeros(skel.getNumDofs()))
    local_anchors = {}
    initial_transforms = {}
    for vi, (bone_name, end_type) in fixed_verts.items():
        body_node = skel.getBodyNode(bone_name)
        if body_node is None:
            continue
        wt = body_node.getWorldTransform()
        R = wt.rotation(); t = wt.translation()
        if bone_name not in initial_transforms:
            initial_transforms[bone_name] = (R.copy(), t.copy())
        local_anchors[vi] = (bone_name, (R.T @ (tet_verts[vi] - t)).copy())

    anchor_vertices = sorted(fixed_verts.keys())
    anchor_set = set(anchor_vertices)
    # Render faces: ORIGINAL OBJ faces remapped to tet vertex indices
    # Tet boundary faces deform badly (8.8% normal flips, 136 >3x area change)
    # OBJ faces are the artist's original surface — deform properly
    from scipy.spatial import cKDTree as _cKDTree2
    _tet_kd2 = _cKDTree2(tet_verts)
    _, _obj_to_tet = _tet_kd2.query(vertices)
    render_faces = _obj_to_tet[faces].astype(np.int32)
    # cap_face_indices must index into render_faces (consumers: init_soft_body,
    # draw code treat it as render-face indices). OBJ surface has open caps, so
    # no render face has all 3 verts anchored — list is empty for original mesh.
    # anchor_vertices already covers all cap verts via init_soft_body.
    cap_face_indices = [fi for fi, f in enumerate(render_faces)
                        if all(int(v) in anchor_set for v in f)]

    origin_verts = [v for v, (b, t) in fixed_verts.items() if t == 'origin']
    insertion_verts = [v for v, (b, t) in fixed_verts.items() if t == 'insertion']
    cap_attachments = []
    if origin_verts:
        cap_attachments.append([origin_verts[0], 0, 0, 0, 0])
    if insertion_verts:
        cap_attachments.append([insertion_verts[0], 0, 1, 0, 0])

    # Copy metadata from contour tet
    contour_data = {}
    if contour_tet_path and os.path.exists(contour_tet_path):
        with open(contour_tet_path, 'rb') as f:
            contour_data = pickle.load(f)

    # Build output
    data = {
        'vertices': tet_verts.astype(np.float32),
        'tetrahedra': tet_elems.astype(np.int32),
        'faces': render_faces,
        'render_faces': render_faces,
        'sim_faces': sim_faces,
        'cap_face_indices': np.array(cap_face_indices, dtype=np.int32),
        'anchor_vertices': np.array(anchor_vertices, dtype=np.int32),
        'cap_attachments': np.array(cap_attachments, dtype=np.int32) if cap_attachments else np.zeros((0, 5), dtype=np.int32),
        'surface_face_count': len(render_faces),
        'vertex_contour_level': np.full(len(tet_verts), -1, dtype=np.int32),
        'contour_to_tet_mapping': [None] * len(tet_verts),
        'mvc_weights': None,
        # Store for bake_layered.py manual init
        'cap_vertex_types': {v: t for v, (b, t) in fixed_verts.items()},
        'fixed_verts_with_bones': fixed_verts,
        'orig_n_verts': len(tet_verts),
    }

    # Set attach_skeleton_names from XML (contour tet may not have it)
    if origin_bone and insertion_bone:
        data['attach_skeleton_names'] = [[origin_bone, insertion_bone]]

    # Copy other metadata from contour tet
    for key in ['attach_skeletons', 'attach_skeletons_sub',
                'waypoints', 'waypoint_bary_coords', 'contours', 'fiber_architecture',
                'bounding_planes', 'draw_contour_stream', 'fiber_sampling_seed',
                'stream_contours', 'stream_bounding_planes', 'stream_groups',
                '_stream_endpoints']:
        if key in contour_data:
            data[key] = contour_data[key]

    # Build contour mapping for position conversion back to contour mesh
    if contour_tet_path and os.path.exists(contour_tet_path):
        from tools.remesh_tet_for_collision import build_mapping
        contour_verts = contour_data['vertices']
        mapping = build_mapping(contour_verts, tet_verts, tet_elems)
        data['contour_mapping'] = mapping
        data['contour_n_verts'] = len(contour_verts)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    return len(tet_verts)


def main():
    parser = argparse.ArgumentParser(description="Create tet from original OBJ meshes")
    parser.add_argument("--output-dir", default="tet_orig_open")
    parser.add_argument("--contour-dir", default="tet")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from core.dartHelper import saveSkeletonInfo, buildFromInfo
    skel_info, root_name, bvh_info, _pd, mesh_info, _smpl = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    skel.setPositions(np.zeros(skel.getNumDofs()))

    # Map muscle names to OBJ paths
    for obj_name in sorted(os.listdir(OBJ_DIR)):
        if not obj_name.endswith('.obj'):
            continue
        # Parse name: UpLegB_L_Vastus_Lateralis.obj → L_Vastus_Lateralis
        parts = obj_name.replace('.obj', '').split('_', 1)
        if len(parts) < 2:
            continue
        muscle_name = parts[1]  # e.g., L_Vastus_Lateralis

        obj_path = os.path.join(OBJ_DIR, obj_name)
        contour_path = os.path.join(args.contour_dir, f"{muscle_name}_tet.npz")
        output_path = os.path.join(args.output_dir, f"{muscle_name}_tet.npz")

        if not os.path.exists(contour_path):
            continue

        try:
            n = process_muscle(muscle_name, obj_path, contour_path, output_path, skel, mesh_info)
            if n > 0:
                with open(output_path, 'rb') as f:
                    d = pickle.load(f)
                print(f"  {muscle_name}: {n} tet verts, {len(d['anchor_vertices'])} anchors, "
                      f"{len(d['cap_face_indices'])} cap faces")
        except Exception as e:
            print(f"  {muscle_name}: ERROR {e}")
            import traceback; traceback.print_exc()

    print(f"\nOutput: {args.output_dir}/")


if __name__ == "__main__":
    main()
