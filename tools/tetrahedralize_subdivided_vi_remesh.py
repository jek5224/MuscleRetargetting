#!/usr/bin/env python3
"""TetWild-remesh the open subdivided VI into a direct ARAP tet mesh."""
import argparse
import json
import os
import pickle

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from tools.tet_from_original_obj import _cap_loop, _find_boundary_loops
from tools.tetrahedralize_subdivided_vi import boundary_faces, quality_statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=(
        "Zygote_Meshes_251229/Muscle/UpLeg/"
        "L_Vastus_Intermedius_Subdivided.obj"))
    ap.add_argument("--output", default=(
        "tet/vi_subdivided_candidates/tetwild_direct_remesh.npz"))
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--edge-length-r", type=float, default=0.012)
    ap.add_argument(
        "--attach-full-caps", action="store_true",
        help="Use every remeshed artificial-cap vertex as an attachment. "
             "This avoids collapsing a dense open contour to a sparse set of "
             "nearest remesh vertices.")
    args = ap.parse_args()

    source = trimesh.load(args.input, process=False, maintain_order=True)
    source_v = np.asarray(source.vertices, dtype=np.float64) * args.scale
    source_f = np.asarray(source.faces, dtype=np.int32)
    loops = _find_boundary_loops(source_v, source_f)
    if len(loops) != 2:
        raise RuntimeError(f"expected two contours, got {len(loops)}")
    loops = sorted(loops, key=lambda x: source_v[x, 1].mean(), reverse=True)
    origin_source = np.asarray(loops[0], dtype=np.int32)
    insertion_source = np.asarray(loops[1], dtype=np.int32)

    closed_v = source_v.tolist()
    closed_f = source_f.tolist()
    cap_f = []
    cap_f_groups = []
    for loop in loops:
        faces, center = _cap_loop(list(map(int, loop)), closed_v)
        if center is not None:
            raise RuntimeError("unexpected cap centroid fallback")
        cap_f.extend(faces)
        cap_f_groups.append(np.asarray(faces, dtype=np.int32))
        closed_f.extend(faces)
    closed_v = np.asarray(closed_v, dtype=np.float64)
    closed_f = np.asarray(closed_f, dtype=np.int32)
    closed = trimesh.Trimesh(closed_v, closed_f, process=False)
    closed.fix_normals(multibody=True)

    import wildmeshing
    tet = wildmeshing.Tetrahedralizer(
        stop_quality=8.0, max_its=100, stage=2, epsilon=0.0005,
        edge_length_r=args.edge_length_r, skip_simplify=True,
        coarsen=False)
    tet.set_log_level(2)
    tet.set_mesh(np.asarray(closed.vertices), np.asarray(closed.faces))
    tet.tetrahedralize()
    tet_output = tet.get_tet_mesh(
        use_input_for_wn=True, manifold_surface=True,
        correct_surface_orientation=False)
    out_v, out_t = tet_output[0], tet_output[1]
    out_v = np.asarray(out_v, dtype=np.float64)
    out_t = np.asarray(out_t, dtype=np.int32)
    stats = quality_statistics(out_v, out_t)
    sim_f = boundary_faces(out_t)

    # Separate anatomical skin from artificial caps by nearest source facet.
    anatomical = trimesh.Trimesh(source_v, source_f, process=False)
    caps = trimesh.Trimesh(closed_v, np.asarray(cap_f), process=False)
    centers = out_v[sim_f].mean(axis=1)
    _, anatomical_d, _ = trimesh.proximity.closest_point(anatomical, centers)
    _, cap_d, _ = trimesh.proximity.closest_point(caps, centers)
    visible_f = sim_f[anatomical_d <= cap_d]
    closure_f = sim_f[anatomical_d > cap_d]

    # Preserve the complete artificial attachment surfaces in the boundary
    # face array.  cap_face_indices conventionally index render_faces and are
    # also consumed against sim_faces, so both arrays must share this order.
    origin_cap = trimesh.Trimesh(
        closed_v, cap_f_groups[0], process=False)
    insertion_cap = trimesh.Trimesh(
        closed_v, cap_f_groups[1], process=False)
    closure_centers = out_v[closure_f].mean(axis=1)
    _, origin_distance, _ = trimesh.proximity.closest_point(
        origin_cap, closure_centers)
    _, insertion_distance, _ = trimesh.proximity.closest_point(
        insertion_cap, closure_centers)
    origin_faces = closure_f[origin_distance <= insertion_distance]
    insertion_faces = closure_f[insertion_distance < origin_distance]
    ordered_boundary_f = np.vstack(
        (visible_f, origin_faces, insertion_faces)).astype(np.int32)
    origin_cap_face_indices = np.arange(
        len(visible_f), len(visible_f) + len(origin_faces), dtype=np.int32)
    insertion_cap_face_indices = np.arange(
        len(visible_f) + len(origin_faces), len(ordered_boundary_f),
        dtype=np.int32)
    cap_face_indices = np.r_[
        origin_cap_face_indices, insertion_cap_face_indices].astype(np.int32)

    boundary_ids = np.unique(sim_f)
    boundary_v = out_v[boundary_ids]
    boundary_tree = cKDTree(boundary_v)
    origin = np.unique(boundary_ids[
        boundary_tree.query(source_v[origin_source])[1]]).astype(np.int32)
    insertion = np.unique(boundary_ids[
        boundary_tree.query(source_v[insertion_source])[1]]).astype(np.int32)
    if args.attach_full_caps:
        # TetWild is allowed to remesh the artificial closures. Recover the
        # complete origin/insertion caps from closure-face centers instead of
        # reducing each authored contour to a sparse nearest-vertex set.
        origin = np.unique(origin_faces).astype(np.int32)
        insertion = np.unique(insertion_faces).astype(np.int32)
    overlap = np.intersect1d(origin, insertion)
    if len(overlap):
        raise RuntimeError("remeshed attachment sets overlap")

    fixed = {
        **{int(i): ("L_Femur0", "origin") for i in origin},
        **{int(i): ("L_Patella0", "insertion") for i in insertion},
    }
    data = {
        "vertices": out_v.astype(np.float32),
        "tetrahedra": out_t,
        "faces": ordered_boundary_f,
        "render_faces": ordered_boundary_f,
        "sim_faces": ordered_boundary_f,
        "collision_faces": visible_f,
        "cap_face_indices": cap_face_indices,
        "origin_cap_face_indices": origin_cap_face_indices,
        "insertion_cap_face_indices": insertion_cap_face_indices,
        "surface_face_count": int(len(visible_f)),
        "anchor_vertices": np.r_[origin, insertion].astype(np.int32),
        "origin_attachment_vertices": origin,
        "insertion_attachment_vertices": insertion,
        "fixed_verts_with_bones": fixed,
        "anchor_bone_map": {i: bone for i, (bone, _) in fixed.items()},
        "cap_vertex_types": {i: kind for i, (_, kind) in fixed.items()},
        "cap_attachments": np.asarray(
            [[int(i), 0, 0, 0, 0] for i in origin]
            + [[int(i), 0, 1, 0, 0] for i in insertion], dtype=np.int32),
        "attach_skeleton_names": [["L_Femur0", "L_Patella0"]],
        "tetrahedralization_method": "tetwild_direct_remesh",
        "tet_quality_stats": stats,
        "tet_quality_profile": f"tetwild_direct_r{args.edge_length_r:g}",
        "source_obj": args.input,
        "source_obj_faces": source_f,
        "orig_n_verts": int(len(out_v)),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)
    print(f"TetWild direct: {len(out_v)} vertices, {len(out_t)} tets")
    print(f"Visible/closure faces: {len(visible_f)}/{len(closure_f)}")
    print(f"Attachments: {len(origin)}+{len(insertion)}")
    print(f"Quality: {json.dumps(stats)}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
