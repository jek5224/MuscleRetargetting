#!/usr/bin/env python3
"""Build the viewer/ARAP tet asset for the open subdivided left VI mesh.

The two OBJ boundary loops are anatomical attachment contours.  Their exact
OBJ vertex indices are recorded before the surface is capped:

* upper loop -> L_Femur0 (origin)
* lower loop -> L_Patella0 (insertion)

The original surface is not decimated or welded.  Constrained triangulation
closes each opening using only its existing boundary vertices, then TetGen
creates a quality volume mesh.  This preserves an exact surface-to-tet
attachment map while allowing interior Steiner vertices for ARAP quality.
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np
import trimesh

from tools.tet_from_original_obj import _cap_loop, _find_boundary_loops


def boundary_faces(tetrahedra):
    counts = Counter()
    oriented = {}
    for tet in tetrahedra:
        for face in ((tet[0], tet[2], tet[1]), (tet[0], tet[1], tet[3]),
                     (tet[1], tet[2], tet[3]), (tet[0], tet[3], tet[2])):
            key = tuple(sorted(map(int, face)))
            counts[key] += 1
            oriented.setdefault(key, tuple(map(int, face)))
    return np.asarray(
        [oriented[key] for key, count in counts.items() if count == 1],
        dtype=np.int32)


def tetrahedralize(vertices, faces, target_tets, timeout,
                   min_dihedral, min_ratio, allow_boundary_split):
    """TetGen the capped subdivided surface without replacing its boundary."""
    # TetGen's predicates are substantially more reliable for this asset in
    # millimetres than with metre-scale coordinates and ~1e-10 volume bounds.
    scale = 1000.0
    scaled_vertices = vertices * scale
    surface = trimesh.Trimesh(
        vertices=scaled_vertices, faces=faces, process=False)
    volume = abs(float(surface.volume))
    max_volume = max(volume / float(target_tets), 1e-12)
    with tempfile.TemporaryDirectory(prefix="vi_subdiv_tet_", dir="/tmp") as tmp:
        input_path = os.path.join(tmp, "surface.npz")
        output_path = os.path.join(tmp, "volume.npz")
        np.savez(input_path, vertices=scaled_vertices, faces=faces)
        code = f"""
import numpy as np
import tetgen
d=np.load({input_path!r})
t=tetgen.TetGen(d['vertices'].copy(), d['faces'].astype(np.int32).copy())
t.tetrahedralize(order=1, mindihedral={min_dihedral!r}, minratio={min_ratio!r},
                 maxvolume={max_volume!r}, nobisect={not allow_boundary_split!r},
                 steinerleft={max(30000, len(vertices) * 6)})
np.savez({output_path!r}, vertices=np.asarray(t.node),
         tetrahedra=np.asarray(t.elem, dtype=np.int32))
"""
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            timeout=timeout)
        if not proc.returncode:
            data = np.load(output_path)
            tet_vertices = np.asarray(data["vertices"], dtype=np.float64) / scale
            if (len(tet_vertices) < len(vertices) or not np.allclose(
                    tet_vertices[:len(vertices)], vertices, atol=1e-10,
                    rtol=0.0)):
                raise RuntimeError(
                    "TetGen did not preserve the subdivided surface vertices")
            return (tet_vertices,
                    np.asarray(data["tetrahedra"], dtype=np.int32),
                    ("tetgen_split_exact_surface" if allow_boundary_split
                     else "tetgen_exact_surface"))
        raise RuntimeError(
            "Exact-boundary TetGen failed; refusing to substitute a cage:\n"
            + proc.stderr[-4000:])


def quality_statistics(vertices, tetrahedra):
    points = vertices[tetrahedra]
    dm = np.stack((
        points[:, 0] - points[:, 3],
        points[:, 1] - points[:, 3],
        points[:, 2] - points[:, 3]), axis=2)
    determinant = np.linalg.det(dm)
    negative = determinant < 0.0
    if np.any(negative):
        tetrahedra[negative, 0], tetrahedra[negative, 1] = (
            tetrahedra[negative, 1].copy(),
            tetrahedra[negative, 0].copy())
        points = vertices[tetrahedra]
        dm = np.stack((
            points[:, 0] - points[:, 3],
            points[:, 1] - points[:, 3],
            points[:, 2] - points[:, 3]), axis=2)
        determinant = np.linalg.det(dm)
    singular = np.linalg.svd(dm, compute_uv=False)
    condition = singular[:, 0] / np.maximum(singular[:, 2], 1e-30)
    volume = determinant / 6.0
    return {
        "min_signed_volume": float(volume.min(initial=np.inf)),
        "volume_percentiles": np.percentile(
            volume, [0, 1, 5, 50, 95, 99, 100]).tolist(),
        "condition_percentiles": np.percentile(
            condition, [50, 90, 95, 99, 100]).tolist(),
        "inverted_tets": int(np.sum(volume <= 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=("Zygote_Meshes_251229/Muscle/UpLeg/"
                 "L_Vastus_Intermedius_Subdivided.obj"))
    parser.add_argument(
        "--output", default="tet/L_Vastus_Intermedius_Subdivided_tet.npz")
    parser.add_argument(
        "--contours-output",
        default=("Zygote_Meshes_251229/Muscle/UpLeg/"
                 "L_Vastus_Intermedius_Subdivided.attachment_contours.json"))
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--target-tets", type=int, default=24000)
    parser.add_argument("--min-dihedral", type=float, default=10.0)
    parser.add_argument("--min-ratio", type=float, default=1.25)
    parser.add_argument(
        "--cap-style", choices=("ear", "fan"), default="ear",
        help="Triangulation used only for the two artificial closure caps.")
    parser.add_argument(
        "--allow-boundary-split", action="store_true",
        help="Allow surface Steiner vertices while preserving input vertices.")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    with open(args.input, "rb") as stream:
        source_hash = hashlib.sha256(stream.read()).hexdigest()
    mesh = trimesh.load(args.input, process=False, maintain_order=True)
    obj_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    obj_faces = np.asarray(mesh.faces, dtype=np.int32)
    loops = _find_boundary_loops(obj_vertices, obj_faces)
    if len(loops) != 2:
        raise RuntimeError(
            f"Expected exactly two VI boundary loops, found "
            f"{[len(loop) for loop in loops]}")
    loops = sorted(loops, key=lambda loop: obj_vertices[loop, 1].mean(),
                   reverse=True)
    origin_obj = np.asarray(loops[0], dtype=np.int32)
    insertion_obj = np.asarray(loops[1], dtype=np.int32)

    contour_record = {
        "source_obj": args.input,
        "source_sha256": source_hash,
        "index_space": "raw OBJ vertex order, zero-based",
        "origin": {
            "bone": "L_Femur0",
            "vertex_indices": origin_obj.tolist(),
            "centroid_obj_units": obj_vertices[origin_obj].mean(0).tolist(),
        },
        "insertion": {
            "bone": "L_Patella0",
            "vertex_indices": insertion_obj.tolist(),
            "centroid_obj_units": obj_vertices[insertion_obj].mean(0).tolist(),
        },
    }
    os.makedirs(os.path.dirname(args.contours_output), exist_ok=True)
    with open(args.contours_output, "w") as stream:
        json.dump(contour_record, stream, indent=2)
        stream.write("\n")

    vertices_list = (obj_vertices * args.scale).tolist()
    closed_faces = obj_faces.tolist()
    cap_faces = []
    cap_centers = []
    for loop in (origin_obj.tolist(), insertion_obj.tolist()):
        if args.cap_style == "fan":
            center = len(vertices_list)
            vertices_list.append(
                np.mean(np.asarray(vertices_list)[loop], axis=0).tolist())
            faces = [[loop[i], loop[(i + 1) % len(loop)], center]
                     for i in range(len(loop))]
            cap_centers.append(center)
        else:
            faces, center = _cap_loop(loop, vertices_list)
            if center is not None:
                raise RuntimeError(
                    "Ear cap unexpectedly required a centroid fallback")
        cap_faces.extend(faces)
        closed_faces.extend(faces)
    closed_vertices = np.asarray(vertices_list, dtype=np.float64)
    closed_faces = np.asarray(closed_faces, dtype=np.int32)
    closed = trimesh.Trimesh(
        vertices=closed_vertices, faces=closed_faces, process=False)
    closed.fix_normals(multibody=True)
    if not closed.is_watertight:
        raise RuntimeError("Constrained caps did not produce a watertight PLC")

    print(
        f"Surface: {len(obj_vertices)} vertices, {len(obj_faces)} faces; "
        f"origin={len(origin_obj)}, insertion={len(insertion_obj)}, "
        f"cap_faces={len(cap_faces)}", flush=True)
    tet_vertices, tetrahedra, tet_method = tetrahedralize(
        closed.vertices, closed.faces, args.target_tets, args.timeout,
        args.min_dihedral, args.min_ratio, args.allow_boundary_split)
    stats = quality_statistics(tet_vertices, tetrahedra)
    if stats["inverted_tets"]:
        raise RuntimeError(f"Tet mesh remains inverted: {stats}")

    # With nobisect=True, raw OBJ vertices retain their indices and are actual
    # tet boundary DOFs. No render cage or displacement embedding is involved.
    origin_tet = origin_obj.copy()
    insertion_tet = insertion_obj.copy()

    render_faces = closed.faces.astype(np.int32)
    sim_faces = boundary_faces(tetrahedra)
    cap_start = len(obj_faces)
    cap_face_indices = np.arange(
        cap_start, len(render_faces), dtype=np.int32)
    anchors = np.r_[origin_tet, insertion_tet].astype(np.int32)
    fixed = {
        **{int(index): ("L_Femur0", "origin") for index in origin_tet},
        **{int(index): ("L_Patella0", "insertion") for index in insertion_tet},
    }
    data = {
        "vertices": tet_vertices.astype(np.float32),
        "tetrahedra": tetrahedra.astype(np.int32),
        "faces": render_faces,
        "render_faces": render_faces,
        "sim_faces": sim_faces,
        "cap_face_indices": cap_face_indices,
        "surface_face_count": int(len(obj_faces)),
        "anchor_vertices": anchors,
        "origin_attachment_vertices": origin_tet,
        "insertion_attachment_vertices": insertion_tet,
        "fixed_verts_with_bones": fixed,
        "anchor_bone_map": {
            int(index): bone for index, (bone, _) in fixed.items()},
        "cap_vertex_types": {
            int(index): end_type for index, (_, end_type) in fixed.items()},
        "cap_attachments": np.asarray(
            [[int(index), 0, 0, 0, 0] for index in origin_tet]
            + [[int(index), 0, 1, 0, 0] for index in insertion_tet],
            dtype=np.int32),
        "attach_skeleton_names": [["L_Femur0", "L_Patella0"]],
        "source_obj": args.input,
        "source_obj_sha256": source_hash,
        "source_obj_vertex_count": int(len(obj_vertices)),
        "source_obj_faces": obj_faces,
        "source_obj_nearest_tet_vertex": np.arange(
            len(obj_vertices), dtype=np.int32),
        "source_surface_nearest_error_max": 0.0,
        "attachment_contours_file": args.contours_output,
        "tet_quality_stats": stats,
        "tet_quality_profile": (
            f"subdivided_vi_exact_{args.cap_style}_"
            f"q{args.min_ratio:g}_d{args.min_dihedral:g}_"
            f"target{args.target_tets}"),
        "tetrahedralization_method": tet_method,
        "orig_n_verts": int(len(tet_vertices)),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as stream:
        pickle.dump(data, stream)
    print(
        f"Tet: {len(tet_vertices)} vertices, {len(tetrahedra)} tets, "
        f"{len(sim_faces)} boundary faces", flush=True)
    print(
        f"Exact surface boundary; attachments="
        f"{len(origin_tet)}+{len(insertion_tet)}; "
        f"condition p50/p95/p99/max="
        f"{np.asarray(stats['condition_percentiles'])[[0,2,3,4]]}",
        flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
