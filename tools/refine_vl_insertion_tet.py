#!/usr/bin/env python3
"""Locally refine a closed VL boundary near insertion, then retetrahedralize."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import tetgen
import trimesh
from scipy.spatial import cKDTree

from tools.bake_surface_fast import load_tet
from tools.tetrahedralize_subdivided_vi import boundary_faces, quality_statistics


def refine_faces(vertices, faces, seeds, radius):
    distance = cKDTree(seeds).query(vertices)[0]
    selected = np.any(distance[faces] <= radius, axis=1)
    marked_edges = set()
    for face in faces[selected]:
        for a, b in ((face[0], face[1]), (face[1], face[2]),
                     (face[2], face[0])):
            marked_edges.add(tuple(sorted((int(a), int(b)))))
    out_vertices = vertices.tolist()
    midpoint = {}
    for edge in marked_edges:
        midpoint[edge] = len(out_vertices)
        out_vertices.append(0.5 * (vertices[edge[0]] + vertices[edge[1]]))
    out_faces = []
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        ab = midpoint.get(tuple(sorted((a, b))))
        bc = midpoint.get(tuple(sorted((b, c))))
        ca = midpoint.get(tuple(sorted((c, a))))
        count = sum(v is not None for v in (ab, bc, ca))
        if count == 0:
            out_faces.append((a, b, c))
        elif count == 1:
            if ab is not None:
                out_faces.extend(((a, ab, c), (ab, b, c)))
            elif bc is not None:
                out_faces.extend(((b, bc, a), (bc, c, a)))
            else:
                out_faces.extend(((c, ca, b), (ca, a, b)))
        elif count == 2:
            if ab is None:
                out_faces.extend(((c, ca, bc), (ca, a, b), (ca, b, bc)))
            elif bc is None:
                out_faces.extend(((a, ab, ca), (ab, b, c), (ab, c, ca)))
            else:
                out_faces.extend(((b, bc, ab), (ab, a, c), (ab, c, bc)))
        else:
            out_faces.extend(((a, ab, ca), (ab, b, bc),
                              (ca, bc, c), (ab, bc, ca)))
    return np.asarray(out_vertices), np.asarray(out_faces, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=0.008)
    ap.add_argument("--attachment-band", type=float, default=0.0015)
    args = ap.parse_args()
    data = load_tet(args.input)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    faces = np.asarray(data["sim_faces"], dtype=np.int32)
    old_visible = np.asarray(data["render_faces"], dtype=np.int32)
    old_insertion = np.asarray(data["insertion_attachment_vertices"], dtype=np.int32)
    refined_v, refined_f = refine_faces(
        vertices, faces, vertices[old_insertion], args.radius)

    generator = tetgen.TetGen(refined_v, refined_f)
    out_v, out_t = generator.tetrahedralize(
        order=1, quality=True, minratio=1.4, mindihedral=5.0,
        nobisect=True, quiet=True)
    out_v = np.asarray(out_v, dtype=np.float64)
    out_t = np.asarray(out_t, dtype=np.int32)
    sim_f = boundary_faces(out_t)
    centers = out_v[sim_f].mean(axis=1)
    visible_mesh = trimesh.Trimesh(vertices, old_visible, process=False)
    full_mesh = trimesh.Trimesh(vertices, faces, process=False)
    _, visible_distance, _ = trimesh.proximity.closest_point(
        visible_mesh, centers)
    _, full_distance, _ = trimesh.proximity.closest_point(full_mesh, centers)
    visible_f = sim_f[visible_distance <= full_distance + 1e-7]

    boundary = np.unique(sim_f)
    boundary_tree = cKDTree(out_v[boundary])
    origin_old = np.asarray(data["origin_attachment_vertices"], dtype=np.int32)
    origin = np.unique(boundary[boundary_tree.query(vertices[origin_old])[1]])
    # Refinement must improve the transition, not turn the whole refined patch
    # into a rigid cap. Keep the same compact Ring-0 samples by transferring
    # each old insertion anchor to its nearest new boundary vertex.
    insertion = np.unique(boundary[
        boundary_tree.query(vertices[old_insertion])[1]])
    insertion = np.setdiff1d(insertion, origin).astype(np.int32)
    fixed = {
        **{int(i): ("L_Femur0", "origin") for i in origin},
        **{int(i): ("L_Patella0", "insertion") for i in insertion},
    }
    result = dict(data)
    result.update({
        "vertices": out_v.astype(np.float32), "tetrahedra": out_t,
        "faces": visible_f, "render_faces": visible_f,
        "sim_faces": sim_f, "collision_faces": visible_f,
        "origin_attachment_vertices": origin.astype(np.int32),
        "insertion_attachment_vertices": insertion,
        "anchor_vertices": np.r_[origin, insertion].astype(np.int32),
        "fixed_verts_with_bones": fixed,
        "anchor_bone_map": {i: bone for i, (bone, _) in fixed.items()},
        "cap_vertex_types": {i: kind for i, (_, kind) in fixed.items()},
        "cap_attachments": np.asarray(
            [[int(i), 0, 0, 0, 0] for i in origin]
            + [[int(i), 0, 1, 0, 0] for i in insertion], dtype=np.int32),
        "tetrahedralization_method": "tetgen_local_insertion_refine",
        "tet_quality_stats": quality_statistics(out_v, out_t),
        "tet_quality_profile": f"local_insertion_r{args.radius:g}",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(result, stream)
    print(f"local refined tet: {len(out_v)} vertices, {len(out_t)} tets")
    print(f"attachments: origin={len(origin)} insertion={len(insertion)}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
