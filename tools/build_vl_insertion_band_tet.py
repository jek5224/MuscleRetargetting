#!/usr/bin/env python3
"""Add a compact insertion band to the clean free-remeshed VL tet."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from tools.bake_surface_fast import load_tet
from tools.tet_from_original_obj import _find_boundary_loops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band", type=float, default=0.0015)
    args = parser.parse_args()

    data = load_tet(args.input)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    surface_vertices = np.unique(np.asarray(data["sim_faces"], dtype=np.int32))
    source = trimesh.load(
        data["source_obj"], process=False, maintain_order=True)
    source_vertices = np.asarray(source.vertices, dtype=np.float64) * 0.01
    loops = _find_boundary_loops(
        source_vertices, np.asarray(source.faces, dtype=np.int32))
    loops = sorted(
        loops, key=lambda loop: source_vertices[loop, 1].mean(), reverse=True)
    insertion_source = source_vertices[np.asarray(loops[1], dtype=np.int32)]
    distance = cKDTree(insertion_source).query(vertices[surface_vertices])[0]
    insertion = np.asarray(
        surface_vertices[distance <= args.band], dtype=np.int32)
    origin = np.asarray(data["origin_attachment_vertices"], dtype=np.int32)
    insertion = np.setdiff1d(insertion, origin).astype(np.int32)

    fixed = {
        **{int(i): ("L_Femur0", "origin") for i in origin},
        **{int(i): ("L_Patella0", "insertion") for i in insertion},
    }
    data["origin_attachment_vertices"] = origin
    data["insertion_attachment_vertices"] = insertion
    data["anchor_vertices"] = np.r_[origin, insertion].astype(np.int32)
    data["fixed_verts_with_bones"] = fixed
    data["anchor_bone_map"] = {i: bone for i, (bone, _) in fixed.items()}
    data["cap_vertex_types"] = {i: kind for i, (_, kind) in fixed.items()}
    data["cap_attachments"] = np.asarray(
        [[int(i), 0, 0, 0, 0] for i in origin]
        + [[int(i), 0, 1, 0, 0] for i in insertion], dtype=np.int32)
    data["tet_quality_profile"] = (
        f"{data.get('tet_quality_profile', 'tetwild')}_insertion_band_"
        f"{args.band * 1000:g}mm")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(data, stream)
    print(f"Saved {args.output}: origin={len(origin)}, insertion={len(insertion)}")


if __name__ == "__main__":
    main()
