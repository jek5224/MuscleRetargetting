"""Discover loadable left upper-leg MVP assets without publication gates."""
import argparse
import pickle
from pathlib import Path

import numpy as np

from muscle_sim.mvp.core import (
    boundary_faces, orient_tets, pathological_tets, signed_tet_volumes)
from muscle_sim.mvp.io import sha256, write_json


def inspect_muscle(path):
    with path.open("rb") as stream:
        data = pickle.load(stream)
    vertices = np.asarray(data["vertices"], dtype=float)
    tets = np.asarray(data["tetrahedra"], dtype=np.int32)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (n,4)")
    if len(tets) == 0 or np.min(tets) < 0 or np.max(tets) >= len(vertices):
        raise ValueError("unusable tetrahedron indexing")
    tets = orient_tets(vertices, tets)
    volumes = signed_tet_volumes(vertices, tets)
    poor, quality = pathological_tets(vertices, tets)
    surface = boundary_faces(tets)
    cap_faces = np.asarray(data.get("cap_face_indices", []), dtype=int)
    sim_faces = np.asarray(data.get("sim_faces", data.get("faces", [])))
    attachment = set(map(int, data.get("anchor_vertices", [])))
    for face_id in cap_faces:
        if 0 <= face_id < len(sim_faces):
            attachment.update(map(int, sim_faces[face_id]))
    names = data.get("attach_skeleton_names", [])
    flat_names = [str(name) for stream in names for name in stream]
    architecture = data.get("fiber_architecture", [])
    def fiber_points(row):
        if isinstance(row, dict):
            return row.get("points", [])
        return row[2] if len(row) > 2 else []
    sample_count = sum(len(fiber_points(row)) for row in architecture)
    return {
        "muscle_name": path.name.replace("_tet.npz", ""),
        "tet_mesh_path": str(path),
        "tet_mesh_sha256": sha256(path),
        "vertex_count": int(len(vertices)),
        "tet_count": int(len(tets)),
        "surface_triangle_count": int(len(surface)),
        "attachment_vertex_count": int(len(attachment)),
        "origin_bone": flat_names[0] if flat_names else None,
        "insertion_bone": flat_names[-1] if flat_names else None,
        "attachment_bones": sorted(set(flat_names)),
        "available_fiber_data": bool(len(architecture)),
        "fiber_sample_count": int(sample_count),
        "mesh_quality_summary": {
            "positive_tet_fraction": float(np.mean(volumes > 0.)),
            "minimum_signed_volume": float(np.min(volumes)),
            "regularized_tet_count": int(len(poor)),
            "minimum_relative_volume_quality": float(np.min(quality))},
        "usable_for_mvp": bool(len(surface) > 0 and np.mean(volumes > 0.) > .5)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle-root", required=True)
    parser.add_argument("--skeleton", required=True)
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--bone-root",
                        default="Zygote_Meshes_251229/Skeleton")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    muscles = []
    errors = []
    for path in sorted(Path(args.muscle_root).glob("L_*_tet.npz")):
        try:
            row = inspect_muscle(path)
            if row["usable_for_mvp"]:
                muscles.append(row)
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
    bones = sorted(str(path) for path in Path(args.bone_root).glob("L_*.obj"))
    manifest = {
        "branch": "muscle_animation_mvp",
        "skeleton_path": args.skeleton,
        "skeleton_sha256": sha256(args.skeleton),
        "bvh_path": args.bvh,
        "bvh_sha256": sha256(args.bvh),
        "bone_root": args.bone_root,
        "bone_meshes": bones,
        "muscles": muscles,
        "discovery_errors": errors}
    output = Path(args.output)
    write_json(output / "mvp_asset_manifest.json", manifest)
    print(output / "mvp_asset_manifest.json")


if __name__ == "__main__":
    main()
