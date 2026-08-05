#!/usr/bin/env python3
"""TetWild upper-leg cage: muscle envelope minus repaired skeleton union."""
import argparse
import json
import os
import pickle

import numpy as np
import trimesh
import wildmeshing

from tools.build_upperleg_bone_conforming_cage import (
    BONES, crop_tibia, embed_points, repaired_mesh)


def load_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscles", default=".muscles_L_UpLeg.json")
    parser.add_argument("--tet-dir", default="tet")
    parser.add_argument(
        "--output", default="cage/L_UpLeg_tetwild_conforming_cage.npz")
    parser.add_argument("--bone-pitch-mm", type=float, default=3.0)
    parser.add_argument("--clearance-mm", type=float, default=8.0)
    parser.add_argument("--edge-length-r", type=float, default=0.015)
    parser.add_argument("--epsilon", type=float, default=0.0005)
    args = parser.parse_args()

    with open(args.muscles) as stream:
        names = [entry["name"] for entry in json.load(stream)]
    muscle = {}
    for name in names:
        data = load_pickle(os.path.join(args.tet_dir, name + "_tet.npz"))
        muscle[name] = np.asarray(data["vertices"], dtype=np.float64)
    all_muscle = np.vstack(list(muscle.values()))

    lower_y = float(all_muscle[:, 1].min() - 0.008)
    repaired_bones = []
    for name in BONES:
        mesh = trimesh.load(
            f"Zygote_Meshes_251229/Skeleton/{name}.obj", process=False)
        mesh.vertices *= 0.01
        if name == "L_Tibia_Fibula":
            mesh = crop_tibia(mesh, lower_y)
        repaired_bones.append(repaired_mesh(mesh))
    pitch = args.bone_pitch_mm / 1000.0
    bone_points = np.vstack([
        mesh.voxelized(pitch).fill().points for mesh in repaired_bones])
    skeleton = repaired_mesh(
        trimesh.voxel.ops.points_to_marching_cubes(
            bone_points, pitch=pitch))

    outer = trimesh.convex.convex_hull(
        np.vstack((all_muscle, skeleton.vertices)))
    center = outer.vertices.mean(axis=0)
    radial = outer.vertices - center
    radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
    outer.vertices += args.clearance_mm / 1000.0 * radial
    outer = repaired_mesh(outer)

    # Opposite orientation makes generalized winding number represent
    # exterior-minus-skeleton without requiring a fragile surface boolean.
    vertices = np.vstack((outer.vertices, skeleton.vertices)).astype(np.float64)
    faces = np.vstack((
        outer.faces,
        skeleton.faces[:, ::-1] + len(outer.vertices))).astype(np.int32)
    tetrahedralizer = wildmeshing.Tetrahedralizer(
        stop_quality=10.0, max_its=80, stage=2,
        epsilon=args.epsilon, edge_length_r=args.edge_length_r,
        skip_simplify=False, coarsen=True)
    tetrahedralizer.set_log_level(3)
    tetrahedralizer.set_mesh(vertices, faces)
    tetrahedralizer.tetrahedralize()
    output = tetrahedralizer.get_tet_mesh(
        use_input_for_wn=True, manifold_surface=True,
        correct_surface_orientation=False)
    cage_vertices = np.asarray(output[0], dtype=np.float64)
    tetrahedra = np.asarray(output[1], dtype=np.int32)
    print("TetWild raw output shapes:",
          [np.asarray(item).shape for item in output])

    q = cage_vertices[tetrahedra]
    dm = np.stack((
        q[:, 0] - q[:, 3], q[:, 1] - q[:, 3],
        q[:, 2] - q[:, 3]), axis=2)
    determinant = np.linalg.det(dm)
    negative = determinant < 0
    tetrahedra[negative, 0], tetrahedra[negative, 1] = (
        tetrahedra[negative, 1].copy(),
        tetrahedra[negative, 0].copy())
    singular = np.linalg.svd(dm, compute_uv=False)
    condition = singular[:, 0] / np.maximum(singular[:, 2], 1e-30)

    embeddings = {}
    unembedded = 0
    for name, points in muscle.items():
        reference = points.copy()
        inside = skeleton.contains(reference)
        if np.any(inside):
            closest, _, face_id = trimesh.proximity.closest_point(
                skeleton, reference[inside])
            reference[inside] = (
                closest + 0.001 * skeleton.face_normals[face_id])
        tet_index, weights, violation = embed_points(
            reference, cage_vertices, tetrahedra, candidates=256)
        bad = (tet_index < 0) | (violation > 1e-5)
        unembedded += int(np.sum(bad))
        embeddings[name] = {
            "tet_index": tet_index,
            "weights": weights.astype(np.float32),
            "vertex_count": len(points),
            "projected_from_bone": inside,
        }
        print(f"{name}: unembedded={int(np.sum(bad))}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as stream:
        pickle.dump({
            "vertices": cage_vertices.astype(np.float32),
            "tetrahedra": tetrahedra,
            "muscles": embeddings,
            "muscle_names": names,
            "source_tet_dir": args.tet_dir,
            "bone_conforming": True,
            "skeleton_surface_vertices": np.asarray(
                skeleton.vertices, dtype=np.float32),
            "skeleton_surface_faces": np.asarray(
                skeleton.faces, dtype=np.int32),
        }, stream)
    print(f"TetWild cage: {len(cage_vertices)} vertices, "
          f"{len(tetrahedra)} tets, unembedded={unembedded}")
    print("condition percentiles:",
          np.percentile(condition, [50, 90, 95, 99, 100]))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
