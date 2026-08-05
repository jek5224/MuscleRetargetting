#!/usr/bin/env python3
"""Gmsh bone-conforming cage with skeleton union as an explicit volume hole."""
import argparse
import json
import os
import pickle

import gmsh
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from tools.build_upperleg_bone_conforming_cage import (
    BONES, crop_tibia, repaired_mesh)


def load_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def add_discrete_surface(tag, mesh, node_offset):
    node_tags = np.arange(
        node_offset + 1, node_offset + 1 + len(mesh.vertices),
        dtype=np.int64)
    gmsh.model.addDiscreteEntity(2, tag)
    gmsh.model.mesh.addNodes(
        2, tag, node_tags.tolist(),
        np.asarray(mesh.vertices, dtype=np.float64).ravel().tolist())
    triangle_tags = np.arange(
        1, len(mesh.faces) + 1, dtype=np.int64)
    gmsh.model.mesh.addElementsByType(
        tag, 2, triangle_tags.tolist(),
        node_tags[np.asarray(mesh.faces, dtype=np.int32)].ravel().tolist())
    return int(node_tags[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscles", default=".muscles_L_UpLeg.json")
    parser.add_argument("--tet-dir", default="tet")
    parser.add_argument(
        "--output", default="cage/L_UpLeg_gmsh_conforming_cage.npz")
    parser.add_argument("--bone-pitch-mm", type=float, default=3.0)
    parser.add_argument("--clearance-mm", type=float, default=8.0)
    parser.add_argument("--size-mm", type=float, default=8.0)
    args = parser.parse_args()

    with open(args.muscles) as stream:
        names = [entry["name"] for entry in json.load(stream)]
    muscle = {}
    for name in names:
        data = load_pickle(os.path.join(args.tet_dir, name + "_tet.npz"))
        muscle[name] = np.asarray(data["vertices"], dtype=np.float64)
    all_muscle = np.vstack(list(muscle.values()))

    bone_meshes = []
    lower_y = float(all_muscle[:, 1].min() - 0.008)
    for name in BONES:
        mesh = trimesh.load(
            f"Zygote_Meshes_251229/Skeleton/{name}.obj", process=False)
        mesh.vertices *= 0.01
        if name == "L_Tibia_Fibula":
            mesh = crop_tibia(mesh, lower_y)
        bone_meshes.append(repaired_mesh(mesh))
    pitch = args.bone_pitch_mm / 1000.0
    points = np.vstack([
        mesh.voxelized(pitch).fill().points for mesh in bone_meshes])
    skeleton = repaired_mesh(
        trimesh.voxel.ops.points_to_marching_cubes(points, pitch=pitch))
    skeleton = skeleton.simplify_quadric_decimation(face_count=3000)
    trimesh.repair.fix_normals(skeleton, multibody=True)

    hull = trimesh.convex.convex_hull(
        np.vstack((all_muscle, skeleton.vertices)))
    center = hull.vertices.mean(axis=0)
    radial = hull.vertices - center
    radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
    hull.vertices += args.clearance_mm / 1000.0 * radial
    outer = repaired_mesh(hull)
    outer = outer.simplify_quadric_decimation(face_count=500)
    trimesh.repair.fix_normals(outer, multibody=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("L_UpLeg_bone_conforming")
        last_node = add_discrete_surface(1, outer, 0)
        add_discrete_surface(2, skeleton, last_node)
        gmsh.model.mesh.classifySurfaces(
            40.0 * np.pi / 180.0,
            boundary=True,
            forReparametrization=False,
            curveAngle=180.0 * np.pi / 180.0)
        gmsh.model.mesh.createGeometry()
        outer_tree = cKDTree(np.asarray(outer.vertices))
        bone_tree = cKDTree(np.asarray(skeleton.vertices))
        outer_surfaces = []
        bone_surfaces = []
        for _, surface_tag in gmsh.model.getEntities(2):
            _, coordinates, _ = gmsh.model.mesh.getNodes(
                2, surface_tag, includeBoundary=True)
            surface_points = np.asarray(
                coordinates, dtype=np.float64).reshape(-1, 3)
            outer_distance = np.median(
                outer_tree.query(surface_points)[0])
            bone_distance = np.median(
                bone_tree.query(surface_points)[0])
            if outer_distance < bone_distance:
                outer_surfaces.append(surface_tag)
            else:
                bone_surfaces.append(surface_tag)
        print(f"surface patches: outer={len(outer_surfaces)}, "
              f"bone={len(bone_surfaces)}")
        outer_loop = gmsh.model.geo.addSurfaceLoop(outer_surfaces)
        bone_loop = gmsh.model.geo.addSurfaceLoop(bone_surfaces)
        volume = gmsh.model.geo.addVolume([outer_loop, bone_loop])
        gmsh.model.geo.synchronize()
        size = args.size_mm / 1000.0
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.5 * size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        vertices = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        tag_to_index = {
            int(tag): i for i, tag in enumerate(node_tags)}
        element_types, _, element_nodes = (
            gmsh.model.mesh.getElements(3, volume))
        tetrahedra = []
        for element_type, nodes in zip(element_types, element_nodes):
            _, _, _, count, _, _ = gmsh.model.mesh.getElementProperties(
                element_type)
            if count != 4:
                continue
            raw = np.asarray(nodes, dtype=np.int64).reshape(-1, 4)
            tetrahedra.append(np.vectorize(tag_to_index.__getitem__)(raw))
        tetrahedra = np.vstack(tetrahedra).astype(np.int32)
    finally:
        gmsh.finalize()

    q = vertices[tetrahedra]
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
    print(f"Gmsh cage: {len(vertices)} vertices, {len(tetrahedra)} tets")
    print("condition percentiles:",
          np.percentile(condition, [50, 90, 95, 99, 100]))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as stream:
        pickle.dump({
            "vertices": vertices.astype(np.float32),
            "tetrahedra": tetrahedra,
            "muscle_names": names,
            "source_tet_dir": args.tet_dir,
            "bone_conforming": True,
            "skeleton_surface_vertices": np.asarray(
                skeleton.vertices, dtype=np.float32),
            "skeleton_surface_faces": np.asarray(
                skeleton.faces, dtype=np.int32),
            "condition_percentiles": np.percentile(
                condition, [50, 90, 95, 99, 100]),
        }, stream)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
