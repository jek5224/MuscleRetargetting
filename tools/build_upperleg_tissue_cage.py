#!/usr/bin/env python3
"""Build a bone-excluding voxel-tet tissue cage for all L upper-leg contours."""
import argparse
import json
import os
import pickle

import numpy as np
import trimesh
from scipy import ndimage


CUBE_TETS = np.asarray([
    [0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
    [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7],
], dtype=np.int32)
CORNERS = np.asarray([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=np.int32)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def voxel_points(mesh, pitch):
    return np.asarray(mesh.voxelized(pitch=pitch).fill().points)


def rasterize_tet_volume(vertices, tetrahedra, origin, pitch, shape):
    """Conservatively mark grid cells intersecting the source tet volume.

    Surface voxelization alone can leave narrow/concave muscles hollow, and
    subtracting the bone cavity can then remove genuine attachment tissue.
    Testing cell centres plus all tet vertices/centroids gives a conservative
    source-volume mask without filling the empty convex hull between muscles.
    """
    grid = np.zeros(shape, dtype=bool)
    shape_a = np.asarray(shape, dtype=np.int32)
    for ids in tetrahedra:
        q = vertices[ids]
        lo = np.floor((q.min(axis=0) - origin) / pitch).astype(np.int32) - 1
        hi = np.floor((q.max(axis=0) - origin) / pitch).astype(np.int32) + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, shape_a - 1)
        ijk = np.stack(np.meshgrid(
            np.arange(lo[0], hi[0] + 1),
            np.arange(lo[1], hi[1] + 1),
            np.arange(lo[2], hi[2] + 1), indexing="ij"), axis=-1).reshape(-1, 3)
        centres = origin + (ijk + 0.5) * pitch
        matrix = np.stack(
            (q[0] - q[3], q[1] - q[3], q[2] - q[3]), axis=1)
        try:
            first = np.linalg.solve(
                matrix, (centres - q[3]).T).T
        except np.linalg.LinAlgError:
            continue
        weights = np.c_[first, 1.0 - first.sum(axis=1)]
        if np.any(np.all(weights >= -1e-8, axis=1)):
            selected = ijk[np.all(weights >= -1e-8, axis=1)]
            grid[tuple(selected.T)] = True
        # Protect the cells containing the full tet's defining samples. This
        # also covers sub-pitch tets for which no grid-cell centre is inside.
        # A half-voxel barycentric lattice catches cells crossed by thin and
        # slanted tets without adding the large empty boxes produced by a
        # per-tet AABB raster.
        edge_length = max(
            np.linalg.norm(q[i] - q[j])
            for i in range(4) for j in range(i + 1, 4))
        order = int(np.clip(np.ceil(2.0 * edge_length / pitch), 4, 12))
        sample_list = []
        for a in range(order + 1):
            for b in range(order + 1 - a):
                for c in range(order + 1 - a - b):
                    d = order - a - b - c
                    sample_list.append(
                        (a * q[0] + b * q[1] + c * q[2] + d * q[3])
                        / order)
        if order != 6:
            for a in range(7):
                for b in range(7 - a):
                    for c in range(7 - a - b):
                        d = 6 - a - b - c
                        sample_list.append(
                            (a * q[0] + b * q[1] + c * q[2] + d * q[3])
                            / 6.0)
        samples = np.asarray(sample_list)
        sample_index = np.floor((samples - origin) / pitch).astype(np.int32)
        sample_index = np.clip(sample_index, 0, shape_a - 1)
        grid[tuple(sample_index.T)] = True
    return grid


def embedding(points, origin, pitch, occupied, cube_to_tets, vertices, tets):
    cell = np.floor((points - origin) / pitch).astype(np.int32)
    cell = np.clip(cell, 0, np.asarray(occupied.shape) - 1)
    tet_id = np.full(len(points), -1, dtype=np.int32)
    weights = np.zeros((len(points), 4), dtype=np.float64)
    violation = np.full(len(points), np.inf)
    for point_i, (point, key) in enumerate(zip(points, map(tuple, cell))):
        candidates = cube_to_tets.get(key, [])
        for ti in candidates:
            q = vertices[tets[ti]]
            matrix = np.stack(
                (q[0] - q[3], q[1] - q[3], q[2] - q[3]), axis=1)
            try:
                first = np.linalg.solve(matrix, point - q[3])
            except np.linalg.LinAlgError:
                continue
            w = np.r_[first, 1.0 - first.sum()]
            v = max(float(-w.min()), float(w.max() - 1.0))
            if v < violation[point_i]:
                violation[point_i] = v
                tet_id[point_i] = ti
                weights[point_i] = w
    return tet_id, weights, violation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muscles", default=".muscles_L_UpLeg.json")
    ap.add_argument("--tet-dir", default="tet")
    ap.add_argument("--output", default="cage/L_UpLeg_contour_tissue_cage.npz")
    ap.add_argument("--pitch-mm", type=float, default=12.0)
    ap.add_argument("--tissue-rings", type=int, default=1)
    args = ap.parse_args()
    pitch = args.pitch_mm / 1000.0

    with open(args.muscles) as f:
        names = [entry["name"] for entry in json.load(f)]
    muscle = {}
    muscle_tets = {}
    muscle_voxels = []
    all_points = []
    for name in names:
        data = load_pickle(os.path.join(
            args.tet_dir, f"{name}_tet.npz"))
        v = np.asarray(data["vertices"], dtype=np.float64)
        tet = np.asarray(data["tetrahedra"], dtype=np.int32)
        f = np.asarray(data.get("sim_faces", data["render_faces"]),
                       dtype=np.int32)
        mesh = trimesh.Trimesh(v, f, process=False)
        muscle[name] = v
        muscle_tets[name] = tet
        all_points.append(v)
        muscle_voxels.append(voxel_points(mesh, pitch))

    bone_voxels = []
    bone_root = "Zygote_Meshes_251229/Skeleton"
    for bone in ("L_Os_Coxae", "L_Femur", "L_Patella",
                 "L_Tibia_Fibula"):
        mesh = trimesh.load(
            os.path.join(bone_root, bone + ".obj"), process=False)
        mesh.vertices *= 0.01
        bone_voxels.append(voxel_points(mesh, pitch))

    combined = np.vstack([*muscle_voxels, *bone_voxels, *all_points])
    origin = np.floor(combined.min(axis=0) / pitch) * pitch - 2 * pitch
    upper = np.ceil(combined.max(axis=0) / pitch) * pitch + 2 * pitch
    shape = np.ceil((upper - origin) / pitch).astype(np.int32) + 1

    def rasterize(point_sets):
        grid = np.zeros(shape, dtype=bool)
        for points in point_sets:
            index = np.rint((points - origin) / pitch).astype(np.int32)
            index = np.clip(index, 0, shape - 1)
            grid[tuple(index.T)] = True
        return grid

    tissue = rasterize(muscle_voxels)
    tissue = ndimage.binary_dilation(
        tissue, iterations=args.tissue_rings)
    tissue = ndimage.binary_closing(tissue, iterations=1)
    bones = rasterize(bone_voxels)
    # Remove only confidently interior bone voxels. The one-voxel boundary
    # layer remains available for anatomical attachment constraints.
    bone_interior = ndimage.binary_erosion(bones, iterations=1)
    occupied = tissue & ~bone_interior

    # Source volume wins over the generic bone cavity. This does not introduce
    # a muscle-specific patch: it uniformly protects only cells demonstrably
    # occupied by the saved anatomical tets.
    protected_source = np.zeros(shape, dtype=bool)
    for name, points in muscle.items():
        protected_source |= rasterize_tet_volume(
            points, muscle_tets[name], origin, pitch, shape)
    occupied |= protected_source

    cells = np.argwhere(occupied)
    vertex_map = {}
    vertices = []
    tets = []
    cube_to_tets = {}
    for cell in cells:
        cube_vertex = []
        for delta in CORNERS:
            key = tuple((cell + delta).tolist())
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append(origin + np.asarray(key) * pitch)
            cube_vertex.append(vertex_map[key])
        first_tet = len(tets)
        cube_vertex = np.asarray(cube_vertex, dtype=np.int32)
        tets.extend(cube_vertex[CUBE_TETS].tolist())
        cube_to_tets[tuple(cell.tolist())] = list(
            range(first_tet, first_tet + len(CUBE_TETS)))
    vertices = np.asarray(vertices, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int32)

    embeddings = {}
    outside = 0
    for name, points in muscle.items():
        ti, w, violation = embedding(
            points, origin, pitch, occupied, cube_to_tets, vertices, tets)
        bad = int(np.sum((ti < 0) | (violation > 1e-5)))
        outside += bad
        embeddings[name] = {
            "tet_index": ti,
            "weights": w.astype(np.float32),
            "vertex_count": len(points),
        }
        print(f"{name}: outside={bad}, worst="
              f"{np.max(violation[np.isfinite(violation)]):.2e}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "vertices": vertices.astype(np.float32),
            "tetrahedra": tets,
            "muscles": embeddings,
            "muscle_names": names,
            "source_tet_dir": args.tet_dir,
            "pitch_mm": args.pitch_mm,
            "occupied_cells": len(cells),
            "bone_interior_voxels_removed": int(
                np.sum(tissue & bone_interior)),
            "protected_source_cells": int(np.sum(protected_source)),
        }, f)
    print(f"Tissue cage: {len(vertices)} vertices, {len(tets)} tets, "
          f"{len(cells)} cells, outside={outside}, "
          f"removed bone voxels={np.sum(tissue & bone_interior)}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
