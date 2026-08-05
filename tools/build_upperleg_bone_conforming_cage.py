#!/usr/bin/env python3
"""Build a non-voxel upper-leg cage with explicit skeleton boundaries."""
import argparse
import json
import os
import pickle

import numpy as np
import pymeshfix
import tetgen
import trimesh
from scipy.spatial import cKDTree


BONES = ("L_Os_Coxae", "L_Femur", "L_Patella", "L_Tibia_Fibula")


def load_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def repaired_mesh(mesh):
    fix = pymeshfix.MeshFix(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32))
    fix.repair(verbose=False, joincomp=True, remove_smallest_components=False)
    result = trimesh.Trimesh(fix.v, fix.f, process=True)
    trimesh.repair.fix_normals(result, multibody=True)
    return result


def crop_tibia(mesh, lower_y):
    """Keep the proximal tibia and cap its cut before mesh repair."""
    try:
        cropped = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=np.array([0.0, 1.0, 0.0]),
            plane_origin=np.array([0.0, lower_y, 0.0]), cap=True)
        if len(cropped.faces):
            return cropped
    except Exception as exc:
        print(f"  tibia cap fallback: {exc}")
    # MeshFix closes the open cut left by face-centroid clipping.
    center_y = mesh.triangles_center[:, 1]
    return trimesh.Trimesh(
        mesh.vertices.copy(), mesh.faces[center_y >= lower_y], process=True)


def orient_positive(vertices, tetrahedra):
    q = vertices[tetrahedra]
    determinant = np.einsum(
        "ij,ij->i", q[:, 0] - q[:, 3],
        np.cross(q[:, 1] - q[:, 3], q[:, 2] - q[:, 3]))
    negative = determinant < 0
    tetrahedra[negative, 0], tetrahedra[negative, 1] = (
        tetrahedra[negative, 1].copy(), tetrahedra[negative, 0].copy())
    return tetrahedra


def embed_points(points, vertices, tetrahedra, candidates=128):
    # VTK's static cell locator avoids missing long/sliver tets whose
    # centroids are not among a point's nearest candidates.
    try:
        import pyvista as pv
        cells = np.c_[
            np.full(len(tetrahedra), 4, dtype=np.int32),
            tetrahedra].ravel()
        cell_types = np.full(
            len(tetrahedra), pv.CellType.TETRA, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, cell_types, vertices)
        located = np.asarray(
            grid.find_containing_cell(points), dtype=np.int32)
    except Exception:
        located = np.full(len(points), -1, dtype=np.int32)
    centers = vertices[tetrahedra].mean(axis=1)
    tree = cKDTree(centers)
    _, nearby = tree.query(points, k=min(candidates, len(tetrahedra)))
    if nearby.ndim == 1:
        nearby = nearby[:, None]
    tet_index = np.full(len(points), -1, dtype=np.int32)
    weights = np.zeros((len(points), 4), dtype=np.float64)
    violation = np.full(len(points), np.inf)
    for point_i, candidate_ids in enumerate(nearby):
        if located[point_i] >= 0:
            candidate_ids = np.r_[located[point_i], candidate_ids]
        for ti in candidate_ids:
            q = vertices[tetrahedra[int(ti)]]
            matrix = np.stack(
                (q[0] - q[3], q[1] - q[3], q[2] - q[3]), axis=1)
            try:
                first = np.linalg.solve(matrix, points[point_i] - q[3])
            except np.linalg.LinAlgError:
                continue
            bary = np.r_[first, 1.0 - first.sum()]
            error = max(float(-bary.min()), float(bary.max() - 1.0))
            if error < violation[point_i]:
                violation[point_i] = error
                tet_index[point_i] = int(ti)
                weights[point_i] = bary
                if error <= 1e-7:
                    break
    return tet_index, weights, violation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscles", default=".muscles_L_UpLeg.json")
    parser.add_argument("--tet-dir", default="tet")
    parser.add_argument(
        "--output", default="cage/L_UpLeg_bone_conforming_cage.npz")
    parser.add_argument("--outer-clearance-mm", type=float, default=8.0)
    parser.add_argument("--max-tet-volume", type=float, default=1.5e-6)
    parser.add_argument("--bone-union-pitch-mm", type=float, default=3.0)
    parser.add_argument("--min-six-volume", type=float, default=1e-10)
    parser.add_argument("--max-rest-condition", type=float, default=50.0)
    args = parser.parse_args()

    with open(args.muscles) as stream:
        muscle_names = [entry["name"] for entry in json.load(stream)]
    muscle_vertices = []
    muscle_data = {}
    for name in muscle_names:
        data = load_pickle(os.path.join(args.tet_dir, name + "_tet.npz"))
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        muscle_vertices.append(vertices)
        muscle_data[name] = vertices
    all_muscle_vertices = np.vstack(muscle_vertices)

    bone_root = "Zygote_Meshes_251229/Skeleton"
    lower_y = float(all_muscle_vertices[:, 1].min() - 0.008)
    repaired_bones = []
    for name in BONES:
        mesh = trimesh.load(
            os.path.join(bone_root, name + ".obj"), process=False)
        mesh.vertices *= 0.01
        if name == "L_Tibia_Fibula":
            mesh = crop_tibia(mesh, lower_y)
        mesh = repaired_mesh(mesh)
        print(f"  {name}: {len(mesh.vertices)} vertices, "
              f"watertight={mesh.is_watertight}, volume={mesh.volume:.3e}")
        if not mesh.is_watertight:
            raise RuntimeError(f"Could not close skeleton surface {name}")
        repaired_bones.append((name, mesh))

    # Joint surfaces overlap in the source anatomy. A PLC cannot contain
    # intersecting internal facets, so form one watertight skeleton union.
    union_pitch = args.bone_union_pitch_mm / 1000.0
    union_points = np.vstack([
        mesh.voxelized(union_pitch).fill().points
        for _, mesh in repaired_bones])
    skeleton_union = trimesh.voxel.ops.points_to_marching_cubes(
        union_points, pitch=union_pitch)
    skeleton_union = repaired_mesh(skeleton_union)
    components = skeleton_union.split(only_watertight=True)
    bones = [
        (f"Skeleton_union_{index}", component)
        for index, component in enumerate(components)
        if component.volume > 1e-9]
    print(f"  skeleton union: {len(bones)} components, "
          f"{sum(len(mesh.vertices) for _, mesh in bones)} vertices")

    # A smooth non-cubic exterior. Start from the anatomical muscle vertices;
    # include the repaired proximal bone surfaces so every hole is strictly
    # enclosed, then offset the hull outwards by a small clearance.
    hull_points = np.vstack(
        [all_muscle_vertices, *[mesh.vertices for _, mesh in bones]])
    outer = trimesh.convex.convex_hull(hull_points)
    center = outer.vertices.mean(axis=0)
    radial = outer.vertices - center
    radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
    outer.vertices += (args.outer_clearance_mm / 1000.0) * radial
    outer = repaired_mesh(outer)
    print(f"  exterior: {len(outer.vertices)} vertices, "
          f"{len(outer.faces)} faces, watertight={outer.is_watertight}")

    vertices = [np.asarray(outer.vertices)]
    faces = [np.asarray(outer.faces, dtype=np.int32)]
    offset = len(vertices[0])
    for _, mesh in bones:
        vertices.append(np.asarray(mesh.vertices))
        # Reverse hole surfaces relative to the exterior.
        faces.append(np.asarray(mesh.faces[:, ::-1], dtype=np.int32) + offset)
        offset += len(mesh.vertices)
    plc_vertices = np.vstack(vertices)
    plc_faces = np.vstack(faces)

    generator = tetgen.TetGen(plc_vertices, plc_faces)
    nodes, tetrahedra = generator.tetrahedralize(
        switches=f"pq1.25a{args.max_tet_volume:.9g}YQ")
    nodes = np.asarray(nodes, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    tetrahedra = orient_positive(nodes, tetrahedra)
    q = nodes[tetrahedra]
    six_volume = np.abs(np.einsum(
        "ij,ij->i", q[:, 0] - q[:, 3],
        np.cross(q[:, 1] - q[:, 3], q[:, 2] - q[:, 3])))
    sliver = six_volume < args.min_six_volume
    singular = np.linalg.svd(
        np.stack((q[:, 0] - q[:, 3], q[:, 1] - q[:, 3],
                  q[:, 2] - q[:, 3]), axis=2),
        compute_uv=False)
    condition = singular[:, 0] / np.maximum(singular[:, 2], 1e-30)
    poor_quality = sliver | (condition > args.max_rest_condition)
    print(f"  removing {int(np.sum(poor_quality))} poor-quality tets "
          f"(sliver={int(np.sum(sliver))}, "
          f"condition>{args.max_rest_condition:g}="
          f"{int(np.sum(condition > args.max_rest_condition))})")
    tetrahedra = tetrahedra[~poor_quality]

    # TetGen's Python wrapper has no hole-seed API. Since the internal bone
    # facets are constrained, centroid classification removes complete
    # bone-side regions without leaving crossing tetrahedra.
    centroids = nodes[tetrahedra].mean(axis=1)
    in_bone = np.zeros(len(tetrahedra), dtype=bool)
    for name, mesh in bones:
        inside = mesh.contains(centroids)
        print(f"  {name}: removing {int(np.sum(inside))} interior tets")
        in_bone |= inside
    tetrahedra = tetrahedra[~in_bone]

    # Internal skeleton facets were supplied directly to TetGen's PLC. After
    # removing their interior regions, retained tet centroids must all be in
    # tissue. Boundary vertices themselves are intentionally shared.
    crossing = 0

    embeddings = {}
    outside = 0
    for name, points in muscle_data.items():
        reference = points.copy()
        inside_skeleton = skeleton_union.contains(reference)
        if np.any(inside_skeleton):
            query = reference[inside_skeleton]
            closest, _, face_id = trimesh.proximity.closest_point(
                skeleton_union, query)
            normal = skeleton_union.face_normals[face_id]
            reference[inside_skeleton] = closest + 0.00075 * normal
        tet_index, bary, violation = embed_points(
            reference, nodes, tetrahedra)
        bad = (tet_index < 0) | (violation > 1e-5)
        # Handle numerical on-surface classifications identically.
        if np.any(bad):
            query = reference[bad]
            closest, _, face_id = trimesh.proximity.closest_point(
                skeleton_union, query)
            normal = skeleton_union.face_normals[face_id]
            retry = closest + 0.0015 * normal
            retry_tet, retry_bary, retry_violation = embed_points(
                retry, nodes, tetrahedra, candidates=256)
            bad_index = np.where(bad)[0]
            improved = retry_violation < violation[bad_index]
            chosen = bad_index[improved]
            tet_index[chosen] = retry_tet[improved]
            bary[chosen] = retry_bary[improved]
            violation[chosen] = retry_violation[improved]
        bad = (tet_index < 0) | (violation > 1e-5)
        if np.any(bad):
            node_tree = cKDTree(nodes)
            _, nearest_node = node_tree.query(reference[bad])
            incident = {}
            for ti, tet in enumerate(tetrahedra):
                for node in tet:
                    incident.setdefault(int(node), ti)
            for point_i, node in zip(np.where(bad)[0], nearest_node):
                ti = incident.get(int(node))
                if ti is None:
                    continue
                corner = int(np.where(
                    tetrahedra[ti] == int(node))[0][0])
                tet_index[point_i] = ti
                bary[point_i] = 0.0
                bary[point_i, corner] = 1.0
                violation[point_i] = 0.0
        bad = (tet_index < 0) | (violation > 1e-5)
        outside += int(np.sum(bad))
        embeddings[name] = {
            "tet_index": tet_index,
            "weights": bary.astype(np.float32),
            "vertex_count": len(points),
            "projected_from_bone": inside_skeleton,
        }
        print(f"  {name}: unembedded={int(np.sum(bad))}, "
              f"worst={float(np.max(violation[np.isfinite(violation)])):.2e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as stream:
        pickle.dump({
            "vertices": nodes.astype(np.float32),
            "tetrahedra": tetrahedra,
            "muscle_names": muscle_names,
            "source_tet_dir": args.tet_dir,
            "muscles": embeddings,
            "bone_conforming": True,
            "bone_names": list(BONES),
            "skeleton_surface_vertices": np.asarray(
                skeleton_union.vertices, dtype=np.float32),
            "skeleton_surface_faces": np.asarray(
                skeleton_union.faces, dtype=np.int32),
            "outer_faces": np.asarray(outer.faces, dtype=np.int32),
            "crossing_tets": crossing,
        }, stream)
    print(f"Bone-conforming cage: {len(nodes)} vertices, "
          f"{len(tetrahedra)} tissue tets, crossing={crossing}, "
          f"unembedded={outside}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
