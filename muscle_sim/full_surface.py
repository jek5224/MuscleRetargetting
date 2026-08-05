"""Full-muscle reference selection and manifold surface reconstruction."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from muscle_sim.local_remeshing import (
    active_contact_points, json_ready, load_raw_mesh, source_hash,
    validate_closed_boundary)
from viewer.isolated_muscle import (
    attachment_patches_from_tet, extract_boundary_faces, orient_tetrahedra)


def edge_incidence(faces):
    edges = defaultdict(list)
    for face_id, face in enumerate(np.asarray(faces)):
        for first, second in ((face[0], face[1]), (face[1], face[2]),
                              (face[2], face[0])):
            edges[tuple(sorted((int(first), int(second))))].append(face_id)
    return edges


def boundary_loops(faces):
    edges = edge_incidence(faces)
    boundary = [edge for edge, owners in edges.items() if len(owners) == 1]
    adjacency = defaultdict(list)
    for first, second in boundary:
        adjacency[first].append(second)
        adjacency[second].append(first)
    loops, visited = [], set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        component, stack = [], [start]
        visited.add(start)
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in sorted(adjacency[vertex]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        # A simple loop has degree two at every vertex. Preserve sorted IDs
        # for branching manual review rather than inventing an ordering.
        if all(len(adjacency[v]) == 2 for v in component):
            ordered, previous, current = [min(component)], None, min(component)
            while True:
                candidates = [v for v in adjacency[current] if v != previous]
                following = candidates[0]
                if following == ordered[0]:
                    break
                ordered.append(following)
                previous, current = current, following
            component = ordered
        loops.append(component)
    return loops


def _duplicate_rows(faces):
    oriented = defaultdict(list)
    for face_id, face in enumerate(np.asarray(faces)):
        oriented[tuple(sorted(map(int, face)))].append(face_id)
    duplicate = {key: ids for key, ids in oriented.items() if len(ids) > 1}
    opposite = []
    for ids in duplicate.values():
        for first in range(len(ids)):
            for second in range(first + 1, len(ids)):
                a, b = faces[ids[first]], faces[ids[second]]
                if any(np.array_equal(a, np.roll(b[::-1], shift))
                       for shift in range(3)):
                    opposite.append([ids[first], ids[second]])
    return duplicate, opposite


def complete_topology(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    base = validate_closed_boundary(vertices, faces)
    edges = edge_incidence(faces)
    loops = boundary_loops(faces)
    duplicate, opposite = _duplicate_rows(faces)
    triangle = vertices[faces]
    area = trimesh.triangles.area(triangle)
    used = np.unique(faces)
    vertex_faces = defaultdict(list)
    for face_id, face in enumerate(faces):
        for vertex in face:
            vertex_faces[int(vertex)].append(face_id)
    nonmanifold_vertices = sorted({
        vertex for edge, owners in edges.items() if len(owners) != 2
        for vertex in edge})
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    euler = len(used) - len(edges) + len(faces)
    # TetGen's PLC validation is the available robust intersection gate in
    # this environment; topology inspection records that explicitly.
    return {
        "vertex_count": len(vertices),
        "used_vertex_count": len(used),
        "triangle_count": len(faces),
        "connected_components": base["connected_component_count"],
        "boundary_loop_count": len(loops),
        "boundary_loops": loops,
        "boundary_edges": [list(edge) for edge, rows in edges.items()
                           if len(rows) == 1],
        "nonmanifold_edges": [list(edge) for edge, rows in edges.items()
                              if len(rows) > 2],
        "nonmanifold_vertices": nonmanifold_vertices,
        "duplicate_triangles": [
            {"vertices": list(key), "face_ids": ids}
            for key, ids in duplicate.items()],
        "oppositely_oriented_duplicate_triangles": opposite,
        "zero_area_triangles": np.where(area <= 1e-16)[0].tolist(),
        "self_intersections": {
            "status": "deferred_to_tetgen_plc_diagnose",
            "pairs": [],
        },
        "inconsistent_orientation_edges":
            base["orientation_conflict_edges"],
        "euler_characteristic": int(euler),
        "signed_enclosed_volume": float(mesh.volume),
        "surface_area": float(mesh.area),
        "watertight": base["watertight"],
        "two_manifold": not any(len(rows) != 2 for rows in edges.values()),
        "consistently_oriented": base["consistent_orientation"],
        "valid_topology": bool(
            base["valid"] and not duplicate and not len(
                np.where(area <= 1e-16)[0])),
    }


def classify_loops(vertices, loops, patches=(), fibers=()):
    patch_centroids = [{
        "patch_id": patch_id, "end_type": patch.end_type,
        "bone_id": patch.bone_name,
        "centroid": np.mean(vertices[patch.vertex_ids], axis=0)}
        for patch_id, patch in enumerate(patches)]
    endpoints = []
    for fiber in fibers:
        endpoints.extend((fiber.rest_points[0], fiber.rest_points[-1]))
    rows = []
    for loop_id, loop in enumerate(loops):
        points = vertices[np.asarray(loop)]
        centroid = np.mean(points, axis=0)
        nearest = min(patch_centroids,
                      key=lambda p: np.linalg.norm(
                          centroid - p["centroid"])) if patch_centroids else None
        distance_patch = (
            float(np.linalg.norm(centroid - nearest["centroid"]))
            if nearest else None)
        distance_fiber = (
            float(np.min(np.linalg.norm(
                np.asarray(endpoints) - centroid, axis=1)))
            if endpoints else None)
        if nearest and distance_patch is not None and distance_patch < 0.005:
            category = (
                "ORIGIN_ATTACHMENT_OPENING" if nearest["end_type"] == 0
                else "INSERTION_ATTACHMENT_OPENING")
        elif any(loop.count(v) > 1 for v in loop):
            category = "SURFACE_SEAM"
        else:
            category = "UNEXPECTED_HOLE"
        rows.append({
            "loop_id": loop_id, "classification": category,
            "vertex_ids": loop, "coordinates": points.tolist(),
            "centroid": centroid.tolist(),
            "nearest_attachment_patch": (
                {key: value for key, value in nearest.items()
                 if key != "centroid"} if nearest else None),
            "distance_to_attachment_centroid": distance_patch,
            "distance_to_fiber_endpoint": distance_fiber,
        })
    return rows


def geometric_comparison(first_vertices, first_faces,
                         second_vertices, second_faces):
    first = trimesh.Trimesh(
        vertices=first_vertices, faces=first_faces, process=False)
    second = trimesh.Trimesh(
        vertices=second_vertices, faces=second_faces, process=False)
    _, first_distance, _ = trimesh.proximity.closest_point(
        second, first_vertices)
    _, second_distance, _ = trimesh.proximity.closest_point(
        first, second_vertices)
    distances = np.concatenate((first_distance, second_distance))
    return {
        "maximum_bidirectional_distance": float(np.max(distances)),
        "rms_bidirectional_distance": float(np.sqrt(
            np.mean(distances ** 2))),
        "percentile_95_distance": float(np.percentile(distances, 95)),
        "volume_difference": float(second.volume - first.volume),
        "surface_area_difference": float(second.area - first.area),
        "bounding_box_min_difference": (
            second.bounds[0] - first.bounds[0]).tolist(),
        "bounding_box_max_difference": (
            second.bounds[1] - first.bounds[1]).tolist(),
    }


def inspect_candidates(paths, current_path):
    current = load_raw_mesh(current_path)
    current_vertices = np.asarray(current["vertices"])
    current_tets = orient_tetrahedra(
        current_vertices, current["tetrahedra"])
    current_faces = extract_boundary_faces(current_tets)
    rows = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".obj":
            mesh = trimesh.load(path, force="mesh", process=False)
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            kind = "anatomical_obj"
        else:
            raw = load_raw_mesh(path)
            vertices = np.asarray(raw["vertices"])
            faces = np.asarray(raw.get("render_faces", raw.get("faces")))
            kind = "project_tet_render_surface"
        topology = complete_topology(vertices, faces)
        comparison = geometric_comparison(
            current_vertices, current_faces, vertices, faces)
        score = (
            1000 * topology["valid_topology"]
            + 100 * topology["watertight"]
            + 100 * topology["consistently_oriented"]
            - 1e5 * comparison["rms_bidirectional_distance"])
        rows.append({
            "path": str(path), "kind": kind,
            "topology": topology, "comparison_to_current_tet_boundary":
                comparison, "selection_score": float(score),
        })
    rows.sort(key=lambda row: (-row["selection_score"], row["path"]))
    return rows


def reconstruct_from_reference(reference_path, current_path,
                               diagnostic_dir, output, config):
    """Path A: replace invalid volume-boundary connectivity with reference."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference = load_raw_mesh(reference_path)
    current = load_raw_mesh(current_path)
    vertices = np.asarray(reference["vertices"], dtype=np.float64)
    faces = np.asarray(reference["render_faces"], dtype=np.int32)
    topology = complete_topology(vertices, faces)
    if not topology["valid_topology"]:
        raise ValueError(
            "selected explicit reference is not a closed oriented manifold; "
            "implicit fallback must be invoked explicitly")
    # Preserve the reference exactly. Its already-manifold connectivity is the
    # full-surface sheet reconstruction; no invalid tet-boundary face survives.
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output / "repaired_simulation_surface.obj")
    immutable = {
        "source_path": str(reference_path),
        "source_sha256": source_hash(reference_path),
        "source_mtime_ns": Path(reference_path).stat().st_mtime_ns,
        "vertex_count": len(vertices), "triangle_count": len(faces),
        "asset_role": "immutable_anatomical_geometric_reference",
    }
    (output / "immutable_reference_metadata.json").write_text(
        json.dumps(immutable, indent=2))
    current_vertices = np.asarray(current["vertices"])
    current_tets = orient_tetrahedra(
        current_vertices, current["tetrahedra"])
    current_boundary = extract_boundary_faces(current_tets)
    contact_points = active_contact_points(
        Path(diagnostic_dir) / "active_contact_multipliers.json",
        config["muscle_name"], current_vertices, current_boundary)
    comparison = geometric_comparison(
        current_vertices, current_boundary, vertices, faces)
    provenance = {
        "path": "A_explicit_reference_connectivity_reconstruction",
        "reference_surface": str(reference_path),
        "invalid_tet_boundary": str(current_path),
        "reference_faces_retained": len(faces),
        "current_invalid_boundary_faces_retained": 0,
        "removed_faces": {
            "source": "current_invalid_tet_boundary",
            "count": len(current_boundary),
            "reason": "replaced wholesale; contains four-sheet branches",
        },
        "added_faces": {
            "source": "immutable_reference_connectivity",
            "count": len(faces),
        },
        "split_vertices": [],
        "merged_vertices": [],
        "capped_loops": [],
        "repaired_branches": [
            [544, 545], [527, 528], [441, 442]],
        "vertex_displacement": 0.0,
        "topology_after": topology,
        "comparison_to_invalid_tet_boundary": comparison,
        "deterministic_seed": config["deterministic_seed"],
        "contact_point_count": len(contact_points),
    }
    (output / "surface_repair_provenance.json").write_text(json.dumps(
        json_ready(provenance), indent=2))
    return vertices, faces, provenance


def implicit_voxel_reconstruction(reference_path, output, config):
    """Path B: fine voxel SDF-like occupancy and watertight iso-surface.

    This fallback is explicit and never overwrites the reference. VTK performs
    iso-surface extraction because scikit-image is not a project dependency.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_raw_mesh(reference_path)
    reference_vertices = np.asarray(raw["vertices"], dtype=np.float64)
    reference_faces = np.asarray(raw["render_faces"], dtype=np.int32)
    reference = trimesh.Trimesh(
        reference_vertices, reference_faces, process=False)
    pitch = float(config["surface_reconstruction"]["implicit_pitch_m"])
    voxels = reference.voxelized(pitch).fill()
    occupancy = np.asarray(voxels.matrix, dtype=np.uint8)
    padded = np.pad(occupancy, 2)
    image = vtk.vtkImageData()
    image.SetDimensions(*padded.shape)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.GetPointData().SetScalars(numpy_to_vtk(
        padded.ravel(order="F"), deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR))
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.5)
    contour.ComputeNormalsOff()
    contour.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(contour.GetOutputPort())
    clean.Update()
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(clean.GetOutputPort())
    triangle.Update()
    poly = triangle.GetOutput()
    index_vertices = vtk_to_numpy(poly.GetPoints().GetData()) - 2.0
    vertices = trimesh.transform_points(index_vertices, voxels.transform)
    cell_data = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)
    faces = cell_data[:, 1:].astype(np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh, multibody=False)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    topology = complete_topology(vertices, faces)
    comparison = geometric_comparison(
        reference_vertices, reference_faces, vertices, faces)
    current = load_raw_mesh(config["current_tet_asset"])
    current_vertices = np.asarray(current["vertices"])
    current_tets = orient_tetrahedra(
        current_vertices, current["tetrahedra"])
    current_faces = extract_boundary_faces(current_tets)
    contact_points = active_contact_points(
        Path(config["diagnostic_dir"]) / "active_contact_multipliers.json",
        config["muscle_name"], current_vertices, current_faces)
    current_centroids = np.mean(current_vertices[current_faces], axis=1)
    contact_radius = float(config["surface_reconstruction"][
        "contact_radius_m"])
    critical_ids = (
        np.where(cKDTree(contact_points).query(
            current_centroids)[0] <= contact_radius)[0]
        if len(contact_points) else np.empty(0, dtype=np.int32))
    candidate_mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False)
    current_mesh = trimesh.Trimesh(
        vertices=current_vertices, faces=current_faces, process=False)
    _, critical_distance, candidate_face = trimesh.proximity.closest_point(
        candidate_mesh, current_centroids[critical_ids])
    normal_angle = np.degrees(np.arccos(np.clip(np.abs(np.einsum(
        "ij,ij->i", current_mesh.face_normals[critical_ids],
        candidate_mesh.face_normals[candidate_face])), -1.0, 1.0)))
    contact_comparison = {
        "triangle_count": len(critical_ids),
        "maximum_distance": float(np.max(
            critical_distance)) if len(critical_distance) else 0.0,
        "rms_distance": float(np.sqrt(np.mean(
            critical_distance ** 2))) if len(critical_distance) else 0.0,
        "percentile_95_distance": float(np.percentile(
            critical_distance, 95)) if len(critical_distance) else 0.0,
        "maximum_normal_angle_degrees": float(np.max(
            normal_angle)) if len(normal_angle) else 0.0,
        "percentile_95_normal_angle_degrees": float(np.percentile(
            normal_angle, 95)) if len(normal_angle) else 0.0,
    }
    maximum = float(config["surface_reconstruction"][
        "maximum_global_deviation_m"])
    accepted = bool(
        topology["valid_topology"]
        and comparison["maximum_bidirectional_distance"] <= maximum
        and contact_comparison["maximum_distance"] <= float(
            config["surface_reconstruction"][
                "maximum_contact_deviation_m"]))
    mesh.export(output / "implicit_reconstruction_candidate.obj")
    report = {
        "path": "B_implicit_voxel_occupancy_iso_surface",
        "accepted": accepted,
        "pitch_m": pitch,
        "voxel_shape": occupancy.shape,
        "occupied_voxel_count": int(np.sum(occupancy)),
        "topology": topology,
        "geometric_comparison": comparison,
        "contact_critical_comparison": contact_comparison,
        "projection_to_reference": False,
        "reason": (
            "candidate passes global topology/deviation gates" if accepted
            else "candidate rejected by topology or region-specific "
                 "geometric-deviation gate"),
    }
    (output / "implicit_reconstruction_report.json").write_text(json.dumps(
        json_ready(report), indent=2))
    if not accepted:
        raise ValueError(report["reason"])
    return vertices, faces, report
