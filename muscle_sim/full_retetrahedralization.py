"""Full-muscle tetrahedralization and metadata transfer."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import cKDTree
import tetgen
import trimesh

from muscle_sim.full_surface import complete_topology
from muscle_sim.local_remeshing import (
    json_ready, load_raw_mesh, quality_summary, reembed_fibers,
    source_hash, tet_quality)
from viewer.isolated_muscle import (
    build_tet_fiber_directions, extract_boundary_faces, load_muscle_data,
    orient_tetrahedra)


def transfer_attachment_faces(reference_vertices, reference_faces,
                              current_muscle, distance):
    rows, all_face_ids = [], []
    centroids = np.mean(reference_vertices[reference_faces], axis=1)
    for patch_id, patch in enumerate(current_muscle.attachment_patches):
        points = current_muscle.vertices[patch.vertex_ids]
        tree = cKDTree(points)
        face_distance = tree.query(centroids)[0]
        face_ids = np.where(face_distance <= distance)[0]
        if not len(face_ids):
            face_ids = np.asarray([int(np.argmin(face_distance))])
        vertex_ids = np.unique(reference_faces[face_ids])
        old_center = np.mean(points, axis=0)
        new_center = np.mean(reference_vertices[vertex_ids], axis=0)
        old_axes = np.linalg.svd(points - old_center, full_matrices=False)[2]
        new_axes = np.linalg.svd(
            reference_vertices[vertex_ids] - new_center,
            full_matrices=False)[2]
        rows.append({
            "patch_id": patch_id, "bone_id": patch.bone_name,
            "end_type": patch.end_type,
            "surface_face_ids": face_ids,
            "vertex_ids": vertex_ids,
            "hard_core_vertex_ids": vertex_ids,
            "transition_vertex_ids": np.empty(0, dtype=np.int32),
            "old_centroid": old_center, "new_centroid": new_center,
            "centroid_change": float(np.linalg.norm(new_center - old_center)),
            "old_principal_axes": old_axes, "new_principal_axes": new_axes,
            "maximum_source_point_distance": float(np.max(
                cKDTree(reference_vertices[vertex_ids]).query(points)[0])),
        })
        all_face_ids.extend(face_ids.tolist())
    return rows, np.asarray(sorted(set(all_face_ids)), dtype=np.int32)


def contact_correspondence(old_vertices, old_faces, new_vertices, new_faces,
                           contact_points, radius):
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices, faces=new_faces, process=False)
    old_mesh = trimesh.Trimesh(
        vertices=old_vertices, faces=old_faces, process=False)
    old_centers = np.mean(old_vertices[old_faces], axis=1)
    if len(contact_points):
        critical = np.where(cKDTree(contact_points).query(
            old_centers)[0] <= radius)[0]
    else:
        critical = np.empty(0, dtype=np.int32)
    rows = []
    for old_face_id in critical:
        point = old_centers[old_face_id]
        closest, distance, new_face_id = trimesh.proximity.closest_point(
            new_mesh, point.reshape(1, 3))
        new_face_id = int(new_face_id[0])
        triangle = new_vertices[new_faces[new_face_id]]
        bary = trimesh.triangles.points_to_barycentric(
            triangle.reshape(1, 3, 3), closest)[0]
        normal_angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(
            old_mesh.face_normals[old_face_id],
            new_mesh.face_normals[new_face_id])), -1.0, 1.0))))
        old_area = trimesh.triangles.area(
            old_vertices[old_faces[old_face_id]].reshape(1, 3, 3))[0]
        new_area = trimesh.triangles.area(
            triangle.reshape(1, 3, 3))[0]
        rows.append({
            "old_face_id": int(old_face_id),
            "old_point": point,
            "nearest_new_surface_triangle": new_face_id,
            "barycentric_coordinates": bary,
            "distance": float(distance[0]),
            "normal_difference_degrees": normal_angle,
            "local_area_ratio": float(new_area / old_area),
        })
    return rows


def render_embedding(render_vertices, tet_vertices, tetrahedra, tolerance):
    # Rendering vertices coincide with the boundary in Path A. Use an exact
    # boundary-face embedding, not nearest tet vertices.
    boundary = extract_boundary_faces(tetrahedra)
    boundary_mesh = trimesh.Trimesh(
        vertices=tet_vertices, faces=boundary, process=False)
    closest, distance, face_id = trimesh.proximity.closest_point(
        boundary_mesh, render_vertices)
    rows = []
    for index, (point, error, selected) in enumerate(
            zip(closest, distance, face_id)):
        triangle = tet_vertices[boundary[int(selected)]]
        bary = trimesh.triangles.points_to_barycentric(
            triangle.reshape(1, 3, 3), point.reshape(1, 3))[0]
        rows.append({
            "render_vertex_id": index, "method": "boundary_face_barycentric",
            "boundary_face_id": int(selected), "barycentric": bary,
            "error": float(error),
        })
    maximum = float(np.max(distance))
    if maximum > tolerance:
        raise ValueError(
            f"render embedding error {maximum} exceeds {tolerance}")
    return rows


def tetrahedralize_full(surface_path, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    surface = trimesh.load(surface_path, force="mesh", process=False)
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int32)
    topology = complete_topology(vertices, faces)
    if not topology["valid_topology"]:
        raise ValueError("full-muscle input surface is not manifold")
    quality_cfg = config["quality_thresholds"]
    generator = tetgen.TetGen(vertices, faces)
    nodes, elements, *_ = generator.tetrahedralize(
        plc=True, quality=True, nobisect=True, nomergefacet=True,
        nomergevertex=True, zeroindex=True, quiet=True, docheck=True,
        mindihedral=float(
            quality_cfg["target_minimum_dihedral_degrees"]),
        opt_max_edge_ratio=float(
            quality_cfg["target_maximum_edge_aspect_ratio"]),
        minratio=2.0,
        opt_iterations=int(config["retetrahedralization"].get(
            "optimization_iterations", 3)),
        smooth_maxiter=int(config["retetrahedralization"].get(
            "smoothing_iterations", 7)),
        opt_max_asp_ratio=float(config["quality_thresholds"].get(
            "target_maximum_edge_aspect_ratio", 6.0)),
        steinerleft=(int(config["retetrahedralization"].get(
            "maximum_steiner_points", 100000))
            if config["retetrahedralization"][
                "allow_interior_steiner_points"] else 0))
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = orient_tetrahedra(nodes, np.asarray(elements, dtype=np.int32))
    boundary = extract_boundary_faces(elements)
    quality = tet_quality(nodes, elements)
    summary = quality_summary(quality)
    hard = float(quality_cfg["hard_minimum_dihedral_degrees"])
    if summary["minimum_dihedral_degrees"] < hard:
        np.savez_compressed(
            output / "rejected_tet_candidate.npz",
            vertices=nodes, tetrahedra=elements, boundary_faces=boundary,
            **quality)
        (output / "rejected_tet_quality.json").write_text(json.dumps(
            json_ready(summary), indent=2))
        raise ValueError(
            f"full tet minimum dihedral "
            f"{summary['minimum_dihedral_degrees']} < {hard}")
    boundary_topology = complete_topology(nodes, boundary)
    if not boundary_topology["valid_topology"]:
        raise ValueError("tetrahedralized boundary is not manifold")
    return nodes, elements, boundary, summary, topology


def build_derived_asset(surface_path, config, output):
    output = Path(output)
    nodes, elements, boundary, quality, surface_topology = (
        tetrahedralize_full(surface_path, config, output))
    reference = load_raw_mesh(config["selected_reference_asset"])
    current = load_raw_mesh(config["current_tet_asset"])
    current_muscle, _ = load_muscle_data(config["current_tet_asset"])
    reference_vertices = np.asarray(reference["vertices"])
    reference_faces = np.asarray(reference["render_faces"])
    attachment_rows, cap_face_ids = transfer_attachment_faces(
        reference_vertices, reference_faces, current_muscle,
        float(config["attachment_transfer"]["footprint_distance_m"]))
    # The initial surface vertices are required to remain first by the TetGen
    # fixed-facet contract. Abort rather than guess if this changes.
    if not np.allclose(nodes[:len(reference_vertices)],
                       reference_vertices, atol=1e-12, rtol=0.0):
        raise ValueError("TetGen changed or reordered reference vertices")
    fibers, fiber_report = reembed_fibers(
        current_muscle.fibers, nodes, elements,
        float(config["fiber_transfer"]["containment_tolerance_m"]))
    if (config["fiber_transfer"]["require_all_samples_embedded"]
            and fiber_report["outside_samples"]):
        raise ValueError("required fiber samples are outside the new volume")
    directions, direction_valid = build_tet_fiber_directions(
        nodes, elements, fibers)
    fiber_report["tets_with_no_fiber_direction"] = int(
        np.sum(~direction_valid))
    render_rows = render_embedding(
        reference_vertices, nodes, elements,
        float(config["render_embedding"]["maximum_error_m"]))
    current_vertices = np.asarray(current["vertices"])
    current_tets = orient_tetrahedra(
        current_vertices, current["tetrahedra"])
    current_boundary = extract_boundary_faces(current_tets)
    from muscle_sim.local_remeshing import active_contact_points
    contact_points = active_contact_points(
        Path(config["diagnostic_dir"]) / "active_contact_multipliers.json",
        config["muscle_name"], current_vertices, current_boundary)
    correspondence = contact_correspondence(
        current_vertices, current_boundary, nodes, boundary, contact_points,
        float(config["surface_reconstruction"]["contact_radius_m"]))
    max_contact = max([row["distance"] for row in correspondence] or [0.0])
    if max_contact > float(config["surface_reconstruction"][
            "maximum_contact_deviation_m"]):
        raise ValueError(
            f"contact surface deviation {max_contact} exceeds tolerance")
    # Build normal project dictionary while preserving waypoint geometry and
    # skeleton naming from the current simulation asset.
    derived = dict(current)
    derived.update({
        "vertices": nodes, "tetrahedra": elements,
        "faces": reference_faces.copy(),
        "render_faces": reference_faces.copy(),
        "sim_faces": boundary,
        "surface_face_count": len(reference_faces),
        "cap_face_indices": cap_face_ids,
        "anchor_vertices": np.unique(np.concatenate([
            row["vertex_ids"] for row in attachment_rows])),
        "cap_attachments": np.asarray([
            [int(row["vertex_ids"][0]), 0, row["end_type"], 0, 0]
            for row in attachment_rows], dtype=np.int32),
        "_transferred_attachment_patches": attachment_rows,
        "_transferred_fiber_embeddings": [{
            "stream_index": fiber.stream_index,
            "fiber_index": fiber.fiber_index,
            "tet_ids": fiber.tet_ids,
            "barycentric": fiber.barycentric,
        } for fiber in fibers],
        "_material_region_labels": np.zeros(
            len(elements), dtype=np.int32),
    })
    asset = output / "L_Semitendinosus_tet.npz"
    with asset.open("wb") as handle:
        pickle.dump(derived, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "attachment_labels.json").write_text(json.dumps(
        json_ready(attachment_rows), indent=2))
    np.savez_compressed(
        output / "fiber_embeddings.npz",
        fibers=np.asarray(fibers, dtype=object),
        directions=directions, valid=direction_valid)
    np.savez_compressed(
        output / "material_labels.npz",
        labels=np.zeros(len(elements), dtype=np.int32))
    (output / "render_embedding.json").write_text(json.dumps(
        json_ready(render_rows), indent=2))
    (output / "contact_surface_correspondence.json").write_text(json.dumps(
        json_ready(correspondence), indent=2))
    provenance = {
        "surface": str(surface_path),
        "surface_sha256": source_hash(surface_path),
        "backend": "TetGen", "backend_version": tetgen.__version__,
        "parameters": config["retetrahedralization"],
        "deterministic_seed": config["retetrahedralization"][
            "deterministic_seed"],
        "surface_topology": surface_topology,
        "tet_count": len(elements), "vertex_count": len(nodes),
        "quality_after": quality,
        "rest_state": {"F": "identity", "J": 1.0},
        "fiber_transfer": fiber_report,
        "contact_metadata": {
            "primitive_ids_rebuilt": True,
            "AL_multipliers_reset_required": True,
            "active_set_reset_required": True,
        },
    }
    (output / "tetrahedralization_provenance.json").write_text(json.dumps(
        json_ready(provenance), indent=2))
    return asset, provenance
