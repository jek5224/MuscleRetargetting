"""Contact-constrained fitting of the manifold Path B surface."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import trimesh

from muscle_sim.full_surface import complete_topology
from muscle_sim.local_remeshing import (
    active_contact_points, json_ready, load_raw_mesh)
from viewer.isolated_muscle import (
    extract_boundary_faces, load_muscle_data, orient_tetrahedra)


ATTACHMENT_LOCKED = 3
CONTACT_LOCKED = 2
TRANSITION = 1
FREE = 0
CLASS_NAMES = {
    ATTACHMENT_LOCKED: "ATTACHMENT_LOCKED",
    CONTACT_LOCKED: "CONTACT_LOCKED",
    TRANSITION: "TRANSITION",
    FREE: "FREE",
}


def surface_graph(vertices, faces):
    edges = set()
    for face in np.asarray(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]),
                              (face[2], face[0])):
            edges.add(tuple(sorted((int(first), int(second)))))
    edges = np.asarray(sorted(edges), dtype=np.int32)
    lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return sp.csr_matrix((
        np.concatenate((lengths, lengths)),
        (np.concatenate((edges[:, 0], edges[:, 1])),
         np.concatenate((edges[:, 1], edges[:, 0])))),
        shape=(len(vertices), len(vertices))), edges


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def geodesic_weights(vertices, faces, contact_points, attachment_points,
                     config):
    graph, edges = surface_graph(vertices, faces)
    tree = cKDTree(vertices)
    contact_sources = np.unique(tree.query(contact_points)[1]) if len(
        contact_points) else np.empty(0, dtype=np.int32)
    attachment_sources = np.unique(tree.query(attachment_points)[1]) if len(
        attachment_points) else np.empty(0, dtype=np.int32)
    contact_geo = (
        dijkstra(graph, directed=False, indices=contact_sources,
                 min_only=True)
        if len(contact_sources) else np.full(len(vertices), np.inf))
    attachment_geo = (
        dijkstra(graph, directed=False, indices=attachment_sources,
                 min_only=True)
        if len(attachment_sources) else np.full(len(vertices), np.inf))
    contact_euclidean = (
        cKDTree(contact_points).query(vertices)[0]
        if len(contact_points) else np.full(len(vertices), np.inf))
    attachment_euclidean = (
        cKDTree(attachment_points).query(vertices)[0]
        if len(attachment_points) else np.full(len(vertices), np.inf))
    contact_radius = float(config["contact_lock_radius_m"])
    contact_width = float(config["contact_transition_width_m"])
    attachment_radius = float(config["attachment_lock_radius_m"])
    attachment_width = float(config["attachment_transition_width_m"])
    contact_distance = np.minimum(contact_geo, contact_euclidean)
    attachment_distance = np.minimum(attachment_geo, attachment_euclidean)
    contact_weight = 1.0 - smoothstep(
        (contact_distance - contact_radius) / contact_width)
    attachment_weight = 1.0 - smoothstep(
        (attachment_distance - attachment_radius) / attachment_width)
    classes = np.full(len(vertices), FREE, dtype=np.int8)
    classes[(contact_weight > 0) | (attachment_weight > 0)] = TRANSITION
    classes[contact_distance <= contact_radius] = CONTACT_LOCKED
    classes[attachment_distance <= attachment_radius] = ATTACHMENT_LOCKED
    lock_weight = np.maximum(contact_weight, attachment_weight)
    return {
        "class": classes,
        "lock_weight": lock_weight,
        "contact_weight": contact_weight,
        "attachment_weight": attachment_weight,
        "contact_geodesic_distance": contact_geo,
        "attachment_geodesic_distance": attachment_geo,
        "contact_euclidean_distance": contact_euclidean,
        "attachment_euclidean_distance": attachment_euclidean,
        "contact_source_vertices": contact_sources,
        "attachment_source_vertices": attachment_sources,
        "edges": edges,
    }


def triangle_correspondence(vertices, faces, reference_vertices,
                            reference_faces, ambiguity_distance=2e-5):
    reference = trimesh.Trimesh(
        vertices=reference_vertices, faces=reference_faces, process=False)
    closest, distance, face_id = trimesh.proximity.closest_point(
        reference, vertices)
    triangles = reference_vertices[reference_faces[face_id]]
    barycentric = trimesh.triangles.points_to_barycentric(
        triangles, closest)
    current = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False)
    normals = np.zeros_like(vertices)
    np.add.at(normals, faces.ravel(),
              np.repeat(current.face_normals, 3, axis=0))
    normals /= np.maximum(np.linalg.norm(
        normals, axis=1, keepdims=True), 1e-30)
    reference_normal = reference.face_normals[face_id]
    normal_alignment = np.abs(np.einsum(
        "ij,ij->i", normals, reference_normal))
    # Query two reference triangle centroids to expose near-tie sheet cases.
    centroid_tree = cKDTree(np.mean(triangles, axis=1))
    # A global face-centroid tree is required; avoid the per-vertex repeated
    # triangle array above.
    centroid_tree = cKDTree(np.mean(
        reference_vertices[reference_faces], axis=1))
    near_distance, near_id = centroid_tree.query(vertices, k=2)
    ambiguous = (
        (near_distance[:, 1] - near_distance[:, 0])
        <= ambiguity_distance) & (
        near_id[:, 0] != near_id[:, 1]) & (normal_alignment < 0.5)
    return {
        "reference_face_id": face_id.astype(np.int32),
        "barycentric": barycentric,
        "target": closest,
        "distance": distance,
        "reference_normal": reference_normal,
        "normal_alignment": normal_alignment,
        "ambiguous": ambiguous,
    }


def umbrella_laplacian(vertex_count, edges):
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sp.csr_matrix((
        np.ones(len(rows)), (rows, columns)),
        shape=(vertex_count, vertex_count))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return sp.eye(vertex_count) - sp.diags(
        1.0 / np.maximum(degree, 1.0)) @ adjacency


def fit_positions(vertices, faces, masks, correspondence, contact_points,
                  attachment_points, config, fit_scale=1.0,
                  include_attachment=True):
    """Solve a smooth finite-weight positional fitting stage."""
    edges = masks["edges"]
    laplacian = umbrella_laplacian(len(vertices), edges)
    weights = masks["contact_weight"].copy()
    if include_attachment:
        weights = np.maximum(weights, masks["attachment_weight"])
    target = correspondence["target"].copy()
    # Actual contact points constrain the previously represented contact
    # corridor. Only vertices already in the contact mask receive this target.
    if len(contact_points):
        locked = masks["class"] == CONTACT_LOCKED
        target[locked] = correspondence["contact_corridor_target"][locked]
    if include_attachment and len(attachment_points):
        closest_attachment = cKDTree(attachment_points).query(vertices)[1]
        locked = masks["class"] == ATTACHMENT_LOCKED
        target[locked] = attachment_points[closest_attachment[locked]]
    position_weight = (
        float(config["position_fit_weight"]) * fit_scale)
    regularization = float(config["laplacian_weight"])
    anchor = float(config.get("path_b_anchor_weight", 1.0))
    diagonal = sp.diags(position_weight * weights + anchor)
    system = (
        diagonal + regularization * (laplacian.T @ laplacian))
    right = (
        (position_weight * weights)[:, None] * target
        + anchor * vertices
        + regularization * (laplacian.T @ (
            laplacian @ vertices)))
    fitted = np.column_stack([
        spsolve(system.tocsc(), right[:, axis]) for axis in range(3)])
    return fitted


def contact_metrics(surface, old_vertices, old_faces, contact_points,
                    radius):
    old_centroids = np.mean(old_vertices[old_faces], axis=1)
    critical = np.where(cKDTree(contact_points).query(
        old_centroids)[0] <= radius)[0]
    old_mesh = trimesh.Trimesh(
        vertices=old_vertices, faces=old_faces, process=False)
    closest, distance, face_id = trimesh.proximity.closest_point(
        surface, old_centroids[critical])
    angle = np.degrees(np.arccos(np.clip(np.abs(np.einsum(
        "ij,ij->i", old_mesh.face_normals[critical],
        surface.face_normals[face_id])), -1.0, 1.0)))
    return {
        "critical_face_count": len(critical),
        "maximum_deviation": float(np.max(distance)),
        "percentile_95_deviation": float(np.percentile(distance, 95)),
        "rms_deviation": float(np.sqrt(np.mean(distance ** 2))),
        "maximum_normal_deviation_degrees": float(np.max(angle)),
        "percentile_95_normal_deviation_degrees": float(
            np.percentile(angle, 95)),
    }


def candidate_report(name, vertices, faces, base_volume, old_vertices,
                     old_faces, contact_points, config):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    topology = complete_topology(vertices, faces)
    contact = contact_metrics(
        mesh, old_vertices, old_faces, contact_points,
        float(config["contact_lock_radius_m"]))
    triangles = vertices[faces]
    lengths = np.stack([
        np.linalg.norm(triangles[:, i] - triangles[:, (i + 1) % 3], axis=1)
        for i in range(3)], axis=1)
    aspect = np.max(lengths, axis=1) / np.maximum(
        np.min(lengths, axis=1), 1e-30)
    accepted = bool(
        topology["valid_topology"]
        and contact["maximum_deviation"] <= float(
            config["contact_max_deviation_m"])
        and contact["percentile_95_deviation"] <= float(
            config["contact_p95_deviation_m"])
        and np.max(aspect) <= float(config["maximum_triangle_aspect_ratio"])
        and abs(mesh.volume - base_volume) / abs(base_volume)
        <= float(config["maximum_volume_change_fraction"]))
    return {
        "name": name, "accepted": accepted,
        "topology": topology, "contact": contact,
        "maximum_triangle_aspect_ratio": float(np.max(aspect)),
        "volume": float(mesh.volume),
        "volume_change_fraction": float(
            abs(mesh.volume - base_volume) / abs(base_volume)),
    }


def build_masks_and_context(base_surface, reference_path, config):
    base = trimesh.load(base_surface, force="mesh", process=False)
    vertices, faces = np.asarray(base.vertices), np.asarray(base.faces)
    reference = load_raw_mesh(reference_path)
    old = load_raw_mesh(config["current_tet_asset"])
    old_vertices = np.asarray(old["vertices"])
    old_tets = orient_tetrahedra(old_vertices, old["tetrahedra"])
    old_faces = extract_boundary_faces(old_tets)
    old_surface_mesh = trimesh.Trimesh(
        vertices=old_vertices, faces=old_faces, process=False)
    contact_points = active_contact_points(
        Path(config["diagnostic_dir"]) / "active_contact_multipliers.json",
        config["muscle_name"], old_vertices, old_faces)
    old_muscle, _ = load_muscle_data(config["current_tet_asset"])
    attachment_points = np.concatenate([
        old_muscle.vertices[patch.vertex_ids]
        for patch in old_muscle.attachment_patches], axis=0)
    masks = geodesic_weights(
        vertices, faces, contact_points, attachment_points,
        config["contact_constrained_reconstruction"])
    correspondence = triangle_correspondence(
        vertices, faces, np.asarray(reference["vertices"]),
        np.asarray(reference["render_faces"]))
    old_closest, old_distance, old_face_id = (
        trimesh.proximity.closest_point(old_surface_mesh, vertices))
    correspondence["contact_corridor_target"] = old_closest
    correspondence["contact_corridor_face_id"] = old_face_id
    correspondence["contact_corridor_distance"] = old_distance
    correspondence["method"] = np.full(
        len(vertices), "nearest_reference_triangle", dtype=object)
    reference_vertices = np.asarray(reference["vertices"])
    reference_faces = np.asarray(reference["render_faces"])
    reference_mesh = trimesh.Trimesh(
        reference_vertices, reference_faces, process=False)
    base_mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False)
    base_normals = np.zeros_like(vertices)
    np.add.at(base_normals, faces.ravel(),
              np.repeat(base_mesh.face_normals, 3, axis=0))
    base_normals /= np.maximum(
        np.linalg.norm(base_normals, axis=1, keepdims=True), 1e-30)
    reference_centroids = np.mean(
        reference_vertices[reference_faces], axis=1)
    reference_tree = cKDTree(reference_centroids)
    # Edge/cap ties are resolved by a nearby triangle only when its geometric
    # distance is nearly equivalent and its normal follows the Path B sheet.
    for vertex in np.where(correspondence["ambiguous"])[0]:
        candidate_ids = reference_tree.query_ball_point(
            vertices[vertex], 0.002)
        if not candidate_ids:
            continue
        candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
        triangles = reference_vertices[reference_faces[candidate_ids]]
        points = np.repeat(
            vertices[vertex].reshape(1, 3), len(candidate_ids), axis=0)
        closest = trimesh.triangles.closest_point(triangles, points)
        distance = np.linalg.norm(closest - points, axis=1)
        alignment = np.abs(
            reference_mesh.face_normals[candidate_ids] @ base_normals[vertex])
        eligible = distance <= (
            correspondence["distance"][vertex] + 0.0002)
        if not np.any(eligible):
            continue
        score = np.where(eligible, alignment, -np.inf)
        selected = int(np.argmax(score))
        if alignment[selected] < 0.7:
            continue
        face_id = int(candidate_ids[selected])
        barycentric = trimesh.triangles.points_to_barycentric(
            triangles[selected:selected + 1],
            closest[selected:selected + 1])[0]
        correspondence["reference_face_id"][vertex] = face_id
        correspondence["barycentric"][vertex] = barycentric
        correspondence["target"][vertex] = closest[selected]
        correspondence["distance"][vertex] = distance[selected]
        correspondence["reference_normal"][vertex] = (
            reference_mesh.face_normals[face_id])
        correspondence["normal_alignment"][vertex] = alignment[selected]
        correspondence["ambiguous"][vertex] = False
        correspondence["method"][vertex] = (
            "normal_aligned_nearby_reference_triangle")
    # Resolve a near-tie only when the existing Path B one-ring supplies a
    # coherent reference sheet. This changes the ambiguity label, not the
    # already triangle-based closest point.
    neighbors = [set() for _ in vertices]
    for first, second in masks["edges"]:
        neighbors[int(first)].add(int(second))
        neighbors[int(second)].add(int(first))
    for vertex in np.where(correspondence["ambiguous"])[0]:
        good = [neighbor for neighbor in neighbors[vertex]
                if not correspondence["ambiguous"][neighbor]]
        if len(good) < 2:
            continue
        targets = correspondence["target"][good]
        normals = correspondence["reference_normal"][good]
        coherent_position = np.max(np.linalg.norm(
            targets - np.mean(targets, axis=0), axis=1)) <= 0.001
        mean_normal = np.mean(normals, axis=0)
        mean_normal /= max(np.linalg.norm(mean_normal), 1e-30)
        coherent_normal = np.min(np.abs(normals @ mean_normal)) >= 0.7
        if coherent_position and coherent_normal:
            correspondence["ambiguous"][vertex] = False
            correspondence["method"][vertex] = (
                "connected_neighbor_sheet_continuity")
    # Attachment geometry has higher priority than generic reference-sheet
    # correspondence. A cap-edge tie may therefore use the known transferred
    # footprint, but this exception is forbidden for contact-only vertices.
    for vertex in np.where(
            correspondence["ambiguous"]
            & (masks["class"] == ATTACHMENT_LOCKED))[0]:
        attachment_id = int(cKDTree(
            attachment_points).query(vertices[vertex])[1])
        correspondence["target"][vertex] = attachment_points[attachment_id]
        correspondence["distance"][vertex] = float(np.linalg.norm(
            vertices[vertex] - attachment_points[attachment_id]))
        correspondence["ambiguous"][vertex] = False
        correspondence["method"][vertex] = (
            "attachment_footprint_priority_override")
    locked_ambiguous = correspondence["ambiguous"] & (
        masks["class"] >= CONTACT_LOCKED)
    return {
        "base": base, "vertices": vertices, "faces": faces,
        "reference": reference, "old_vertices": old_vertices,
        "old_faces": old_faces, "contact_points": contact_points,
        "attachment_points": attachment_points, "masks": masks,
        "correspondence": correspondence,
        "locked_ambiguous_vertices": np.where(locked_ambiguous)[0],
    }


def write_mask_outputs(context, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vertices, faces, masks = (
        context["vertices"], context["faces"], context["masks"])
    rows = [{
        "vertex_id": index,
        "coordinates": vertices[index].tolist(),
        "mask_class": CLASS_NAMES[int(masks["class"][index])],
        "lock_weight": float(masks["lock_weight"][index]),
        "geodesic_distance_to_contact": float(
            masks["contact_geodesic_distance"][index]),
        "euclidean_distance_to_contact": float(
            masks["contact_euclidean_distance"][index]),
    } for index in range(len(vertices))]
    (output / "contact_critical_mask.json").write_text(json.dumps(
        json_ready({
            "source_contact_ids": list(range(
                len(context["contact_points"]))),
            "surface_vertex_ids": np.where(
                masks["class"] >= CONTACT_LOCKED)[0],
            "surface_triangle_ids": np.where(np.any(
                masks["class"][faces] >= CONTACT_LOCKED, axis=1))[0],
            "vertices": rows,
        }), indent=2))
    (output / "attachment_critical_mask.json").write_text(json.dumps(
        json_ready({
            "surface_vertex_ids": np.where(
                masks["class"] == ATTACHMENT_LOCKED)[0],
            "overlap_with_contact": np.where(
                (masks["attachment_weight"] > 0)
                & (masks["contact_weight"] > 0))[0],
        }), indent=2))
    colors = np.asarray([
        ([230, 30, 30, 255] if value == CONTACT_LOCKED else
         [30, 80, 240, 255] if value == ATTACHMENT_LOCKED else
         [250, 180, 30, 255] if value == TRANSITION else
         [130, 130, 130, 255])
        for value in masks["class"]], dtype=np.uint8)
    trimesh.points.PointCloud(vertices, colors=colors).export(
        output / "contact_critical_vertices.ply")
    critical_faces = faces[np.any(
        masks["class"][faces] >= CONTACT_LOCKED, axis=1)]
    trimesh.Trimesh(
        vertices=vertices, faces=critical_faces, process=False).export(
            output / "contact_critical_faces.obj")
    correspondence = context["correspondence"]
    (output / "pathB_to_reference_correspondence.json").write_text(
        json.dumps(json_ready({
            "unresolved_locked_vertex_ids":
                context["locked_ambiguous_vertices"],
            "rows": [{
                "path_b_vertex_id": index,
                "nearest_reference_triangle": int(
                    correspondence["reference_face_id"][index]),
                "barycentric_coordinates":
                    correspondence["barycentric"][index],
                "distance": float(correspondence["distance"][index]),
                "reference_normal":
                    correspondence["reference_normal"][index],
                "normal_alignment": float(
                    correspondence["normal_alignment"][index]),
                "ambiguous": bool(correspondence["ambiguous"][index]),
                "method": str(correspondence["method"][index]),
                "mask_class": CLASS_NAMES[int(masks["class"][index])],
            } for index in range(len(vertices))],
        }), indent=2))


def fit_candidates(context, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    local = config["contact_constrained_reconstruction"]
    if len(context["locked_ambiguous_vertices"]):
        summary = {
            "candidates": [],
            "accepted_candidate": None,
            "metadata_transfer_performed": False,
            "tetrahedralization_performed": False,
            "failure_classification": "REFERENCE_CORRESPONDENCE_AMBIGUOUS",
            "unresolved_locked_vertex_ids":
                context["locked_ambiguous_vertices"],
        }
        (output / "hybrid_candidate_comparison.json").write_text(json.dumps(
            json_ready(summary), indent=2))
        raise ValueError(
            f"{len(context['locked_ambiguous_vertices'])} unresolved "
            "locked-region reference correspondences")
    vertices, faces = context["vertices"], context["faces"]
    base_volume = float(context["base"].volume)
    definitions = [
        ("B0", 0.0, False),
        ("B1", 0.2, False),
        ("B2", 0.35, False),
        ("B3", 0.6, True),
    ]
    candidates = []
    for name, scale, attachments in definitions:
        current = vertices.copy()
        if scale:
            current = fit_positions(
                current, faces, context["masks"],
                context["correspondence"], context["contact_points"],
                context["attachment_points"], local, scale, attachments)
        report = candidate_report(
            name, current, faces, base_volume, context["old_vertices"],
            context["old_faces"], context["contact_points"], local)
        path = output / f"{name}_surface.obj"
        trimesh.Trimesh(
            vertices=current, faces=faces, process=False).export(path)
        report["surface_path"] = str(path)
        candidates.append(report)
    accepted = [row for row in candidates if row["accepted"]]
    ranking = sorted(candidates, key=lambda row: (
        not row["accepted"], row["contact"]["maximum_deviation"],
        row["contact"]["percentile_95_deviation"],
        row["maximum_triangle_aspect_ratio"]))
    summary = {
        "candidates": ranking,
        "accepted_candidate": ranking[0]["name"] if accepted else None,
        "metadata_transfer_performed": False,
        "tetrahedralization_performed": False,
        "failure_classification": (
            None if accepted else "IMPLICIT_TOPOLOGY_INCOMPATIBLE_WITH_CONTACT_GEOMETRY"),
    }
    (output / "hybrid_candidate_comparison.json").write_text(json.dumps(
        json_ready(summary), indent=2))
    if not accepted:
        raise ValueError(
            "no hybrid candidate passes all mandatory surface gates")
    best = ranking[0]
    Path(best["surface_path"]).replace(output / "hybrid_surface.obj")
    return output / "hybrid_surface.obj", summary
