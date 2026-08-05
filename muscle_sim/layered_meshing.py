"""Thin-region diagnostics and limited interior topology optimization."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from muscle_sim.cross_sections import (
    assign_bundle_coordinates, representative_bundle_path,
    rotation_minimizing_frames)
from muscle_sim.local_remeshing import (
    LOCAL_FACES, active_contact_points, json_ready, load_raw_mesh,
    tet_quality)
from viewer.isolated_muscle import (
    extract_boundary_faces, load_muscle_data, orient_tetrahedra)


def face_owners(tetrahedra):
    owners = defaultdict(list)
    for tet_id, tet in enumerate(tetrahedra):
        for local in LOCAL_FACES:
            face = tuple(sorted(map(int, tet[local])))
            owners[face].append(tet_id)
    return owners


def union_boundary(tetrahedra):
    counts = defaultdict(int)
    for tet in tetrahedra:
        for local in LOCAL_FACES:
            counts[tuple(sorted(map(int, tet[local])))] += 1
    return {face for face, count in counts.items() if count == 1}


def try_two_to_three_flip(vertices, tetrahedra, first_id, second_id):
    first, second = tetrahedra[first_id], tetrahedra[second_id]
    shared = sorted(set(first).intersection(second))
    if len(shared) != 3:
        return None
    first_opposite = list(set(first) - set(shared))
    second_opposite = list(set(second) - set(shared))
    if len(first_opposite) != 1 or len(second_opposite) != 1:
        return None
    d, e = first_opposite[0], second_opposite[0]
    a, b, c = shared
    candidate = np.asarray([
        [d, e, a, b], [d, e, b, c], [d, e, c, a]],
        dtype=np.int32)
    candidate = orient_tetrahedra(vertices, candidate)
    if union_boundary(candidate) != union_boundary(
            np.asarray([first, second])):
        return None
    quality = tet_quality(vertices, candidate)
    if np.any(quality["signed_volume"] <= 0.):
        return None
    return candidate, quality


def optimize_isolated_sliver(vertices, tetrahedra, tet_id):
    owners = face_owners(tetrahedra)
    original_quality = tet_quality(
        vertices, tetrahedra[[tet_id]])
    best = None
    for local in LOCAL_FACES:
        face = tuple(sorted(map(int, tetrahedra[tet_id][local])))
        adjacent = owners[face]
        if len(adjacent) != 2:
            continue
        neighbor = adjacent[0] if adjacent[1] == tet_id else adjacent[1]
        result = try_two_to_three_flip(
            vertices, tetrahedra, tet_id, neighbor)
        if result is None:
            continue
        candidate, quality = result
        score = float(np.min(quality["minimum_dihedral_degrees"]))
        row = {
            "neighbor_tet_id": neighbor, "shared_face": face,
            "replacement_tets": candidate,
            "minimum_dihedral_degrees": score,
            "maximum_edge_aspect_ratio": float(np.max(
                quality["edge_aspect_ratio"])),
        }
        if best is None or (
                row["minimum_dihedral_degrees"],
                -row["maximum_edge_aspect_ratio"],
                tuple(row["shared_face"])) > (
                    best["minimum_dihedral_degrees"],
                    -best["maximum_edge_aspect_ratio"],
                    tuple(best["shared_face"])):
            best = row
    if best is None:
        return None
    old_minimum = float(np.min(
        original_quality["minimum_dihedral_degrees"]))
    if best["minimum_dihedral_degrees"] <= old_minimum:
        return None
    remove = {tet_id, best["neighbor_tet_id"]}
    retained = np.asarray([
        tet for index, tet in enumerate(tetrahedra)
        if index not in remove], dtype=np.int32)
    result = np.vstack((retained, best["replacement_tets"]))
    best["removed_tet_ids"] = sorted(remove)
    best["old_minimum_dihedral_degrees"] = old_minimum
    return result, best


def failure_map(candidate_path, surface_path, config):
    data = np.load(candidate_path)
    vertices = data["vertices"]
    tetrahedra = data["tetrahedra"]
    quality = {
        key: data[key] for key in data.files
        if key not in ("vertices", "tetrahedra", "boundary_faces")}
    surface = trimesh.load(surface_path, force="mesh", process=False)
    centroids = np.mean(vertices[tetrahedra], axis=1)
    closest, surface_distance, surface_face = (
        trimesh.proximity.closest_point(surface, centroids))
    muscle, _ = load_muscle_data(config["current_tet_asset"])
    frames = rotation_minimizing_frames(
        representative_bundle_path(muscle.fibers))
    coordinates = assign_bundle_coordinates(centroids, frames)
    old = load_raw_mesh(config["current_tet_asset"])
    old_vertices = np.asarray(old["vertices"])
    old_faces = extract_boundary_faces(orient_tetrahedra(
        old_vertices, old["tetrahedra"]))
    contacts = active_contact_points(
        Path(config["diagnostic_dir"]) / "active_contact_multipliers.json",
        config["muscle_name"], old_vertices, old_faces)
    contact_distance = cKDTree(contacts).query(centroids)[0]
    attachment_points = np.concatenate([
        muscle.vertices[patch.vertex_ids]
        for patch in muscle.attachment_patches])
    attachment_distance = cKDTree(
        attachment_points).query(centroids)[0]
    triangle = np.asarray(surface.vertices)[surface.faces[surface_face]]
    local_edges = np.stack([
        np.linalg.norm(triangle[:, i] - triangle[:, (i + 1) % 3], axis=1)
        for i in range(3)], axis=1)
    local_edge = np.mean(local_edges, axis=1)
    failed_ids = np.where(
        quality["minimum_dihedral_degrees"] < 5.)[0]
    rows = []
    for tet_id in failed_ids:
        rows.append({
            "tet_id": int(tet_id),
            "minimum_dihedral_degrees": float(
                quality["minimum_dihedral_degrees"][tet_id]),
            "below_hard_2_degrees": bool(
                quality["minimum_dihedral_degrees"][tet_id] < 2.),
            "centroid": centroids[tet_id],
            "nearest_surface_triangle": int(surface_face[tet_id]),
            "distance_to_surface": float(surface_distance[tet_id]),
            "section_coordinate_s": float(coordinates["s"][tet_id]),
            "nearest_bundle_sample": int(
                coordinates["nearest_bundle_path_sample"][tet_id]),
            "local_surface_edge_length": float(local_edge[tet_id]),
            "distance_to_contact_region": float(contact_distance[tet_id]),
            "distance_to_attachment": float(attachment_distance[tet_id]),
            "edge_aspect_ratio": float(
                quality["edge_aspect_ratio"][tet_id]),
            "rest_volume": float(quality["rest_volume"][tet_id]),
        })
    return {
        "tet_count": len(tetrahedra),
        "below_2_degree_count": int(np.sum(
            quality["minimum_dihedral_degrees"] < 2.)),
        "below_5_degree_count": len(failed_ids),
        "minimum_dihedral_degrees": float(np.min(
            quality["minimum_dihedral_degrees"])),
        "failed_tets": rows,
    }, vertices, tetrahedra, failed_ids
