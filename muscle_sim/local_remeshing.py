"""Deterministic, fixed-boundary tetrahedral cavity repair utilities.

The routines in this module are deliberately independent of the contact
solver.  They operate in rest space, preserve the original surface/interface
vertices, and reject a tetrahedralizer result unless every cavity facet is
recovered exactly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import platform
import itertools

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from viewer.isolated_muscle import (
    EmbeddedFiber, MuscleData, TetLocator, attachment_patches_from_tet,
    build_tet_fiber_directions, explicit_fiber_polylines,
    extract_boundary_faces, orient_tetrahedra,
)


LOCAL_FACES = np.asarray(
    ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def load_raw_mesh(path):
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a project tet dictionary")
    return data


def source_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tet_face_owners(tetrahedra):
    owners = defaultdict(list)
    for tet_id, tet in enumerate(np.asarray(tetrahedra)):
        for local_face in LOCAL_FACES:
            oriented = tuple(int(v) for v in tet[local_face])
            owners[tuple(sorted(oriented))].append((tet_id, oriented))
    return owners


def tet_adjacency(tetrahedra, shared_vertices=1):
    """Adjacency sharing a vertex (default), edge, or face."""
    incident = defaultdict(list)
    for tet_id, tet in enumerate(np.asarray(tetrahedra)):
        for vertex in tet:
            incident[int(vertex)].append(tet_id)
    counts = defaultdict(int)
    for ids in incident.values():
        for i, first in enumerate(ids):
            for second in ids[i + 1:]:
                counts[(min(first, second), max(first, second))] += 1
    result = [set() for _ in tetrahedra]
    for (first, second), count in counts.items():
        if count >= shared_vertices:
            result[first].add(second)
            result[second].add(first)
    return result


def tet_quality(vertices, tetrahedra):
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    q = vertices[tetrahedra]
    signed_six_volume = np.einsum(
        "ij,ij->i", q[:, 0] - q[:, 3],
        np.cross(q[:, 1] - q[:, 3], q[:, 2] - q[:, 3]))
    lengths = np.stack([
        np.linalg.norm(q[:, i] - q[:, j], axis=1)
        for i in range(4) for j in range(i + 1, 4)], axis=1)
    areas = np.stack([
        0.5 * np.linalg.norm(np.cross(
            q[:, face[1]] - q[:, face[0]],
            q[:, face[2]] - q[:, face[0]]), axis=1)
        for face in LOCAL_FACES], axis=1)
    face_normals = []
    for face in LOCAL_FACES:
        normal = np.cross(
            q[:, face[1]] - q[:, face[0]],
            q[:, face[2]] - q[:, face[0]])
        normal /= np.maximum(
            np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        face_normals.append(normal)
    dihedral = []
    for first in range(4):
        for second in range(first + 1, 4):
            dot = np.clip(np.einsum(
                "ij,ij->i", face_normals[first],
                face_normals[second]), -1.0, 1.0)
            dihedral.append(np.pi - np.arccos(dot))
    dihedral = np.stack(dihedral, axis=1)
    volume = np.abs(signed_six_volume) / 6.0
    # R/r = (abc)/(6Vr), r = 3V/surface area.
    circumradius = (
        lengths[:, 0] * lengths[:, 1] * lengths[:, 3]
        / np.maximum(12.0 * volume, 1e-30))
    inradius = 3.0 * volume / np.maximum(np.sum(areas, axis=1), 1e-30)
    mean_length = np.mean(lengths, axis=1)
    return {
        "signed_volume": signed_six_volume / 6.0,
        "rest_volume": volume,
        "minimum_dihedral_degrees": np.degrees(np.min(dihedral, axis=1)),
        "maximum_dihedral_degrees": np.degrees(np.max(dihedral, axis=1)),
        "edge_aspect_ratio": (
            np.max(lengths, axis=1)
            / np.maximum(np.min(lengths, axis=1), 1e-30)),
        "radius_ratio": circumradius / np.maximum(inradius, 1e-30),
        "volume_length_quality": (
            6.0 * np.sqrt(2.0) * volume
            / np.maximum(mean_length ** 3, 1e-30)),
    }


def quality_summary(quality, ids=None, sliver_degrees=2.0):
    ids = np.arange(len(quality["rest_volume"])) if ids is None else np.asarray(ids)
    return {
        "tet_count": int(len(ids)),
        "minimum_dihedral_degrees": float(np.min(
            quality["minimum_dihedral_degrees"][ids])),
        "maximum_dihedral_degrees": float(np.max(
            quality["maximum_dihedral_degrees"][ids])),
        "maximum_radius_ratio": float(np.max(quality["radius_ratio"][ids])),
        "maximum_edge_aspect_ratio": float(np.max(
            quality["edge_aspect_ratio"][ids])),
        "minimum_rest_volume": float(np.min(quality["rest_volume"][ids])),
        "maximum_rest_volume": float(np.max(quality["rest_volume"][ids])),
        "nonpositive_count": int(np.sum(
            quality["signed_volume"][ids] <= 0.0)),
        "sliver_count": int(np.sum(
            quality["minimum_dihedral_degrees"][ids] < sliver_degrees)),
    }


def _diagnostic_seed_tets(diagnostic_dir, muscle):
    result = set()
    for filename in ("low_J_tet_report.json",
                     "local_remeshing_recommendation.json"):
        path = Path(diagnostic_dir) / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows = data.get("regions", []) if isinstance(data, dict) else data
        result.update(int(row["tet_id"]) for row in rows
                      if row.get("muscle") == muscle)
    return result


def active_contact_points(path, muscle, vertices, surface_faces):
    if not Path(path).exists():
        return np.empty((0, 3))
    rows = json.loads(Path(path).read_text())
    points = []
    for row in rows:
        if row.get("target_body") == muscle:
            triangle = row.get("target_triangle", [])
            if len(triangle) == 3:
                points.append(np.mean(vertices[np.asarray(triangle)], axis=0))
        if row.get("source_body") == muscle and "source_vertex" in row:
            points.append(vertices[int(row["source_vertex"])])
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


@dataclass
class RegionSelection:
    selected_tets: np.ndarray
    seed_tets: np.ndarray
    graph_distance: np.ndarray
    boundary_faces: np.ndarray
    external_faces: np.ndarray
    interface_faces: np.ndarray
    selected_vertices: np.ndarray
    report: dict


def select_region(vertices, tetrahedra, external_surface, config,
                  diagnostic_dir, muscle, attachments=(), fibers=()):
    local = config["local_remeshing"]
    thresholds = config["quality_thresholds"]
    quality = tet_quality(vertices, tetrahedra)
    seeds = set(int(v) for v in local.get("seed_tets", []))
    if local.get("use_diagnostic_seed_tets", True):
        seeds |= _diagnostic_seed_tets(diagnostic_dir, muscle)
    poor = (
        (quality["minimum_dihedral_degrees"]
         < float(thresholds["include_minimum_dihedral_degrees"]))
        | (quality["edge_aspect_ratio"]
           > float(thresholds["include_maximum_edge_aspect_ratio"]))
        | (quality["radius_ratio"]
           > float(thresholds.get("maximum_radius_ratio", np.inf)))
        | (quality["rest_volume"]
           < float(thresholds.get("minimum_rest_volume", -np.inf))))
    globally_poor = set(np.where(poor)[0].tolist())
    contact_path = Path(diagnostic_dir) / "active_contact_multipliers.json"
    contact_points = active_contact_points(
        contact_path, muscle, vertices, external_surface)
    contact_tets = set()
    radius = float(local["contact_radius_m"])
    if len(contact_points) and local.get("include_contact_neighbors", True):
        surface_keys = {tuple(sorted(map(int, face)))
                        for face in np.asarray(external_surface)}
        tree = cKDTree(contact_points)
        for tet_id, tet in enumerate(tetrahedra):
            boundary = [tet[face] for face in LOCAL_FACES
                        if tuple(sorted(map(int, tet[face]))) in surface_keys]
            if boundary and min(tree.query(
                    np.mean(vertices[face], axis=0))[0]
                                for face in boundary) <= radius:
                contact_tets.add(tet_id)
        seeds |= contact_tets
    if not seeds:
        raise ValueError("local remeshing selection has no seed tetrahedra")
    adjacency = tet_adjacency(tetrahedra, shared_vertices=1)
    distance = np.full(len(tetrahedra), -1, dtype=np.int32)
    queue = deque(sorted(seeds))
    for tet_id in seeds:
        if tet_id < 0 or tet_id >= len(tetrahedra):
            raise IndexError(f"seed tet {tet_id} is outside the mesh")
        distance[tet_id] = 0
    rings = int(local["adjacency_rings"])
    while queue:
        current = queue.popleft()
        if distance[current] >= rings:
            continue
        for neighbor in sorted(adjacency[current]):
            if distance[neighbor] < 0:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    # "include_low_quality_neighbors" is intentionally local: on legacy
    # coarse assets the nominal threshold can classify most of the muscle.
    # Global inclusion would silently turn cavity mode into full remeshing.
    # The graph expansion above includes every poor tet reached within the
    # configured neighborhood, while retaining the global count in the report.
    selected = np.where(distance >= 0)[0]
    selected_set = set(selected.tolist())
    owners = tet_face_owners(tetrahedra)
    boundary, external, interface = [], [], []
    external_keys = {tuple(sorted(map(int, face)))
                     for face in np.asarray(external_surface)}
    for key, rows in owners.items():
        chosen = [row for row in rows if row[0] in selected_set]
        if len(chosen) == 1 and len(chosen) != len(rows):
            boundary.append(chosen[0][1])
            interface.append(chosen[0][1])
        elif len(chosen) == 1 and len(rows) == 1:
            boundary.append(chosen[0][1])
            external.append(chosen[0][1])
    boundary = np.asarray(boundary, dtype=np.int32).reshape(-1, 3)
    external = np.asarray(external, dtype=np.int32).reshape(-1, 3)
    interface = np.asarray(interface, dtype=np.int32).reshape(-1, 3)
    vertices_selected = np.unique(tetrahedra[selected])
    attachment_ids = set(int(v) for patch in attachments for v in patch.vertex_ids)
    fiber_points = np.concatenate(
        [fiber.rest_points for fiber in fibers], axis=0) if fibers else np.empty((0, 3))
    cavity_mesh = trimesh.Trimesh(
        vertices=vertices, faces=boundary, process=False)
    contains_fiber = bool(
        len(fiber_points) and np.any(cavity_mesh.contains(fiber_points)))
    report = {
        "muscle": muscle,
        "selected_tet_ids": selected.tolist(),
        "seed_tet_ids": sorted(seeds),
        "contact_critical_tet_ids": sorted(contact_tets),
        "globally_low_quality_tet_count": len(globally_poor),
        "selected_low_quality_tet_ids": sorted(
            globally_poor.intersection(selected.tolist())),
        "selected_vertex_ids": vertices_selected.tolist(),
        "selected_boundary_faces": boundary.tolist(),
        "selected_external_faces": external.tolist(),
        "selected_internal_interface_faces": interface.tolist(),
        "intersects_external_surface": bool(len(external)),
        "intersects_attachment_patches": bool(
            attachment_ids.intersection(vertices_selected.tolist())),
        "contains_embedded_fiber_samples": contains_fiber,
        "graph_distance_by_tet": {
            str(int(i)): int(distance[i]) for i in selected},
        "quality_before": quality_summary(quality, selected),
    }
    return RegionSelection(
        selected, np.asarray(sorted(seeds)), distance, boundary, external,
        interface, vertices_selected, report)


def validate_closed_boundary(vertices, faces, area_tolerance=1e-16):
    faces = np.asarray(faces, dtype=np.int32)
    keys = np.sort(faces, axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    duplicate = unique[counts > 1]
    edge_rows = defaultdict(list)
    for face_id, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]),
                     (face[2], face[0])):
            edge_rows[tuple(sorted((int(a), int(b))))].append(
                (face_id, int(a), int(b)))
    bad_edges = [edge for edge, rows in edge_rows.items() if len(rows) != 2]
    orientation_edges = [
        edge for edge, rows in edge_rows.items()
        if len(rows) == 2 and rows[0][1:] == rows[1][1:]]
    triangle = np.asarray(vertices)[faces]
    areas = 0.5 * np.linalg.norm(np.cross(
        triangle[:, 1] - triangle[:, 0],
        triangle[:, 2] - triangle[:, 0]), axis=1)
    zero_area = np.where(areas <= area_tolerance)[0]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1)
    report = {
        "watertight": not bad_edges,
        "manifold": not bad_edges,
        "consistent_orientation": not orientation_edges,
        "duplicate_faces": duplicate.tolist(),
        "nonmanifold_or_open_edges": [list(v) for v in bad_edges],
        "orientation_conflict_edges": [list(v) for v in orientation_edges],
        "zero_area_face_ids": zero_area.tolist(),
        "connected_component_count": len(components),
        "self_intersection_check": "not_available_in_trimesh_without_fcl",
        "valid": bool(not bad_edges and not orientation_edges
                      and not len(duplicate) and not len(zero_area)
                      and len(components) == 1),
    }
    return report


def _edge_face_ids(faces, edge):
    edge = set(map(int, edge))
    return [face_id for face_id, face in enumerate(faces)
            if edge.issubset(set(map(int, face)))]


def attempt_local_seam_repair(vertices, faces, external_faces,
                              authored_faces, attachment_vertices,
                              active_points, config):
    """Try a transactionally safe repair limited to invalid-edge faces.

    Four-sheet branches are classified using topology, authored-surface
    membership, orientation, normals, and contact distance.  Candidate
    removals are accepted only if the *complete* resulting cavity is closed.
    This deliberately cannot turn a branching sheet into an unreported hole.
    """
    before = validate_closed_boundary(vertices, faces)
    bad_edges = before["nonmanifold_or_open_edges"]
    external_keys = {tuple(sorted(map(int, face)))
                     for face in np.asarray(external_faces)}
    authored_keys = {tuple(sorted(map(int, face)))
                     for face in np.asarray(authored_faces)}
    attachment_vertices = set(map(int, attachment_vertices))
    contact_tree = cKDTree(active_points) if len(active_points) else None
    groups, decisions = [], []
    for edge in bad_edges:
        ids = _edge_face_ids(faces, edge)
        if len(ids) != 4:
            decisions.append({
                "edge": edge, "classification": "not_a_four_sheet_branch",
                "incident_face_ids": ids})
            continue
        group = []
        for face_id in ids:
            face = faces[face_id]
            triangle = vertices[face]
            cross = np.cross(
                triangle[1] - triangle[0], triangle[2] - triangle[0])
            area = 0.5 * np.linalg.norm(cross)
            normal = cross / max(np.linalg.norm(cross), 1e-30)
            centroid = np.mean(triangle, axis=0)
            group.append({
                "cavity_face_id": face_id,
                "vertices": face.tolist(),
                "area": float(area),
                "normal": normal.tolist(),
                "is_external": (
                    tuple(sorted(map(int, face))) in external_keys),
                "is_authored_anatomical_face": (
                    tuple(sorted(map(int, face))) in authored_keys),
                "touches_attachment": bool(
                    attachment_vertices.intersection(map(int, face))),
                "distance_to_active_contact": (
                    float(contact_tree.query(centroid)[0])
                    if contact_tree is not None else None),
            })
        groups.append((edge, ids))
        decisions.append({
            "edge": edge,
            "classification": "four_sheet_branch_not_duplicate_face",
            "incident_faces": group,
        })
    candidates = []
    if len(groups) == len(bad_edges):
        choices = [
            list(itertools.combinations(ids, 2)) for _, ids in groups]
        for removed_by_edge in itertools.product(*choices):
            removed = sorted(set(itertools.chain.from_iterable(
                removed_by_edge)))
            candidate = np.delete(faces, removed, axis=0)
            topology = validate_closed_boundary(vertices, candidate)
            removed_rows = [faces[index] for index in removed]
            authored_removed = sum(
                tuple(sorted(map(int, face))) in authored_keys
                for face in removed_rows)
            external_removed = sum(
                tuple(sorted(map(int, face))) in external_keys
                for face in removed_rows)
            attachment_removed = sum(bool(
                attachment_vertices.intersection(map(int, face)))
                for face in removed_rows)
            contact_distance = [
                float(contact_tree.query(
                    np.mean(vertices[face], axis=0))[0])
                for face in removed_rows] if contact_tree is not None else []
            candidates.append({
                "removed_face_ids": removed,
                "valid": topology["valid"],
                "remaining_invalid_edge_count": len(
                    topology["nonmanifold_or_open_edges"]),
                "orientation_conflict_count": len(
                    topology["orientation_conflict_edges"]),
                "authored_faces_removed": authored_removed,
                "external_faces_removed": external_removed,
                "attachment_faces_removed": attachment_removed,
                "minimum_contact_distance": min(
                    contact_distance or [float("inf")]),
                "_faces": candidate,
                "_topology": topology,
            })
    candidates.sort(key=lambda row: (
        not row["valid"], row["attachment_faces_removed"],
        row["authored_faces_removed"], row["external_faces_removed"],
        -row["minimum_contact_distance"],
        row["remaining_invalid_edge_count"],
        row["removed_face_ids"]))
    accepted = next((row for row in candidates if row["valid"]
                     and row["attachment_faces_removed"] == 0), None)
    report_candidates = [{
        key: value for key, value in row.items()
        if not key.startswith("_")} for row in candidates]
    if accepted is None:
        return faces.copy(), {
            "accepted": False,
            "reason": (
                "no face-only repair confined to the identified four-sheet "
                "branches produces a closed cavity"),
            "scope_edge_count": len(bad_edges),
            "classified_edges": decisions,
            "candidate_count": len(candidates),
            "best_candidates": report_candidates[:12],
            "removed_faces": [],
            "retained_faces": list(range(len(faces))),
            "retriangulated_faces": [],
            "vertex_displacement_maximum": 0.0,
            "contact_surface_changes": {
                "changed_triangle_count": 0,
                "changed_surface_area": 0.0,
                "maximum_geometric_deviation": 0.0,
                "rms_geometric_deviation": 0.0,
                "maximum_normal_deviation_degrees": 0.0,
                "modified_triangle_contact_distances": [],
            },
            "attachment_region_changed": False,
            "requires_full_muscle_manifold_surface_repair": True,
        }
    removed = accepted["removed_face_ids"]
    retained = [index for index in range(len(faces))
                if index not in set(removed)]
    repaired = accepted["_faces"]
    old_area = float(np.sum(trimesh.triangles.area(vertices[faces])))
    new_area = float(np.sum(trimesh.triangles.area(vertices[repaired])))
    return repaired, {
        "accepted": True,
        "reason": "topologically valid classified four-sheet removal",
        "classified_edges": decisions,
        "removed_faces": [{
            "cavity_face_id": index,
            "vertices": faces[index].tolist(),
            "classification": "removed_branch_sheet",
        } for index in removed],
        "retained_faces": retained,
        "retriangulated_faces": [],
        "vertex_displacement_maximum": 0.0,
        "contact_surface_changes": {
            "changed_triangle_count": len(removed),
            "changed_surface_area": new_area - old_area,
            "maximum_geometric_deviation": 0.0,
            "rms_geometric_deviation": 0.0,
            "maximum_normal_deviation_degrees": 0.0,
            "minimum_distance_to_active_contacts":
                accepted["minimum_contact_distance"],
        },
        "topology_after": accepted["_topology"],
        "requires_full_muscle_manifold_surface_repair": False,
    }


class LocalTetBackend(ABC):
    @abstractmethod
    def tetrahedralize(self, boundary_vertices, boundary_faces,
                       quality_options):
        raise NotImplementedError


class TetGenBackend(LocalTetBackend):
    def __init__(self):
        import tetgen
        self.module = tetgen

    @property
    def version(self):
        return getattr(self.module, "__version__", "unknown")

    def tetrahedralize(self, boundary_vertices, boundary_faces,
                       quality_options):
        generator = self.module.TetGen(
            np.asarray(boundary_vertices, dtype=np.float64),
            np.asarray(boundary_faces, dtype=np.int32))
        nodes, elements, *_ = generator.tetrahedralize(
            plc=True, quality=True, nobisect=True, nomergefacet=True,
            nomergevertex=True, zeroindex=True, quiet=True,
            mindihedral=float(quality_options["target_minimum_dihedral_degrees"]),
            opt_max_edge_ratio=float(
                quality_options["target_maximum_edge_aspect_ratio"]),
            minratio=float(quality_options.get("tetgen_minratio", 2.0)),
            steinerleft=(
                100000 if quality_options.get(
                    "allow_interior_steiner_points", True) else 0),
        )
        return np.asarray(nodes), np.asarray(elements, dtype=np.int32)


def backend_from_name(name):
    if name in ("auto", "tetgen"):
        try:
            return TetGenBackend()
        except ImportError:
            if name == "tetgen":
                raise
    raise RuntimeError(
        "no fixed-boundary local tetrahedralization backend is available")


def _surface_correspondence(new_vertices, new_faces, old_vertices, old_faces):
    old_mesh = trimesh.Trimesh(
        vertices=old_vertices, faces=old_faces, process=False)
    centroids = np.mean(new_vertices[new_faces], axis=1)
    closest, distance, old_face = trimesh.proximity.closest_point(
        old_mesh, centroids)
    return old_face.astype(np.int32), distance, closest


def surface_deviation(old_vertices, old_faces, new_vertices, new_faces):
    old_mesh = trimesh.Trimesh(
        vertices=old_vertices, faces=old_faces, process=False)
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices, faces=new_faces, process=False)
    old_points = old_vertices[np.unique(old_faces)]
    new_points = new_vertices[np.unique(new_faces)]
    _, old_new, _ = trimesh.proximity.closest_point(new_mesh, old_points)
    _, new_old, _ = trimesh.proximity.closest_point(old_mesh, new_points)
    new_face_id, _, _ = _surface_correspondence(
        new_vertices, new_faces, old_vertices, old_faces)
    old_normals = old_mesh.face_normals[new_face_id]
    new_normals = new_mesh.face_normals
    normal_angle = np.degrees(np.arccos(np.clip(
        np.abs(np.einsum("ij,ij->i", old_normals, new_normals)),
        -1.0, 1.0)))
    distances = np.concatenate((old_new, new_old))
    return {
        "maximum_bidirectional_hausdorff_distance": float(np.max(distances)),
        "rms_surface_distance": float(np.sqrt(np.mean(distances ** 2))),
        "maximum_normal_deviation_degrees": float(np.max(normal_angle)),
        "rms_normal_deviation_degrees": float(np.sqrt(
            np.mean(normal_angle ** 2))),
        "maximum_boundary_edge_displacement": 0.0,
        "maximum_contact_neighborhood_displacement": 0.0,
    }


def reembed_fibers(old_fibers, vertices, tetrahedra, tolerance):
    locator = TetLocator(vertices, tetrahedra)
    fibers, outside, ambiguous = [], [], []
    max_error = 0.0
    for fiber in old_fibers:
        tet_ids, barycentric = [], []
        for sample_id, point in enumerate(fiber.rest_points):
            tet_id, bary, score = locator.locate(
                point, tolerance=tolerance,
                candidates=len(tetrahedra))
            if tet_id < 0:
                outside.append({
                    "stream_index": fiber.stream_index,
                    "fiber_index": fiber.fiber_index,
                    "sample_index": sample_id,
                    "minimum_barycentric_score": score,
                })
            elif score <= tolerance:
                ambiguous.append({
                    "stream_index": fiber.stream_index,
                    "fiber_index": fiber.fiber_index,
                    "sample_index": sample_id,
                    "chosen_tet": tet_id,
                })
            tet_ids.append(tet_id)
            barycentric.append(bary)
        tet_ids = np.asarray(tet_ids, dtype=np.int32)
        barycentric = np.asarray(barycentric)
        valid = tet_ids >= 0
        if np.any(valid):
            reconstructed = np.einsum(
                "ni,nij->nj", barycentric[valid],
                vertices[tetrahedra[tet_ids[valid]]])
            max_error = max(max_error, float(np.max(np.linalg.norm(
                reconstructed - fiber.rest_points[valid], axis=1))))
        fibers.append(EmbeddedFiber(
            fiber.stream_index, fiber.fiber_index, tet_ids, barycentric,
            fiber.rest_points.copy(), fiber.rest_segment_lengths.copy()))
    report = {
        "number_of_fibers": len(old_fibers),
        "number_of_samples": int(sum(len(v.rest_points) for v in old_fibers)),
        "successfully_reembedded_samples": int(
            sum(len(v.rest_points) for v in old_fibers) - len(outside)),
        "outside_samples": outside,
        "ambiguous_boundary_samples": ambiguous,
        "maximum_reconstruction_error": max_error,
    }
    return fibers, report


def repair_cavity(vertices, tetrahedra, selection, backend, options):
    boundary_ids = np.unique(selection.boundary_faces)
    global_to_local = {int(v): i for i, v in enumerate(boundary_ids)}
    local_faces = np.vectorize(global_to_local.__getitem__)(
        selection.boundary_faces)
    local_vertices, local_tets = backend.tetrahedralize(
        vertices[boundary_ids], local_faces, options)
    tolerance = float(options.get("boundary_vertex_tolerance", 1e-12))
    if len(local_vertices) < len(boundary_ids) or not np.allclose(
            local_vertices[:len(boundary_ids)], vertices[boundary_ids],
            atol=tolerance, rtol=0.0):
        raise ValueError("backend changed or reordered fixed cavity vertices")
    global_vertices = np.vstack((
        vertices, local_vertices[len(boundary_ids):]))
    local_to_global = np.concatenate((
        boundary_ids,
        np.arange(len(vertices), len(global_vertices), dtype=np.int32)))
    new_cavity_tets = local_to_global[local_tets]
    new_cavity_tets = orient_tetrahedra(global_vertices, new_cavity_tets)
    produced = {
        tuple(sorted(map(int, face)))
        for face in extract_boundary_faces(new_cavity_tets)}
    required = {
        tuple(sorted(map(int, face))) for face in selection.boundary_faces}
    if produced != required:
        raise ValueError(
            "tetrahedralizer did not recover the fixed cavity facets: "
            f"missing={len(required - produced)}, extra={len(produced - required)}")
    keep = np.ones(len(tetrahedra), dtype=bool)
    keep[selection.selected_tets] = False
    all_tets = np.vstack((tetrahedra[keep], new_cavity_tets))
    return global_vertices, all_tets, new_cavity_tets, np.where(keep)[0]


def transfer_raw_metadata(raw, vertices, tetrahedra, source_surface,
                          source_muscle, fibers):
    result = dict(raw)
    result["vertices"] = vertices
    result["tetrahedra"] = tetrahedra
    boundary = extract_boundary_faces(tetrahedra)
    result["sim_faces"] = boundary
    # Fixed external surface means all original authored/render faces and cap
    # indices remain valid and attachment IDs remain exact.
    result["faces"] = np.asarray(raw["faces"]).copy()
    result["render_faces"] = np.asarray(raw["render_faces"]).copy()
    result["_local_remeshing_fiber_embedding"] = [{
        "stream_index": fiber.stream_index,
        "fiber_index": fiber.fiber_index,
        "tet_ids": fiber.tet_ids,
        "barycentric": fiber.barycentric,
        "rest_points": fiber.rest_points,
        "rest_segment_lengths": fiber.rest_segment_lengths,
    } for fiber in fibers]
    return result


def write_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def attachment_transfer_report(old_patches, old_vertices, new_vertices):
    rows = []
    for patch_id, patch in enumerate(old_patches):
        old = old_vertices[patch.vertex_ids]
        new = new_vertices[patch.vertex_ids]
        rows.append({
            "patch_id": patch_id,
            "bone_id": patch.bone_name,
            "constrained_vertex_count": len(patch.vertex_ids),
            "hard_core_vertex_count": len(patch.vertex_ids),
            "transition_vertex_count": 0,
            "centroid_deviation": float(np.linalg.norm(
                np.mean(old, axis=0) - np.mean(new, axis=0))),
            "maximum_positional_deviation": float(np.max(
                np.linalg.norm(old - new, axis=1))),
            "footprint_area_relative_change": 0.0,
            "transfer_method": "preserved_vertex_id_and_surface_geometry",
        })
    return rows


def save_debug_region(output, vertices, tetrahedra, selection):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(
        vertices=vertices, faces=selection.boundary_faces,
        process=False).export(output / "selected_cavity_boundary.ply")
    with (output / "selected_region.vtk").open("w") as handle:
        handle.write("# vtk DataFile Version 2.0\nlocal tet region\nASCII\n")
        handle.write("DATASET UNSTRUCTURED_GRID\n")
        handle.write(f"POINTS {len(vertices)} double\n")
        for point in vertices:
            handle.write(" ".join(map(str, point)) + "\n")
        cells = tetrahedra[selection.selected_tets]
        handle.write(f"CELLS {len(cells)} {5 * len(cells)}\n")
        for tet in cells:
            handle.write("4 " + " ".join(map(str, tet)) + "\n")
        handle.write(f"CELL_TYPES {len(cells)}\n")
        handle.write("10\n" * len(cells))
    (output / "region_report.json").write_text(json.dumps(
        json_ready(selection.report), indent=2))


def export_boundary_problems(path, vertices, faces, report):
    """Export exact invalid edges/faces without attempting tetrahedralization."""
    path = Path(path)
    edges = (
        report["nonmanifold_or_open_edges"]
        + report["orientation_conflict_edges"])
    with path.open("w") as handle:
        handle.write("# Invalid cavity boundary edges\n")
        used = sorted({int(v) for edge in edges for v in edge})
        local = {vertex: i + 1 for i, vertex in enumerate(used)}
        for vertex in used:
            handle.write("v " + " ".join(map(str, vertices[vertex])) + "\n")
        for first, second in edges:
            handle.write(f"l {local[int(first)]} {local[int(second)]}\n")
        for face_id in report["zero_area_face_ids"]:
            face = faces[int(face_id)]
            handle.write("# zero-area face global vertices "
                         + " ".join(map(str, face)) + "\n")


def run_repair(source_path, config, diagnostic_dir, output_dir):
    source_path, output_dir = Path(source_path), Path(output_dir)
    raw = load_raw_mesh(source_path)
    vertices = np.asarray(raw["vertices"], dtype=np.float64)
    tetrahedra = orient_tetrahedra(vertices, raw["tetrahedra"])
    source_surface = extract_boundary_faces(tetrahedra)
    patches = attachment_patches_from_tet(raw)
    # Use the normal loader's already validated embedded samples.
    from viewer.isolated_muscle import load_muscle_data
    source_muscle, _ = load_muscle_data(source_path)
    selection = select_region(
        vertices, tetrahedra, source_surface, config, diagnostic_dir,
        config["local_remeshing"]["target_muscle"], patches,
        source_muscle.fibers)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_debug_region(output_dir / "inspection", vertices, tetrahedra, selection)
    cavity_report = validate_closed_boundary(
        vertices, selection.boundary_faces)
    (output_dir / "cavity_validation.json").write_text(json.dumps(
        json_ready(cavity_report), indent=2))
    if not cavity_report["valid"]:
        attachment_ids = np.unique(np.concatenate([
            patch.vertex_ids for patch in patches])).astype(np.int32)
        contact_points = active_contact_points(
            Path(diagnostic_dir) / "active_contact_multipliers.json",
            config["local_remeshing"]["target_muscle"],
            vertices, source_surface)
        repaired_faces, seam_report = attempt_local_seam_repair(
            vertices, selection.boundary_faces, selection.external_faces,
            np.asarray(raw.get("render_faces", raw.get("faces"))),
            attachment_ids, contact_points,
            config.get("seam_repair", {}))
        (output_dir / "seam_repair_provenance.json").write_text(json.dumps(
            json_ready(seam_report), indent=2))
        if seam_report["accepted"]:
            selection.boundary_faces = repaired_faces
            cavity_report = validate_closed_boundary(
                vertices, selection.boundary_faces)
            (output_dir / "cavity_validation_after_seam_repair.json").write_text(
                json.dumps(json_ready(cavity_report), indent=2))
        else:
            export_boundary_problems(
                output_dir / "invalid_cavity_boundary.obj", vertices,
                selection.boundary_faces, cavity_report)
            raise ValueError(
                "localized seam repair rejected; full-muscle manifold-surface "
                "repair and retetrahedralization is required; see "
                "seam_repair_provenance.json")
    if not cavity_report["valid"]:
        export_boundary_problems(
            output_dir / "invalid_cavity_boundary.obj", vertices,
            selection.boundary_faces, cavity_report)
        raise ValueError("selected cavity boundary is invalid; see cavity_validation.json")
    backend = backend_from_name(config["local_remeshing"]["backend"])
    options = {
        **config["quality_thresholds"],
        **config["local_remeshing"],
    }
    new_vertices, new_tets, cavity_tets, unchanged_ids = repair_cavity(
        vertices, tetrahedra, selection, backend, options)
    new_quality = tet_quality(new_vertices, new_tets)
    cavity_ids = np.arange(
        len(unchanged_ids), len(new_tets), dtype=np.int32)
    cavity_quality = quality_summary(new_quality, cavity_ids)
    hard = float(config["quality_thresholds"][
        "hard_minimum_dihedral_degrees"])
    if cavity_quality["minimum_dihedral_degrees"] < hard:
        raise ValueError(
            f"repaired cavity minimum dihedral "
            f"{cavity_quality['minimum_dihedral_degrees']:.6g} < {hard}")
    new_surface = extract_boundary_faces(new_tets)
    old_surface_keys = {
        tuple(sorted(map(int, f))) for f in source_surface}
    new_surface_keys = {
        tuple(sorted(map(int, f))) for f in new_surface}
    if old_surface_keys != new_surface_keys:
        raise ValueError("fixed-boundary repair changed the external surface")
    fibers, fiber_report = reembed_fibers(
        source_muscle.fibers, new_vertices, new_tets,
        float(config["fiber_transfer"]["containment_tolerance_m"]))
    if (config["fiber_transfer"]["require_all_samples_embedded"]
            and fiber_report["outside_samples"]):
        raise ValueError("fiber samples lie outside repaired volume")
    directions, fiber_valid = build_tet_fiber_directions(
        new_vertices, new_tets, fibers)
    old_directions, old_valid = build_tet_fiber_directions(
        vertices, tetrahedra, source_muscle.fibers)
    interface_old = set(range(len(tetrahedra))) - set(
        selection.selected_tets.tolist())
    old_to_new_tet = {
        int(old): int(new) for new, old in enumerate(unchanged_ids)}
    interface_angles = []
    adjacency = tet_adjacency(new_tets, shared_vertices=3)
    for new_id in range(len(unchanged_ids), len(new_tets)):
        for neighbor in adjacency[new_id]:
            if neighbor < len(unchanged_ids) and fiber_valid[new_id] and fiber_valid[neighbor]:
                interface_angles.append(float(np.degrees(np.arccos(np.clip(
                    abs(np.dot(directions[new_id], directions[neighbor])),
                    -1.0, 1.0)))))
    fiber_report.update({
        "tets_with_no_fiber_direction": int(np.sum(~fiber_valid)),
        "maximum_interface_angular_discontinuity_degrees": max(
            interface_angles or [0.0]),
        "fiber_direction_discontinuity_across_cavity_interface":
            interface_angles,
    })
    deviation = surface_deviation(
        vertices, source_surface, new_vertices, new_surface)
    attachment_report = attachment_transfer_report(
        patches, vertices, new_vertices)
    old_face_id, _, _ = _surface_correspondence(
        new_vertices, new_surface, vertices, source_surface)
    derived = transfer_raw_metadata(
        raw, new_vertices, new_tets, source_surface, source_muscle, fibers)
    asset_path = output_dir / f"{source_path.stem}.npz"
    write_pickle(asset_path, derived)
    provenance = {
        "source_path": str(source_path),
        "source_sha256": source_hash(source_path),
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        "target_asset": str(asset_path),
        "selected_seed_tets": selection.seed_tets,
        "selected_cavity_tets": selection.selected_tets,
        "backend": "tetgen",
        "backend_version": backend.version,
        "python_version": platform.python_version(),
        "tetrahedralizer_parameters": options,
        "deterministic_seed": int(
            config["local_remeshing"]["deterministic_seed"]),
        "quality_before": selection.report["quality_before"],
        "quality_after": cavity_quality,
        "surface_deviation": deviation,
        "attachment_transfer": attachment_report,
        "fiber_transfer": fiber_report,
        "mapping": {
            "old_vertex_to_new_vertex": {
                str(i): i for i in range(len(vertices))},
            "old_tet_to_new_tet": old_to_new_tet,
            "removed_old_tets": selection.selected_tets,
            "new_tet_source_region": {
                str(i): ("unchanged" if i < len(unchanged_ids)
                         else "local_cavity")
                for i in range(len(new_tets))},
            "new_surface_face_to_old_surface_face":
                old_face_id,
            "old_fiber_sample_to_new_tet_embedding": [{
                "stream_index": f.stream_index,
                "fiber_index": f.fiber_index,
                "tet_ids": f.tet_ids,
                "barycentric": f.barycentric,
            } for f in fibers],
            "old_attachment_surface_point_to_new_surface_representation": {
                str(int(v)): int(v) for p in patches for v in p.vertex_ids},
        },
        "contact_metadata": {
            "surface_rebuilt": True,
            "old_contact_primitive_ids_valid": False,
            "old_AL_multipliers_valid": False,
            "required_initialization": "reset_multipliers_and_regenerate_active_set",
        },
        "rest_state": {
            "deformation_gradient": "identity",
            "J": 1.0,
            "transferred_failed_AL_state": False,
        },
    }
    (output_dir / "retetrahedralization_provenance.json").write_text(
        json.dumps(json_ready(provenance), indent=2, sort_keys=True))
    np.savez_compressed(
        output_dir / "fiber_direction_transfer.npz",
        old_directions=old_directions, old_valid=old_valid,
        new_directions=directions, new_valid=fiber_valid)
    return asset_path, provenance
