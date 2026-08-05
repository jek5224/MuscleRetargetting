"""Validation utilities for the pre-C3 mechanical diagnostic branch."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from muscle_sim.fiber_routing import (
    build_tet_adjacency, traverse_straight_segment)
from muscle_sim.local_remeshing import (
    load_raw_mesh, tet_quality, validate_closed_boundary)
from viewer.isolated_muscle import (
    TetLocator, explicit_fiber_polylines, extract_boundary_faces,
    orient_tetrahedra)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sliver_classification(tetrahedra, quality, threshold=2.):
    bad = set(np.flatnonzero(
        quality["minimum_dihedral_degrees"] < threshold).tolist())
    if not bad:
        return "NONE"
    neighbors, _ = build_tet_adjacency(tetrahedra)
    components = []
    while bad:
        seed = bad.pop()
        stack, component = [seed], {seed}
        while stack:
            current = stack.pop()
            for adjacent in neighbors[current]:
                adjacent = int(adjacent)
                if adjacent in bad:
                    bad.remove(adjacent)
                    component.add(adjacent)
                    stack.append(adjacent)
        components.append(component)
    count = sum(map(len, components))
    if count == 1:
        return "ISOLATED_SLIVER"
    if count <= 16 and len(components) <= 4:
        return "LOCAL_CLUSTER"
    return "GLOBAL_LOW_QUALITY"


def validate_fiber_segments(vertices, tetrahedra, fibers):
    neighbors, _ = build_tet_adjacency(tetrahedra)
    locator = TetLocator(vertices, tetrahedra)
    sample_count = outside = segment_count = valid_segments = 0
    failures = []
    for fiber_id, points in enumerate(fibers):
        points = np.asarray(points)
        sample_count += len(points)
        located = [
            locator.locate(point, tolerance=2e-8, candidates=min(
                512, len(tetrahedra)))[0] for point in points]
        outside += sum(value < 0 for value in located)
        for segment_id, (start, end) in enumerate(zip(
                points[:-1], points[1:])):
            segment_count += 1
            try:
                traverse_straight_segment(
                    vertices, tetrahedra, neighbors, locator, start, end,
                    fiber_id, segment_id)
                valid_segments += 1
            except ValueError as error:
                failures.append({
                    "fiber_id": fiber_id, "source_segment_index": segment_id,
                    "reason": str(error)})
    return {
        "fiber_sample_count": sample_count,
        "outside_sample_count": outside,
        "fiber_segment_count": segment_count,
        "valid_complete_segment_count": valid_segments,
        "embedding_success": outside == 0,
        "complete_segment_containment_success": (
            valid_segments == segment_count),
        "segment_failures": failures}


def inspect_asset(path, expected_fiber_count=25, expected_sample_count=450):
    data = load_raw_mesh(path)
    vertices = np.asarray(data["vertices"], dtype=float)
    raw_tets = np.asarray(data["tetrahedra"], dtype=np.int32)
    tetrahedra = orient_tetrahedra(vertices, raw_tets)
    quality = tet_quality(vertices, tetrahedra)
    fibers = [np.asarray(row[2]) for row in explicit_fiber_polylines(data)]
    fiber_report = validate_fiber_segments(vertices, tetrahedra, fibers)
    boundary = extract_boundary_faces(tetrahedra)
    topology = validate_closed_boundary(vertices, boundary)
    sorted_tets = np.sort(tetrahedra, axis=1)
    duplicates = len(sorted_tets) - len(np.unique(sorted_tets, axis=0))
    attachment_exists = bool(
        len(data.get("anchor_vertices", []))
        and len(data.get("cap_attachments", [])))
    fiber_signature_matches = bool(
        len(fibers) == expected_fiber_count
        and fiber_report["fiber_sample_count"] == expected_sample_count)
    sliver_class = sliver_classification(tetrahedra, quality)
    valid_for_diagnostic = bool(
        np.all(quality["signed_volume"] > 0.)
        and duplicates == 0 and topology["watertight"]
        and topology["manifold"] and attachment_exists
        and fiber_signature_matches and fiber_report["embedding_success"]
        and fiber_report["complete_segment_containment_success"])
    return {
        "tet_mesh_path": str(path), "fiber_path": str(path),
        "attachment_metadata_path": str(path),
        "vertex_count": len(vertices), "tet_count": len(tetrahedra),
        "minimum_dihedral_degrees": float(np.min(
            quality["minimum_dihedral_degrees"])),
        "minimum_signed_volume": float(np.min(quality["signed_volume"])),
        "positive_orientation": bool(np.all(
            quality["signed_volume"] > 0.)),
        "duplicate_tet_count": int(duplicates),
        "boundary_watertight": bool(topology["watertight"]),
        "boundary_manifold": bool(topology["manifold"]),
        "attachment_metadata_exists": attachment_exists,
        "fiber_count": len(fibers),
        "fiber_signature_matches": fiber_signature_matches,
        **fiber_report,
        "tets_below_two_degrees": int(np.sum(
            quality["minimum_dihedral_degrees"] < 2.)),
        "sliver_classification": sliver_class,
        "known_geometry_defects": [
            value for condition, value in (
                (sliver_class != "NONE", sliver_class),
                (not topology["watertight"], "NONWATERTIGHT_BOUNDARY"),
                (not topology["manifold"], "NONMANIFOLD_BOUNDARY"),
                (fiber_report["outside_sample_count"] > 0,
                 "OUTSIDE_SOURCE_SAMPLES"),
                (not fiber_report["complete_segment_containment_success"],
                 "INVALID_SOURCE_SEGMENTS"))
            if condition],
        "sha256": sha256(path),
        "valid_for_diagnostic_fem": valid_for_diagnostic}


def embed_proxy_vertices(vertices, tetrahedra, proxy, maximum_projection):
    locator = TetLocator(vertices, tetrahedra)
    centroids = vertices[tetrahedra].mean(axis=1)
    rows = []
    for index, point in enumerate(np.asarray(proxy)):
        tet, barycentric, score = locator.locate(
            point, tolerance=2e-8, candidates=min(512, len(tetrahedra)))
        projected = False
        distance = 0.
        if tet < 0:
            tet = int(np.argmin(np.linalg.norm(centroids - point, axis=1)))
            q = vertices[tetrahedra[tet]]
            candidate = np.maximum(barycentric, 0.)
            candidate /= max(candidate.sum(), 1e-30)
            reconstructed = candidate @ q
            distance = float(np.linalg.norm(reconstructed - point))
            if distance <= maximum_projection:
                barycentric, projected = candidate, True
            else:
                tet = -1
        rows.append({
            "proxy_vertex_id": index, "tet_id": int(tet),
            "barycentric": barycentric, "projected": projected,
            "projection_distance_m": distance})
    return rows


def proxy_positions(vertices, tetrahedra, tet_ids, barycentric):
    return np.einsum(
        "ni,nij->nj", barycentric, vertices[tetrahedra[tet_ids]])


def transfer_proxy_forces(vertex_count, tetrahedra, tet_ids,
                          barycentric, proxy_forces):
    result = np.zeros((vertex_count, 3), dtype=float)
    for tet, weights, force in zip(tet_ids, barycentric, proxy_forces):
        for vertex, weight in zip(tetrahedra[tet], weights):
            result[vertex] += weight * force
    return result


def contact_mode(mode, proxy_available):
    if mode == "native_boundary":
        return "native_boundary"
    if mode == "embedded_proxy" and proxy_available:
        return "embedded_proxy"
    raise ValueError("requested diagnostic contact mode is unavailable")


def diagnostic_decision(d0=None, d1=None, isolated_failure=None):
    if isolated_failure:
        return "CASE_C_OR_D_ORIGINAL_DISCRETIZATION_BLOCKER"
    if d0 and d0.get("mechanically_promising"):
        return "CASE_A_PHYSICAL_FORMULATION_WORKS"
    if d0 and not d0.get("mechanically_promising") and d1 and d1.get(
            "mechanically_promising"):
        return "CASE_B_NATIVE_CONTACT_BOUNDARY_PROBLEM"
    if d0 and d1 and not d0.get("mechanically_promising") and not d1.get(
            "mechanically_promising"):
        return "CASE_C_SHARED_MESH_OR_MATERIAL_FAILURE"
    return "DIAGNOSTIC_NOT_RUN"
