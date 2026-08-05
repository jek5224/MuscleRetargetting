"""Classified, curve-consistent repair of fibers against an accepted volume.

The anatomical fiber samples are immutable inputs.  This module produces a
separate simulation-rest bundle and an explicit sample-by-sample mapping.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from muscle_sim.cross_sections import (
    extract_section, representative_bundle_path, rotation_minimizing_frames)
from muscle_sim.local_remeshing import json_ready, load_raw_mesh
from viewer.isolated_muscle import (
    EmbeddedFiber, TetLocator, extract_boundary_faces, load_muscle_data)


SURFACE_CLASSES = (
    "INSIDE_ALL_REFERENCE_SURFACES", "OUTSIDE_ORIGINAL_REFERENCE",
    "OUTSIDE_ONLY_C3", "OUTSIDE_PATH_B_AND_C3",
    "AMBIGUOUS_NEAR_BOUNDARY")
_SECTION_CACHE = {}


@dataclass
class RepairRun:
    fiber_id: int
    first: int
    last: int
    classification: str
    policy: str


def load_surface(path, boundary_from_tets=False):
    path = Path(path)
    if boundary_from_tets:
        raw = load_raw_mesh(path)
        vertices = np.asarray(raw["vertices"])
        tetrahedra = np.asarray(raw["tetrahedra"])
        return trimesh.Trimesh(
            vertices=vertices, faces=extract_boundary_faces(tetrahedra),
            process=False)
    if path.suffix.lower() == ".obj":
        return trimesh.load(path, force="mesh", process=False)
    raw = load_raw_mesh(path)
    vertices = np.asarray(raw["vertices"])
    faces = raw.get("render_faces")
    if faces is None:
        faces = raw.get("faces")
    if faces is None:
        faces = extract_boundary_faces(np.asarray(raw["tetrahedra"]))
    return trimesh.Trimesh(
        vertices=vertices, faces=np.asarray(faces), process=False)


def signed_clearance(mesh, points):
    """Positive-inside clearance without trusting winding-number sign."""
    points = np.asarray(points, dtype=np.float64)
    closest, distance, face = trimesh.proximity.closest_point(mesh, points)
    inside = np.asarray(mesh.contains(points), dtype=bool)
    signed = np.where(inside, distance, -distance)
    return signed, closest, np.asarray(face, dtype=np.int64)


def polyline_tangents(points):
    points = np.asarray(points)
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-30)
    return tangent


def contiguous_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    result, first = [], None
    for index, value in enumerate(np.r_[mask, False]):
        if value and first is None:
            first = index
        elif not value and first is not None:
            result.append((first, index - 1))
            first = None
    return result


def classify_run(first, last, count, distances, short_run_max, endpoint_radius):
    length = last - first + 1
    if first == 0 or last == count - 1:
        return "ENDPOINT_OUTSIDE_NEAR_ATTACHMENT"
    if length == 1 and distances[first] <= endpoint_radius:
        return "ISOLATED_NUMERICAL_OUTLIER"
    if length <= short_run_max:
        return "SHORT_CONTIGUOUS_OUTSIDE_RUN"
    return "LONG_CONTIGUOUS_OUTSIDE_RUN"


def surface_comparison_class(inside, distances, ambiguity):
    ref, invalid, path_b, c3 = [bool(value) for value in inside]
    if np.min(distances) <= ambiguity:
        return "AMBIGUOUS_NEAR_BOUNDARY"
    if ref and invalid and path_b and c3:
        return "INSIDE_ALL_REFERENCE_SURFACES"
    if not ref:
        return "OUTSIDE_ORIGINAL_REFERENCE"
    if ref and invalid and path_b and not c3:
        return "OUTSIDE_ONLY_C3"
    if not path_b and not c3:
        return "OUTSIDE_PATH_B_AND_C3"
    return "SURFACE_RECONSTRUCTION_MISMATCH"


def diagnose_fibers(fibers, surfaces, frames, config):
    """Return mandatory per-sample, per-run, and per-surface diagnostics."""
    names = list(surfaces)
    per_surface = {
        name: {"inside_samples": 0, "outside_samples": 0,
               "maximum_outside_distance_m": 0.0}
        for name in names}
    sample_rows, runs = [], []
    short_max = int(config["short_run_max_samples"])
    ambiguity = float(config.get("boundary_ambiguity_m", 2e-6))
    endpoint_radius = float(config.get(
        "isolated_numerical_outlier_m", 5e-5))
    frame_tree = cKDTree(frames["origin"])
    surface_results = {}
    for fiber in fibers:
        points = np.asarray(fiber.rest_points)
        surface_results[fiber.fiber_index] = {}
        for name, mesh in surfaces.items():
            signed, closest, faces = signed_clearance(mesh, points)
            surface_results[fiber.fiber_index][name] = (
                signed, closest, faces)
            outside = signed < -ambiguity
            per_surface[name]["inside_samples"] += int(np.sum(~outside))
            per_surface[name]["outside_samples"] += int(np.sum(outside))
            if np.any(outside):
                per_surface[name]["maximum_outside_distance_m"] = max(
                    per_surface[name]["maximum_outside_distance_m"],
                    float(np.max(-signed[outside])))
        c3_signed, c3_closest, c3_faces = surface_results[
            fiber.fiber_index]["c3"]
        tangents = polyline_tangents(points)
        outside = c3_signed < -ambiguity
        run_lookup = {}
        for first, last in contiguous_runs(outside):
            classification = classify_run(
                first, last, len(points), np.abs(c3_signed),
                short_max, endpoint_radius)
            policy = ("SHORT_RUN_CONSTRAINED" if
                      classification in (
                          "ISOLATED_NUMERICAL_OUTLIER",
                          "SHORT_CONTIGUOUS_OUTSIDE_RUN",
                          "ENDPOINT_OUTSIDE_NEAR_ATTACHMENT")
                      and last - first + 1 <= short_max
                      else "SECTION_COORDINATE")
            run = RepairRun(
                int(fiber.fiber_index), first, last, classification, policy)
            runs.append(asdict(run))
            for index in range(first, last + 1):
                run_lookup[index] = run
        arc = np.r_[0., np.cumsum(np.linalg.norm(
            np.diff(points, axis=0), axis=1))]
        if arc[-1] > 0:
            arc /= arc[-1]
        _, frame_ids = frame_tree.query(points)
        for index in np.flatnonzero(outside):
            face_id = int(c3_faces[index])
            normal = np.asarray(surfaces["c3"].face_normals[face_id])
            all_inside, all_distances = [], []
            containment = {}
            for name in names:
                signed = surface_results[fiber.fiber_index][name][0][index]
                all_inside.append(signed >= -ambiguity)
                all_distances.append(abs(float(signed)))
                containment[name] = {
                    "inside": bool(signed >= -ambiguity),
                    "signed_clearance_m": float(signed)}
            run = run_lookup[index]
            sample_rows.append({
                "fiber_id": int(fiber.fiber_index),
                "sample_index": int(index),
                "sample_position": points[index],
                "closest_C3_surface_triangle": face_id,
                "closest_point": c3_closest[index],
                "unsigned_distance_m": abs(float(c3_signed[index])),
                "signed_clearance_m": float(c3_signed[index]),
                "signed_side_classification": "OUTSIDE",
                "local_surface_normal": normal,
                "fiber_tangent": tangents[index],
                "tangent_normal_angle_degrees": float(np.degrees(np.arccos(
                    np.clip(abs(np.dot(tangents[index], normal)), 0., 1.)))),
                "normalized_arc_length": float(arc[index]),
                "distance_to_origin_attachment_m": float(np.linalg.norm(
                    points[index] - points[0])),
                "distance_to_insertion_attachment_m": float(np.linalg.norm(
                    points[index] - points[-1])),
                "bundle_longitudinal_section_id": int(frame_ids[index]),
                "previous_sample_inside": bool(index > 0 and not outside[index-1]),
                "next_sample_inside": bool(
                    index + 1 < len(points) and not outside[index+1]),
                "outside_run": [run.first, run.last],
                "outside_run_classification": run.classification,
                "sample_role": ("FIBER_ENDPOINT" if index in (
                    0, len(points) - 1) else "MUSCLE_FIBER_INTERIOR"),
                "surface_comparison_class": surface_comparison_class(
                    all_inside, all_distances, ambiguity),
                "surface_containment": containment,
            })
    return {
        "outside_sample_count": len(sample_rows),
        "affected_fiber_count": len(set(
            row["fiber_id"] for row in sample_rows)),
        "sample_rows": sample_rows, "outside_runs": runs,
        "per_surface_statistics": per_surface,
        "surface_order": names,
    }


def ray_polygon_radius(origin, direction, polygon):
    """Distance from origin to first positive 2-D polygon intersection."""
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= 1e-14:
        return None
    direction /= norm
    result = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        matrix = np.column_stack((direction, -edge))
        determinant = np.linalg.det(matrix)
        if abs(determinant) < 1e-14:
            continue
        distance, edge_parameter = np.linalg.solve(matrix, first - origin)
        if distance > 0 and -1e-9 <= edge_parameter <= 1. + 1e-9:
            result.append(float(distance))
    return min(result) if result else None


def section_target(point, surface, frames, margin):
    """Map a point inward while preserving bundle s and contour angle."""
    index = int(np.argmin(np.linalg.norm(
        frames["origin"] - point, axis=1)))
    frame = {key: frames[key][index].copy() for key in (
        "origin", "tangent", "u", "v")}
    longitudinal_offset = float(np.dot(
        point - frame["origin"], frame["tangent"]))
    cache_key = (id(surface), index)
    if cache_key not in _SECTION_CACHE:
        _SECTION_CACHE[cache_key] = extract_section(
            surface, frame, sample_count=192)
    section = _SECTION_CACHE[cache_key]
    # Keep the cached contour in its stable frame, while reconstruction below
    # preserves the continuous longitudinal residual.
    polygon = np.asarray(section["points"])
    center = np.asarray(section["centroid"])
    offset = point - frame["origin"]
    uv = np.asarray([np.dot(offset, frame["u"]),
                     np.dot(offset, frame["v"])])
    radial = uv - center
    boundary_radius = ray_polygon_radius(center, radial, polygon)
    if boundary_radius is None:
        raise ValueError("FIBER_SECTION_CORRESPONDENCE_FAILURE")
    radius = np.linalg.norm(radial)
    target_radius = max(0., boundary_radius - margin)
    scale = min(1., target_radius / max(radius, 1e-30))
    target_uv = center + radial * scale
    source_fraction = float(radius / max(boundary_radius, 1e-30))
    repaired_fraction = float(
        min(radius, target_radius) / max(boundary_radius, 1e-30))
    return (frame["origin"] + longitudinal_offset * frame["tangent"]
            + target_uv[0] * frame["u"] + target_uv[1] * frame["v"]), {
            "frame_id": index, "radial_fraction_original": source_fraction,
                "boundary_radius_m": boundary_radius,
                "target_radial_fraction": repaired_fraction,
                "preserved_longitudinal_offset_m": longitudinal_offset}


def smooth_run(original, targets, first, last, position_weight,
               curvature_weight):
    """Joint quadratic run solve with fixed neighboring anchors."""
    result = targets.copy()
    unknown = np.arange(first, last + 1)
    count = len(unknown)
    if count <= 1:
        return result
    matrix = position_weight * np.eye(count)
    rhs = position_weight * targets[unknown]
    # Penalize changes in displacement, preserving tangent/curvature together.
    difference = np.zeros((count - 1, count))
    difference[np.arange(count - 1), np.arange(count - 1)] = -1.
    difference[np.arange(count - 1), np.arange(1, count)] = 1.
    matrix += curvature_weight * difference.T @ difference
    rhs += curvature_weight * difference.T @ difference @ original[unknown]
    result[unknown] = np.linalg.solve(matrix, rhs)
    return result


def endpoint_run_targets(points, first, last, surface, margin):
    """Move an endpoint run to its cap and inward toward its fiber anchor."""
    targets = np.asarray(points).copy()
    if first == 0 and last + 1 < len(points):
        anchor_id = last + 1
        while (anchor_id < len(points) and signed_clearance(
                surface, np.asarray(points[anchor_id])[None])[0][0] < margin):
            anchor_id += 1
        if anchor_id == len(points):
            raise ValueError("endpoint repair has no interior anchor")
        anchor = np.asarray(points[anchor_id])
        order = range(last, first - 1, -1)
    elif last == len(points) - 1 and first > 0:
        anchor_id = first - 1
        while (anchor_id >= 0 and signed_clearance(
                surface, np.asarray(points[anchor_id])[None])[0][0] < margin):
            anchor_id -= 1
        if anchor_id < 0:
            raise ValueError("endpoint repair has no interior anchor")
        anchor = np.asarray(points[anchor_id])
        order = range(first, last + 1)
    else:
        raise ValueError("endpoint run has no inside anchor")
    metadata = {}
    run_ids = list(order)
    run_points = np.asarray(points)[run_ids]
    _, closest, face_ids = signed_clearance(surface, run_points)
    for index, cap_point, face_id in zip(run_ids, closest, face_ids):
        anchor_direction = anchor - cap_point
        anchor_direction /= max(np.linalg.norm(anchor_direction), 1e-30)
        normal = np.asarray(surface.face_normals[int(face_id)])
        directions = (anchor_direction, -normal, normal)
        feasible = []
        for direction in directions:
            for multiplier in (1.5, 2., 3., 5., 8., 12.):
                candidate = cap_point + direction * (margin * multiplier)
                clearance = signed_clearance(
                    surface, candidate[None])[0][0]
                if clearance >= margin:
                    feasible.append((
                        np.linalg.norm(candidate - points[index]),
                        -clearance, candidate))
                    break
        if not feasible:
            raise ValueError("attachment-cap inward direction unresolved")
        target = min(feasible, key=lambda row: (row[0], row[1]))[2]
        targets[index] = target
        metadata[index] = {
            "attachment_curve_interpolation": True,
            "anchor_sample": int(anchor_id),
            "attachment_cap_point": cap_point}
        anchor = targets[index]
    return targets, metadata


def medial_interior_target(point, surface, frames, margin):
    """Boundary correction directed toward the bundle scaffold, not a vertex."""
    point = np.asarray(point)
    _, closest, face = signed_clearance(surface, point[None])
    cap = closest[0]
    frame_id = int(np.argmin(np.linalg.norm(
        frames["origin"] - point, axis=1)))
    directions = [
        frames["origin"][frame_id] - cap,
        -np.asarray(surface.face_normals[int(face[0])]),
        np.asarray(surface.face_normals[int(face[0])])]
    feasible = []
    for raw_direction in directions:
        direction = np.asarray(raw_direction, dtype=float).copy()
        direction /= max(np.linalg.norm(direction), 1e-30)
        for multiplier in (1.5, 2., 3., 5., 8., 12., 20.):
            candidate = cap + direction * (margin * multiplier)
            clearance = signed_clearance(surface, candidate[None])[0][0]
            if clearance >= margin:
                feasible.append((
                    np.linalg.norm(candidate - point), candidate))
                break
    if not feasible:
        raise ValueError("bundle-medial inward direction unresolved")
    return min(feasible, key=lambda row: row[0])[1]


def repair_fibers(fibers, c3_surface, frames, diagnosis, config, candidate):
    margin = float(config["interior_margin_m"])
    row_by_key = {(row["fiber_id"], row["sample_index"]): row
                  for row in diagnosis["sample_rows"]}
    run_by_fiber = {}
    for run in diagnosis["outside_runs"]:
        run_by_fiber.setdefault(run["fiber_id"], []).append(run)
    output, mappings = [], []
    for fiber in fibers:
        original = np.asarray(fiber.rest_points)
        repaired = original.copy()
        target_meta = {}
        for run in run_by_fiber.get(fiber.fiber_index, []):
            is_short = run["last"] - run["first"] + 1 <= int(
                config["short_run_max_samples"])
            use_run = (
                candidate in ("F2", "F3", "F4")
                or (candidate == "F1" and is_short))
            if not use_run:
                continue
            targets = repaired.copy()
            endpoint = run["first"] == 0 or run["last"] == len(original) - 1
            if endpoint:
                endpoint_targets, endpoint_meta = endpoint_run_targets(
                    original, run["first"], run["last"], c3_surface,
                    margin)
                targets[run["first"]:run["last"] + 1] = endpoint_targets[
                    run["first"]:run["last"] + 1]
                target_meta.update(endpoint_meta)
            else:
                for index in range(run["first"], run["last"] + 1):
                    targets[index], target_meta[index] = section_target(
                        original[index], c3_surface, frames, margin * 1.25)
            if is_short and candidate != "F2":
                repaired = smooth_run(
                    original, targets, run["first"], run["last"],
                    float(config["position_weight"]),
                    float(config["curvature_weight"]))
            else:
                repaired[run["first"]:run["last"] + 1] = targets[
                    run["first"]:run["last"] + 1]
            # The joint solve may relax slightly outward. Reapply a section
            # constraint to the whole run, never an independent surface normal.
            signed = signed_clearance(c3_surface, repaired[
                run["first"]:run["last"] + 1])[0]
            for local in np.flatnonzero(signed < margin):
                index = run["first"] + int(local)
                if endpoint:
                    repaired, endpoint_meta = endpoint_run_targets(
                        repaired, run["first"], run["last"], c3_surface,
                        margin)
                    target_meta.update(endpoint_meta)
                    break
                else:
                    repaired[index], target_meta[index] = section_target(
                        repaired[index], c3_surface, frames,
                        margin * 1.5 + max(0., margin - signed[local]))
        source_parameter = np.arange(len(repaired), dtype=float)
        if candidate == "F4":
            # Keep every source sample, but replace invalid straight chords by
            # a section-following polyline. Added samples have fractional
            # source parameters and therefore retain explicit provenance.
            chord_samples, chord_owners = sample_complete_segments(
                repaired, subdivisions=12)
            chord_signed = signed_clearance(c3_surface, chord_samples)[0]
            bad = set(np.unique(chord_owners[chord_signed < margin]).tolist())
            dense_points, dense_parameter = [], []
            for segment in range(len(repaired) - 1):
                dense_points.append(repaired[segment])
                dense_parameter.append(float(segment))
                if segment not in bad:
                    continue
                for alpha in np.linspace(0., 1., 17)[1:-1]:
                    linear = ((1. - alpha) * repaired[segment]
                              + alpha * repaired[segment + 1])
                    target, _ = section_target(
                        linear, c3_surface, frames, margin * 5.)
                    if signed_clearance(
                            c3_surface, target[None])[0][0] < margin:
                        target = medial_interior_target(
                            linear, c3_surface, frames, margin * 5.)
                    dense_points.append(target)
                    dense_parameter.append(segment + float(alpha))
            dense_points.append(repaired[-1])
            dense_parameter.append(float(len(repaired) - 1))
            repaired = np.asarray(dense_points)
            source_parameter = np.asarray(dense_parameter)
        repaired_lengths = np.linalg.norm(np.diff(repaired, axis=0), axis=1)
        output.append({
            "stream_index": int(fiber.stream_index),
            "fiber_index": int(fiber.fiber_index),
            "points": repaired,
            "anatomical_source_rest_lengths": np.asarray(
                fiber.rest_segment_lengths),
            "simulation_repaired_rest_lengths": repaired_lengths,
            "source_sample_parameter": source_parameter,
        })
        for index, (before, after) in enumerate(zip(original, repaired)):
            row = row_by_key.get((fiber.fiber_index, index), {})
            mappings.append({
                "fiber_id": int(fiber.fiber_index),
                "sample_id": index, "original": before, "repaired": after,
                "displacement_m": float(np.linalg.norm(after - before)),
                "classification": row.get(
                    "outside_run_classification", "UNCHANGED_INSIDE"),
                "sample_role": row.get(
                    "sample_role", "MUSCLE_FIBER_INTERIOR"),
                "repair_policy": (
                    "NONE" if np.array_equal(before, after)
                    else ("SHORT_RUN_CONSTRAINED" if
                          row.get("outside_run_classification") in (
                              "ISOLATED_NUMERICAL_OUTLIER",
                              "SHORT_CONTIGUOUS_OUTSIDE_RUN")
                          else "SECTION_COORDINATE")),
                "section_mapping": target_meta.get(index),
            })
    return output, mappings


def sample_complete_segments(points, subdivisions=12):
    samples, owners = [], []
    alpha = np.linspace(0., 1., subdivisions + 1)[1:-1]
    for segment, (first, second) in enumerate(zip(points[:-1], points[1:])):
        samples.extend((1. - alpha[:, None]) * first
                       + alpha[:, None] * second)
        owners.extend([segment] * len(alpha))
    return np.asarray(samples), np.asarray(owners, dtype=np.int32)


def embed_repaired_fibers(repaired, vertices, tetrahedra, tolerance=2e-8):
    locator = TetLocator(vertices, tetrahedra)
    embedded, outside, minimum_weight, maximum_error = [], [], np.inf, 0.
    for row in repaired:
        ids, weights = [], []
        for sample_id, point in enumerate(row["points"]):
            tet, barycentric, score = locator.locate(
                point, tolerance=tolerance, candidates=128)
            if tet < 0:
                outside.append([row["fiber_index"], sample_id, score])
            ids.append(tet)
            weights.append(barycentric)
            minimum_weight = min(minimum_weight, float(np.min(barycentric)))
        ids, weights = np.asarray(ids), np.asarray(weights)
        if np.all(ids >= 0):
            reconstruction = np.einsum(
                "ni,nij->nj", weights,
                vertices[tetrahedra[ids]])
            maximum_error = max(maximum_error, float(np.max(np.linalg.norm(
                reconstruction - row["points"], axis=1))))
        embedded.append(EmbeddedFiber(
            row["stream_index"], row["fiber_index"], ids, weights,
            np.asarray(row["points"]),
            np.asarray(row["simulation_repaired_rest_lengths"])))
    return embedded, {
        "outside_samples": outside,
        "maximum_barycentric_reconstruction_error_m": maximum_error,
        "minimum_barycentric_weight": minimum_weight,
    }


def fiber_shape_metrics(original_fibers, repaired, surface, config):
    repaired_by_id = {row["fiber_index"]: row for row in repaired}
    rows, all_displacements, outside_segments = [], [], []
    margin = float(config["interior_margin_m"])
    for fiber in original_fibers:
        before = np.asarray(fiber.rest_points)
        after = np.asarray(repaired_by_id[fiber.fiber_index]["points"])
        source_parameter = np.asarray(repaired_by_id[
            fiber.fiber_index].get(
                "source_sample_parameter", np.arange(len(after))))
        source_linear = np.column_stack([
            np.interp(source_parameter, np.arange(len(before)), before[:, axis])
            for axis in range(3)])
        displacement = np.linalg.norm(after - source_linear, axis=1)
        source_mask = np.isclose(source_parameter, np.round(
            source_parameter), atol=1e-12)
        source_displacement = displacement[source_mask]
        all_displacements.extend(source_displacement)
        before_length = np.sum(np.linalg.norm(np.diff(before, axis=0), axis=1))
        after_segments = np.linalg.norm(np.diff(after, axis=0), axis=1)
        after_length = np.sum(after_segments)
        tangent_before = polyline_tangents(before)
        reconstructed_at_source = np.column_stack([
            np.interp(np.arange(len(before)), source_parameter, after[:, axis])
            for axis in range(3)])
        tangent_after = polyline_tangents(reconstructed_at_source)
        tangent_angle = np.degrees(np.arccos(np.clip(np.einsum(
            "ij,ij->i", tangent_before, tangent_after), -1., 1.)))
        segment_samples, owners = sample_complete_segments(after)
        signed = signed_clearance(surface, np.vstack((after, segment_samples)))[0]
        sample_signed = signed[:len(after)]
        segment_signed = signed[len(after):]
        bad_segments = np.unique(owners[segment_signed < margin])
        outside_segments.extend([
            [int(fiber.fiber_index), int(segment)]
            for segment in bad_segments])
        source_lengths = np.asarray(fiber.rest_segment_lengths)
        accumulated = np.zeros_like(source_lengths)
        for value, length in zip(source_parameter[:-1], after_segments):
            accumulated[min(
                int(np.floor(value + 1e-12)), len(accumulated) - 1)] += length
        segment_relative = np.abs(
            accumulated - source_lengths) / np.maximum(source_lengths, 1e-30)
        rows.append({
            "fiber_id": int(fiber.fiber_index),
            "maximum_displacement_m": float(np.max(source_displacement)),
            "RMS_displacement_m": float(np.sqrt(np.mean(
                source_displacement ** 2))),
            "anatomical_source_total_length_m": float(before_length),
            "simulation_repaired_total_length_m": float(after_length),
            "total_length_relative_error": float(
                abs(after_length - before_length) / before_length),
            "maximum_segment_length_relative_error": float(
                np.max(segment_relative)),
            "maximum_tangent_change_degrees": float(np.max(tangent_angle)),
            "minimum_sample_clearance_m": float(np.min(sample_signed)),
            "outside_or_margin_violating_segments": bad_segments,
        })
    gates = {
        "all_required_samples_inside_with_margin": all(
            row["minimum_sample_clearance_m"] >= margin for row in rows),
        "all_required_segments_inside_with_margin": not outside_segments,
        "maximum_sample_displacement": max(
            row["maximum_displacement_m"] for row in rows)
            <= float(config["maximum_sample_displacement_m"]),
        "maximum_total_length_relative_error": max(
            row["total_length_relative_error"] for row in rows)
            <= float(config["maximum_total_length_relative_error"]),
        "maximum_segment_length_relative_error": max(
            row["maximum_segment_length_relative_error"] for row in rows)
            <= float(config["maximum_segment_length_relative_error"]),
        "maximum_tangent_change": max(
            row["maximum_tangent_change_degrees"] for row in rows)
            <= float(config["maximum_tangent_change_degrees"]),
    }
    return {
        "fiber_rows": rows,
        "outside_or_margin_violating_segments": outside_segments,
        "maximum_displacement_m": float(np.max(all_displacements)),
        "p95_displacement_m": float(np.percentile(all_displacements, 95)),
        "RMS_displacement_m": float(np.sqrt(np.mean(
            np.asarray(all_displacements) ** 2))),
        "gates": gates, "accepted": all(gates.values())}


def save_fibers(path, repaired, embedded=None):
    payload = {
        "format_version": 1,
        "fibers": np.asarray(repaired, dtype=object)}
    if embedded is not None:
        payload["embedded_fibers"] = np.asarray(embedded, dtype=object)
    np.savez_compressed(path, **payload)


def load_repaired_fibers(path):
    data = np.load(path, allow_pickle=True)
    return list(data["fibers"]), (
        list(data["embedded_fibers"]) if "embedded_fibers" in data else None)


def write_obj_polylines(path, fibers, original=None):
    vertices, lines = [], []
    groups = []
    for label, rows in (("repaired", fibers), ("original", original or [])):
        for row in rows:
            points = (row["points"] if isinstance(row, dict)
                      else row.rest_points)
            start = len(vertices) + 1
            vertices.extend(points)
            lines.append(np.arange(start, start + len(points)))
            groups.append(label)
    with Path(path).open("w") as handle:
        for vertex in vertices:
            handle.write("v %.17g %.17g %.17g\n" % tuple(vertex))
        for group, line in zip(groups, lines):
            handle.write(f"g {group}\n")
            handle.write("l " + " ".join(map(str, line)) + "\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), indent=2))
if not hasattr(np, "_core"):
    # Object NPZ files produced by NumPy >=2 refer to numpy._core in their
    # pickle stream.  NumPy 1.x exposes the same implementation as numpy.core.
    # Alias it for read compatibility; never rewrite the source fiber asset.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
