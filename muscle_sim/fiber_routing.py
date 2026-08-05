"""Exact tet-complex routing for sparse fiber chords.

Each routed subsegment is owned by one convex tetrahedron.  Crossings between
owners are a single barycentric point on their exact shared face, so complete
containment does not depend on surface sampling density.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from muscle_sim.fiber_repair import (
    load_repaired_fibers, load_surface, polyline_tangents, signed_clearance)
from muscle_sim.local_remeshing import json_ready
from viewer.isolated_muscle import EmbeddedFiber, TetLocator


LOCAL_FACES = np.asarray([
    [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
OPPOSITE_TO_LOCAL_FACE = np.asarray([3, 2, 1, 0], dtype=np.int8)


@dataclass
class RoutedSegment:
    fiber_id: int
    source_segment: int
    classification: str
    tet_path: list
    points: np.ndarray
    owner_tets: np.ndarray
    roles: list
    source_parameters: np.ndarray


def build_tet_adjacency(tetrahedra):
    """Return exact face-neighbors and matching local-face IDs.

    The vectorized lexicographic sort avoids a million-entry Python dictionary.
    """
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    count = len(tetrahedra)
    raw = tetrahedra[:, LOCAL_FACES].reshape(-1, 3)
    faces = np.sort(raw, axis=1)
    owners = np.repeat(np.arange(count, dtype=np.int32), 4)
    locals_ = np.tile(np.arange(4, dtype=np.int8), count)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    sorted_faces = faces[order]
    equal = np.all(sorted_faces[1:] == sorted_faces[:-1], axis=1)
    starts = np.flatnonzero(equal)
    # A valid tet mesh has at most two owners; runs of three are rejected.
    triple = equal[:-1] & equal[1:]
    if np.any(triple):
        raise ValueError("INVALID_TET_ADJACENCY: nonmanifold tet face")
    neighbor = np.full((count, 4), -1, dtype=np.int32)
    neighbor_local = np.full((count, 4), -1, dtype=np.int8)
    for first_position in starts:
        first = order[first_position]
        second = order[first_position + 1]
        a, b = owners[first], owners[second]
        la, lb = locals_[first], locals_[second]
        neighbor[a, la], neighbor[b, lb] = b, a
        neighbor_local[a, la], neighbor_local[b, lb] = lb, la
    return neighbor, neighbor_local


def shared_face(tetrahedra, first, second):
    shared = sorted(set(map(int, tetrahedra[first])).intersection(
        map(int, tetrahedra[second])))
    if len(shared) != 3:
        raise ValueError("INVALID_TET_ADJACENCY: path edge lacks shared face")
    return np.asarray(shared, dtype=np.int32)


def tet_barycentric(vertices, tetrahedra, tet_id, point):
    q = np.asarray(vertices)[np.asarray(tetrahedra)[tet_id]]
    matrix = np.column_stack((q[0] - q[3], q[1] - q[3], q[2] - q[3]))
    local = np.linalg.solve(matrix, np.asarray(point) - q[3])
    return np.r_[local, 1. - np.sum(local)]


def face_barycentric(triangle, point):
    triangle = np.asarray(triangle)
    basis = np.column_stack((triangle[0] - triangle[2],
                             triangle[1] - triangle[2]))
    local = np.linalg.lstsq(basis, np.asarray(point) - triangle[2],
                            rcond=None)[0]
    return np.r_[local, 1. - np.sum(local)]


def closest_point_triangle(triangle, point):
    return trimesh.triangles.closest_point(
        np.asarray(triangle)[None], np.asarray(point)[None])[0]


def classify_straight_segment(surface, start, end, margin, samples=129):
    alpha = np.linspace(0., 1., samples)
    points = (1. - alpha[:, None]) * start + alpha[:, None] * end
    signed, closest, faces = signed_clearance(surface, points)
    outside = signed < margin
    runs = []
    first = None
    for index, value in enumerate(np.r_[outside, False]):
        if value and first is None:
            first = index
        elif not value and first is not None:
            runs.append({
                "alpha_first": float(alpha[first]),
                "alpha_last": float(alpha[index - 1]),
                "maximum_outside_depth_m": float(max(
                    0., -np.min(signed[first:index]))),
                "crossed_surface_triangles": sorted(set(map(
                    int, faces[first:index]))),
            })
            first = None
    maximum_depth = float(max(0., -np.min(signed)))
    if not runs:
        classification = "STRAIGHT_SEGMENT_VALID"
    elif len(runs) > 1:
        classification = "MULTIPLE_SURFACE_CROSSINGS"
    elif maximum_depth <= 5e-5:
        classification = "SHALLOW_CHORD_ESCAPE"
    else:
        classification = "CURVED_BOUNDARY_FOLLOWING_REQUIRED"
    return {
        "classification": classification,
        "valid": not runs, "outside_intervals": runs,
        "start_anchor_inside": bool(signed[0] >= margin),
        "end_anchor_inside": bool(signed[-1] >= margin),
        "both_endpoints_valid_interior_anchors": bool(
            signed[0] >= margin and signed[-1] >= margin),
        "maximum_outside_depth_m": maximum_depth,
        "minimum_signed_clearance_m": float(np.min(signed)),
        "sampled_points": points, "sampled_signed_clearance": signed,
        "sampled_closest_points": closest,
    }


class ClearanceField:
    """Conservative-enough geometric cost field with exact point validation."""
    def __init__(self, surface):
        self.surface = surface
        self.tree = cKDTree(np.asarray(surface.vertices))
        self.cache = {}

    def approximate(self, points):
        return self.tree.query(np.asarray(points))[0]

    def exact(self, point):
        key = tuple(np.round(np.asarray(point), 12))
        if key not in self.cache:
            self.cache[key] = float(signed_clearance(
                self.surface, np.asarray(point)[None])[0][0])
        return self.cache[key]


def edge_cost(first_point, second_point, source_start, source_end,
              source_direction, clearance, settings):
    midpoint = .5 * (first_point + second_point)
    edge = second_point - first_point
    length = np.linalg.norm(edge)
    if length <= 1e-14:
        return np.inf
    chord = source_end - source_start
    chord_squared = max(float(np.dot(chord, chord)), 1e-30)
    alpha = np.clip(np.dot(
        midpoint - source_start, chord) / chord_squared, 0., 1.)
    source_point = (1. - alpha) * source_start + alpha * source_end
    source_deviation = np.linalg.norm(midpoint - source_point)
    alignment = 1. - abs(float(np.dot(
        edge / length, source_direction)))
    preferred = float(settings["preferred_clearance_m"])
    boundary = max(0., preferred - clearance) / max(preferred, 1e-30)
    # Every geometric term is integrated over edge length.  The old
    # source-deviation term was charged once per dual-graph edge, making the
    # answer depend on tet density and encouraging long, oscillatory detours
    # through coarse regions.
    corridor = max(float(settings.get(
        "maximum_route_corridor_radius_m", np.linalg.norm(chord))), 1e-30)
    return (
        length
        + float(settings["source_deviation_weight"]) * length
        * source_deviation / corridor
        + float(settings["tangent_weight"]) * length * alignment
        + float(settings["boundary_penalty_weight"]) * length * boundary ** 2)


def source_curve_prior(points, samples_per_segment=16):
    """Orientation-consistent cubic Hermite prior through sparse anchors."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points.copy()
    tangents = polyline_tangents(points)
    result = []
    for index, (first, second) in enumerate(zip(points[:-1], points[1:])):
        length = np.linalg.norm(second - first)
        for u in np.linspace(0., 1., samples_per_segment, endpoint=False):
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h10 = u ** 3 - 2 * u ** 2 + u
            h01 = -2 * u ** 3 + 3 * u ** 2
            h11 = u ** 3 - u ** 2
            result.append(h00 * first + h10 * length * tangents[index]
                          + h01 * second + h11 * length * tangents[index + 1])
    result.append(points[-1])
    return np.asarray(result)


def endpoint_tangent_errors(route_points, source_points, segment_index):
    """Return oriented start/end tangent errors in degrees."""
    route_points = np.asarray(route_points)
    source_points = np.asarray(source_points)
    source_tangents = polyline_tangents(source_points)
    route_directions = np.diff(route_points, axis=0)
    route_directions /= np.maximum(
        np.linalg.norm(route_directions, axis=1)[:, None], 1e-30)
    desired = (source_tangents[segment_index],
               source_tangents[segment_index + 1])
    actual = (route_directions[0], route_directions[-1])
    return tuple(float(np.degrees(np.arccos(np.clip(
        np.dot(a, b), -1., 1.)))) for a, b in zip(actual, desired))


def astar_tet_path(vertices, tetrahedra, neighbors, start_tet, end_tet,
                   source_start, source_end, clearance_field, settings):
    """Deterministic A* over face adjacency."""
    source_start = np.asarray(source_start, dtype=float)
    source_end = np.asarray(source_end, dtype=float)
    if start_tet == end_tet:
        return [int(start_tet)]
    centroids = settings.get("_tet_centroids")
    if centroids is None:
        centroids = np.asarray(vertices)[np.asarray(tetrahedra)].mean(axis=1)
    direction = source_end - source_start
    direction /= max(np.linalg.norm(direction), 1e-30)
    margin = float(settings["interior_margin_m"])
    best = {int(start_tet): 0.}
    parent = {}
    heap = [(float(np.linalg.norm(
        centroids[start_tet] - centroids[end_tet])), 0., int(start_tet))]
    expanded = 0
    maximum = int(settings.get("maximum_astar_expansions", 200000))
    while heap:
        _, cost, current = heapq.heappop(heap)
        if cost != best.get(current):
            continue
        if current == end_tet:
            result = [current]
            while current != start_tet:
                current = parent[current]
                result.append(current)
            return result[::-1]
        expanded += 1
        if expanded > maximum:
            break
        for adjacent in sorted(int(value) for value in neighbors[current]
                               if value >= 0):
            chord = source_end - source_start
            chord_squared = max(float(np.dot(chord, chord)), 1e-30)
            alpha = np.clip(np.dot(
                centroids[adjacent] - source_start, chord) / chord_squared,
                0., 1.)
            source_projection = source_start + alpha * chord
            if np.linalg.norm(
                    centroids[adjacent] - source_projection) > float(
                        settings.get(
                            "maximum_route_corridor_radius_m", np.inf)):
                continue
            face = shared_face(tetrahedra, current, adjacent)
            face_center = np.mean(np.asarray(vertices)[face], axis=0)
            approximate_clearance = float(
                clearance_field.approximate(face_center[None])[0])
            if approximate_clearance < margin * .5:
                continue
            step = edge_cost(
                centroids[current], centroids[adjacent],
                source_start, source_end, direction,
                approximate_clearance, settings)
            candidate = cost + step
            if candidate < best.get(adjacent, np.inf) - 1e-15:
                best[adjacent] = candidate
                parent[adjacent] = current
                heuristic = np.linalg.norm(
                    centroids[adjacent] - centroids[end_tet])
                heapq.heappush(
                    heap, (candidate + heuristic, candidate, adjacent))
    raise ValueError("NO_INTERIOR_TET_PATH")


def crossing_points(vertices, tetrahedra, tet_path, start, end,
                    clearance_field, settings, mode="projected"):
    """Create one common point on every traversed shared face."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if len(tet_path) == 1:
        return np.vstack((start, end)), []
    centroids = np.asarray(vertices)[np.asarray(tetrahedra)[tet_path]].mean(
        axis=1)
    progress = np.r_[0., np.cumsum(np.linalg.norm(
        np.diff(centroids, axis=0), axis=1))]
    if progress[-1] > 0:
        progress /= progress[-1]
    result = [np.asarray(start)]
    rows = []
    margin = float(settings["interior_margin_m"])
    for index, (first, second) in enumerate(zip(
            tet_path[:-1], tet_path[1:])):
        face = shared_face(tetrahedra, first, second)
        triangle = np.asarray(vertices)[face]
        target = ((1. - progress[index + 1]) * start
                  + progress[index + 1] * end)
        if mode == "centroid":
            point = np.mean(triangle, axis=0)
        else:
            point = closest_point_triangle(triangle, target)
            # Pull a boundary-edge projection toward the face centroid without
            # leaving the shared face.
            bary = face_barycentric(triangle, point)
            if np.min(bary) < .05:
                bary = .9 * bary + .1 / 3.
                point = bary @ triangle
        clearance = float(clearance_field.approximate(point[None])[0])
        if clearance < margin:
            centroid = np.mean(triangle, axis=0)
            alternatives = [
                (1. - alpha) * point + alpha * centroid
                for alpha in (.1, .25, .5, .75, 1.)]
            valid = [(float(clearance_field.approximate(value[None])[0]), value)
                     for value in alternatives]
            valid = [value for value in valid if value[0] >= margin]
            if not valid:
                raise ValueError("CLEARANCE_FIELD_BLOCKS_ROUTE")
            clearance, point = max(valid, key=lambda value: value[0])
        barycentric = face_barycentric(triangle, point)
        result.append(point)
        rows.append({
            "from_tet": int(first), "to_tet": int(second),
            "face_vertex_ids": face, "face_barycentric": barycentric,
            "clearance_m": clearance})
    result.append(np.asarray(end))
    return np.asarray(result), rows


def optimize_face_crossings(vertices, tetrahedra, tet_path, points,
                            clearance_field, settings, iterations=30):
    """Shorten/smooth a portal route while each crossing stays on its face."""
    points = np.asarray(points).copy()
    if len(points) <= 2:
        return points
    faces = [
        shared_face(tetrahedra, first, second)
        for first, second in zip(tet_path[:-1], tet_path[1:])]
    margin = float(settings["interior_margin_m"])
    source_start, source_end = points[0].copy(), points[-1].copy()
    for _ in range(iterations):
        previous = points.copy()
        for index, face in enumerate(faces, start=1):
            triangle = np.asarray(vertices)[face]
            neighbor_midpoint = .5 * (points[index - 1] + points[index + 1])
            source_alpha = index / (len(points) - 1)
            source_target = (
                (1. - source_alpha) * source_start + source_alpha * source_end)
            target = .8 * neighbor_midpoint + .2 * source_target
            candidate = closest_point_triangle(triangle, target)
            barycentric = face_barycentric(triangle, candidate)
            # Avoid edge/vertex portal degeneracies and the associated route
            # reversals while remaining exactly on the shared face.
            if np.min(barycentric) < .01:
                barycentric = .97 * barycentric + .03 / 3.
                candidate = barycentric @ triangle
            if float(clearance_field.approximate(
                    candidate[None])[0]) >= margin:
                points[index] = candidate
        if np.max(np.linalg.norm(points - previous, axis=1)) < 1e-10:
            break
    return points


def validate_exact_route(vertices, tetrahedra, tet_path, points,
                         tolerance=2e-9):
    if len(points) != len(tet_path) + 1:
        return {"valid": False, "reason": "point/owner count mismatch"}
    rows, valid = [], True
    for index, tet_id in enumerate(tet_path):
        first = tet_barycentric(
            vertices, tetrahedra, tet_id, points[index])
        second = tet_barycentric(
            vertices, tetrahedra, tet_id, points[index + 1])
        local_valid = bool(
            np.min(first) >= -tolerance and np.min(second) >= -tolerance)
        valid &= local_valid
        rows.append({
            "owner_tet": int(tet_id), "start_barycentric": first,
            "end_barycentric": second, "valid": local_valid})
    transitions = all(
        first == second or len(set(tetrahedra[first]).intersection(
            tetrahedra[second])) == 3
        for first, second in zip(tet_path[:-1], tet_path[1:]))
    return {
        "valid": bool(valid and transitions),
        "subsegments": rows, "face_adjacency_valid": transitions}


def refine_route(points, owners, source_parameters, maximum_length,
                 maximum_turn_degrees, maximum_inserted=None):
    """Subdivide within owner tets; subdivision preserves exact containment."""
    points = np.asarray(points)
    result, result_owners, result_parameter = [points[0]], [], [
        source_parameters[0]]
    requested = [
        max(1, int(np.ceil(np.linalg.norm(points[index + 1] - points[index])
                           / maximum_length)))
        for index in range(len(owners))]
    mandatory_inserted = len(points) - 2
    requested_inserted = sum(requested) - 1
    if maximum_inserted is not None and requested_inserted > maximum_inserted:
        budget = max(0, int(maximum_inserted) - mandatory_inserted)
        requested = [1] * len(owners)
        # Deterministic longest-first allocation. Face crossings are mandatory;
        # optional within-tet subdivisions consume only the remaining budget.
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        for index in np.argsort(-lengths, kind="stable"):
            need = max(0, int(np.ceil(lengths[index] / maximum_length)) - 1)
            take = min(need, budget)
            requested[index] += take
            budget -= take
            if budget == 0:
                break
    for index, owner in enumerate(owners):
        length = np.linalg.norm(points[index + 1] - points[index])
        pieces = requested[index]
        for piece in range(1, pieces + 1):
            alpha = piece / pieces
            result.append(
                (1. - alpha) * points[index] + alpha * points[index + 1])
            result_parameter.append(
                (1. - alpha) * source_parameters[index]
                + alpha * source_parameters[index + 1])
            result_owners.append(owner)
    return (np.asarray(result), np.asarray(result_owners, dtype=np.int32),
            np.asarray(result_parameter))


def route_length(points):
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def maximum_turn(points):
    direction = np.diff(points, axis=0)
    direction /= np.maximum(np.linalg.norm(
        direction, axis=1)[:, None], 1e-30)
    if len(direction) < 2:
        return 0.
    return float(np.max(np.degrees(np.arccos(np.clip(np.einsum(
        "ij,ij->i", direction[:-1], direction[1:]), -1., 1.)))))


def route_source_segment(vertices, tetrahedra, neighbors, locator,
                         clearance_field, start, end, fiber_id, segment_id,
                         classification, settings, candidate):
    first_tet = locator.locate(
        start, tolerance=2e-8, candidates=512)[0]
    last_tet = locator.locate(
        end, tolerance=2e-8, candidates=512)[0]
    if first_tet < 0 or last_tet < 0:
        raise ValueError("ATTACHMENT_ENDPOINT_INCOMPATIBLE")
    path = astar_tet_path(
        vertices, tetrahedra, neighbors, first_tet, last_tet,
        start, end, clearance_field, settings)
    mode = "centroid" if candidate == "R1" else "projected"
    points, crossings = crossing_points(
        vertices, tetrahedra, path, start, end,
        clearance_field, settings, mode=mode)
    if candidate in ("R2", "R3", "R4", "R5"):
        for row, point in zip(crossings, points[1:-1]):
            row["initial_position"] = np.asarray(point).copy()
            row["initial_face_barycentric"] = np.asarray(
                row["face_barycentric"]).copy()
        points = optimize_face_crossings(
            vertices, tetrahedra, path, points,
            clearance_field, settings)
        for row, point in zip(crossings, points[1:-1]):
            triangle = vertices[np.asarray(row["face_vertex_ids"])]
            row["face_barycentric"] = face_barycentric(triangle, point)
            row["optimized"] = True
    parameters = np.linspace(0., 1., len(points))
    if candidate == "R5":
        points, owners, parameters = refine_route(
            points, path, parameters,
            float(settings["maximum_subsegment_length_m"]),
            float(settings["maximum_turn_angle_degrees"]),
            int(settings["maximum_inserted_samples_per_source_segment"]))
    else:
        owners = np.asarray(path, dtype=np.int32)
    validation = validate_exact_route(
        vertices, tetrahedra, owners, points)
    if not validation["valid"]:
        raise ValueError("INVALID_TET_ADJACENCY")
    return RoutedSegment(
        fiber_id, segment_id, classification, list(map(int, path)),
        points, owners, (
            ["REPAIRED_ORIGINAL_ANCHOR"]
            + ["INSERTED_INTERIOR_SAMPLE"] * (len(points) - 2)
            + ["REPAIRED_ORIGINAL_ANCHOR"]),
        parameters), crossings, validation


def traverse_straight_segment(vertices, tetrahedra, neighbors, locator,
                              start, end, fiber_id, segment_id,
                              tolerance=2e-9):
    """Walk a valid chord through exact tet faces without graph search."""
    current = locator.locate(
        start, tolerance=2e-8, candidates=512)[0]
    end_tet = locator.locate(
        end, tolerance=2e-8, candidates=512)[0]
    if current < 0 or end_tet < 0:
        raise ValueError("ATTACHMENT_ENDPOINT_INCOMPATIBLE")
    points, path, crossings, parameters = [np.asarray(start)], [], [], [0.]
    alpha_global = 0.
    for _ in range(10000):
        path.append(int(current))
        end_bary = tet_barycentric(
            vertices, tetrahedra, current, end)
        if np.min(end_bary) >= -tolerance:
            points.append(np.asarray(end))
            parameters.append(1.)
            break
        here = points[-1]
        start_bary = tet_barycentric(
            vertices, tetrahedra, current, here)
        delta = end_bary - start_bary
        candidates = [
            (-start_bary[index] / delta[index], index)
            for index in range(4) if delta[index] < -1e-14
            and -start_bary[index] / delta[index] > 1e-10]
        if not candidates:
            raise ValueError("INVALID_TET_ADJACENCY: chord walk stalled")
        local_alpha, opposite = min(candidates)
        point = (1. - local_alpha) * here + local_alpha * end
        new_alpha = alpha_global + (1. - alpha_global) * local_alpha
        local_face = int(OPPOSITE_TO_LOCAL_FACE[opposite])
        adjacent = int(neighbors[current, local_face])
        if adjacent < 0:
            raise ValueError("straight-valid chord reached boundary")
        face = np.sort(tetrahedra[current, LOCAL_FACES[local_face]])
        triangle = vertices[face]
        bary = face_barycentric(triangle, point)
        points.append(point)
        parameters.append(new_alpha)
        crossings.append({
            "from_tet": int(current), "to_tet": adjacent,
            "face_vertex_ids": face, "face_barycentric": bary})
        current = adjacent
        alpha_global = new_alpha
    else:
        raise ValueError("INVALID_TET_ADJACENCY: chord walk exceeded limit")
    points = np.asarray(points)
    validation = validate_exact_route(
        vertices, tetrahedra, path, points)
    if not validation["valid"]:
        raise ValueError("INVALID_TET_ADJACENCY: chord validation")
    return RoutedSegment(
        fiber_id, segment_id, "STRAIGHT_SEGMENT_VALID", path, points,
        np.asarray(path, dtype=np.int32),
        ["REPAIRED_ORIGINAL_ANCHOR"]
        + ["INSERTED_INTERIOR_SAMPLE"] * (len(points) - 2)
        + ["REPAIRED_ORIGINAL_ANCHOR"],
        np.asarray(parameters)), crossings, validation


def assemble_fiber(fiber_id, anchors, routed_segments):
    points, roles, source_parameters, owner_tets = [], [], [], []
    for segment_id, route in enumerate(routed_segments):
        if segment_id == 0:
            points.extend(route.points)
            roles.extend(route.roles)
            source_parameters.extend(
                segment_id + route.source_parameters)
        else:
            points.extend(route.points[1:])
            roles.extend(route.roles[1:])
            source_parameters.extend(
                segment_id + route.source_parameters[1:])
        owner_tets.extend(route.owner_tets)
    return {
        "fiber_index": int(fiber_id), "points": np.asarray(points),
        "roles": roles, "source_sample_parameter": np.asarray(
            source_parameters),
        "subsegment_owner_tets": np.asarray(owner_tets, dtype=np.int32),
        "source_anchor_points": np.asarray(anchors),
        "source_polyline_length": route_length(anchors),
        "embedded_route_length": route_length(points)}


def embed_routed_fiber(route, vertices, tetrahedra):
    points = np.asarray(route["points"])
    owners = np.asarray(route["subsegment_owner_tets"])
    point_tets = np.r_[owners, owners[-1]]
    weights = np.asarray([
        tet_barycentric(vertices, tetrahedra, int(tet), point)
        for tet, point in zip(point_tets, points)])
    reconstruction = np.einsum(
        "ni,nij->nj", weights, vertices[tetrahedra[point_tets]])
    return EmbeddedFiber(
        0, int(route["fiber_index"]), point_tets, weights, points,
        np.linalg.norm(np.diff(points, axis=0), axis=1)), {
            "maximum_reconstruction_error_m": float(np.max(np.linalg.norm(
                reconstruction - points, axis=1))),
            "minimum_barycentric_weight": float(np.min(weights))}


def introduced_crossings(routes, minimum_spacing):
    """Detect new non-neighbor segment approaches using a midpoint tree."""
    segments, labels, source_segments = [], [], []
    for route in routes:
        points = np.asarray(route["points"])
        parameters = np.asarray(route["source_sample_parameter"])
        for index, pair in enumerate(zip(points[:-1], points[1:])):
            segments.append(pair)
            labels.append((route["fiber_index"], index))
            source_segments.append(int(np.floor(parameters[index] + 1e-12)))
    segments = np.asarray(segments)
    midpoint = segments.mean(axis=1)
    radius = .5 * np.linalg.norm(
        segments[:, 1] - segments[:, 0], axis=1) + minimum_spacing
    tree = cKDTree(midpoint)
    rows = []
    for index, candidates in enumerate(tree.query_ball_point(
            midpoint, radius + np.max(radius))):
        for other in candidates:
            if other <= index or labels[index][0] == labels[other][0]:
                continue
            first, second = segments[index], segments[other]
            distance = segment_segment_distance(first, second)
            if distance < minimum_spacing:
                rows.append({
                    "first": labels[index], "second": labels[other],
                    "distance_m": distance})
    return rows


def segment_segment_distance(first, second):
    """Exact closest distance between two 3-D closed line segments."""
    p0, p1 = np.asarray(first, dtype=float)
    q0, q1 = np.asarray(second, dtype=float)
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    d, e = np.dot(u, w), np.dot(v, w)
    denominator = a * c - b * b
    s_n, s_d = denominator, denominator
    t_n, t_d = denominator, denominator
    if denominator < 1e-30:
        s_n, s_d, t_n, t_d = 0., 1., e, c
    else:
        s_n, t_n = b * e - c * d, a * e - b * d
        if s_n < 0.:
            s_n, t_n, t_d = 0., e, c
        elif s_n > s_d:
            s_n, t_n, t_d = s_d, e + b, c
    if t_n < 0.:
        t_n = 0.
        s_n = np.clip(-d, 0., a)
        s_d = a
    elif t_n > t_d:
        t_n = t_d
        s_n = np.clip(b - d, 0., a)
        s_d = a
    s = 0. if abs(s_n) < 1e-30 else s_n / max(s_d, 1e-30)
    t = 0. if abs(t_n) < 1e-30 else t_n / max(t_d, 1e-30)
    return float(np.linalg.norm(w + s * u - t * v))


def arc_length_weighted_tet_directions(routes, vertices, tetrahedra):
    """Accumulate anisotropy without weighting densely sampled routes more."""
    sums = np.zeros((len(tetrahedra), 3), dtype=float)
    covered = np.zeros(len(tetrahedra), dtype=float)
    for route in routes:
        delta = np.diff(np.asarray(route["points"]), axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        directions = delta / np.maximum(lengths[:, None], 1e-30)
        for tet, direction, length in zip(
                route["subsegment_owner_tets"], directions, lengths):
            # Axial directions are sign-equivalent. Align to the accumulated
            # direction to prevent cancellation.
            if np.dot(sums[tet], direction) < 0:
                direction = -direction
            sums[tet] += length * direction
            covered[tet] += length
    norms = np.linalg.norm(sums, axis=1)
    result = np.zeros_like(sums)
    valid = norms > 0
    result[valid] = sums[valid] / norms[valid, None]
    return result, covered


def save_routed_fibers(path, routes, embedded, provenance):
    np.savez_compressed(
        path, routes=np.asarray(routes, dtype=object),
        embedded_fibers=np.asarray(embedded, dtype=object))
    Path(path).with_name("fiber_route_provenance.json").write_text(
        json.dumps(json_ready(provenance), indent=2))


def load_routed_fibers(path):
    data = np.load(path, allow_pickle=True)
    return list(data["routes"]), list(data["embedded_fibers"])
