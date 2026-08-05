"""Source-corridor routing and continuous shared-face portal optimization."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from muscle_sim.fiber_repair import polyline_tangents
from muscle_sim.fiber_routing import (
    face_barycentric, maximum_turn, route_length, shared_face,
    traverse_straight_segment)


def sample_polyline(points, maximum_spacing):
    points = np.asarray(points, dtype=float)
    result, parameters = [], []
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = max(float(lengths.sum()), 1e-30)
    before = 0.
    for index, length in enumerate(lengths):
        pieces = max(1, int(np.ceil(length / maximum_spacing)))
        for piece in range(pieces):
            alpha = piece / pieces
            result.append((1. - alpha) * points[index] + alpha * points[index + 1])
            parameters.append((before + alpha * length) / total)
        before += length
    result.append(points[-1])
    parameters.append(1.)
    return np.asarray(result), np.asarray(parameters)


def tet_source_coordinates(vertices, tetrahedra, source_points):
    """Distance, ordered curve coordinate, and tangent at each tet centroid."""
    centroids = np.asarray(vertices)[np.asarray(tetrahedra)].mean(axis=1)
    sampled, parameters = sample_polyline(source_points, max(
        route_length(source_points) / 1024., 1e-5))
    tree = cKDTree(sampled)
    distance, nearest = tree.query(centroids)
    tangents = polyline_tangents(sampled)
    return distance, parameters[nearest], tangents[nearest]


def source_corridor(vertices, tetrahedra, source_points, clearance,
                    radius, minimum_clearance, start_tet=None, end_tet=None):
    distance, coordinate, tangent = tet_source_coordinates(
        vertices, tetrahedra, source_points)
    allowed = (distance <= radius) & (
        np.asarray(clearance) >= minimum_clearance)
    if start_tet is not None:
        allowed[int(start_tet)] = True
    if end_tet is not None:
        allowed[int(end_tet)] = True
    return allowed, {
        "distance_to_source_m": distance,
        "longitudinal_coordinate": coordinate,
        "source_tangent": tangent}


def corridor_connected(neighbors, allowed, start, end):
    if not allowed[start] or not allowed[end]:
        return False
    seen, stack = {int(start)}, [int(start)]
    while stack:
        current = stack.pop()
        if current == end:
            return True
        for adjacent in neighbors[current]:
            adjacent = int(adjacent)
            if adjacent >= 0 and allowed[adjacent] and adjacent not in seen:
                seen.add(adjacent)
                stack.append(adjacent)
    return False


def expand_corridor(vertices, tetrahedra, neighbors, source_points,
                    clearance, start, end, settings):
    radius = float(settings["initial_radius_m"])
    maximum = float(settings["maximum_radius_m"])
    growth = float(settings["radius_growth_factor"])
    attempts = []
    while True:
        allowed, fields = source_corridor(
            vertices, tetrahedra, source_points, clearance, radius,
            float(settings["minimum_clearance_m"]), start, end)
        connected = corridor_connected(neighbors, allowed, start, end)
        attempts.append({"radius_m": radius, "tet_count": int(allowed.sum()),
                         "connected": connected})
        if connected:
            return allowed, fields, radius, attempts
        if radius >= maximum:
            break
        radius = min(radius * growth, maximum)
    raise ValueError("SOURCE_CORRIDOR_DISCONNECTED")


def monotonic_step_allowed(first_s, second_s, maximum_backward_s):
    return second_s >= first_s - maximum_backward_s


def corridor_astar(vertices, tetrahedra, neighbors, allowed, coordinates,
                   start, end, maximum_backward_s, edge_penalty=None):
    centroids = np.asarray(vertices)[np.asarray(tetrahedra)].mean(axis=1)
    edge_penalty = edge_penalty or {}
    best, parent = {int(start): 0.}, {}
    heap = [(float(np.linalg.norm(centroids[start] - centroids[end])),
             0., int(start))]
    while heap:
        _, cost, current = heapq.heappop(heap)
        if cost != best.get(current):
            continue
        if current == end:
            path = [current]
            while current != start:
                current = parent[current]
                path.append(current)
            return path[::-1]
        for adjacent in sorted(int(x) for x in neighbors[current] if x >= 0):
            if not allowed[adjacent] or not monotonic_step_allowed(
                    coordinates[current], coordinates[adjacent],
                    maximum_backward_s):
                continue
            edge = (min(current, adjacent), max(current, adjacent))
            step = np.linalg.norm(centroids[adjacent] - centroids[current])
            step *= 1. + edge_penalty.get(edge, 0.)
            candidate = cost + step
            if candidate < best.get(adjacent, np.inf) - 1e-15:
                best[adjacent], parent[adjacent] = candidate, current
                heuristic = np.linalg.norm(centroids[adjacent] - centroids[end])
                heapq.heappush(
                    heap, (candidate + heuristic, candidate, adjacent))
    raise ValueError("SOURCE_CORRIDOR_DISCONNECTED")


def directional_corridor_astar(
        vertices, tetrahedra, neighbors, allowed, coordinates,
        source_tangents, start, end, settings, edge_penalty=None):
    """A* on (previous,current) state to reject oscillatory face fans."""
    centroids = np.asarray(vertices)[np.asarray(tetrahedra)].mean(axis=1)
    edge_penalty = edge_penalty or {}
    start_state = (-1, int(start))
    best, parent = {start_state: 0.}, {}
    heap = [(float(np.linalg.norm(centroids[start] - centroids[end])),
             0., -1, int(start))]
    backward = float(settings["maximum_backward_s"])
    turn_weight = float(settings.get("directional_turn_weight", 2.))
    tangent_weight = float(settings.get("tangent_alignment_weight", 1.))
    while heap:
        _, cost, previous, current = heapq.heappop(heap)
        state = (previous, current)
        if cost != best.get(state):
            continue
        if current == end:
            states = [state]
            while states[-1] != start_state:
                states.append(parent[states[-1]])
            states.reverse()
            return [item[1] for item in states]
        for adjacent in sorted(int(x) for x in neighbors[current] if x >= 0):
            if adjacent == previous or not allowed[adjacent]:
                continue
            if not monotonic_step_allowed(
                    coordinates[current], coordinates[adjacent], backward):
                continue
            delta = centroids[adjacent] - centroids[current]
            length = np.linalg.norm(delta)
            direction = delta / max(length, 1e-30)
            source_direction = source_tangents[current]
            alignment = 1. - np.clip(
                np.dot(direction, source_direction), -1., 1.)
            turn = 0.
            if previous >= 0:
                incoming = centroids[current] - centroids[previous]
                incoming /= max(np.linalg.norm(incoming), 1e-30)
                turn = 1. - np.clip(np.dot(incoming, direction), -1., 1.)
            edge = (min(current, adjacent), max(current, adjacent))
            step = length * (
                1. + tangent_weight * alignment + turn_weight * turn
                + edge_penalty.get(edge, 0.))
            candidate = cost + step
            next_state = (current, adjacent)
            if candidate < best.get(next_state, np.inf) - 1e-15:
                best[next_state], parent[next_state] = candidate, state
                heuristic = np.linalg.norm(
                    centroids[adjacent] - centroids[end])
                heapq.heappush(
                    heap, (candidate + heuristic, candidate,
                           current, adjacent))
    raise ValueError("SOURCE_CORRIDOR_DISCONNECTED")


def diverse_portal_sequences(vertices, tetrahedra, neighbors, allowed,
                             coordinates, start, end, settings,
                             source_tangents=None):
    count = int(settings["maximum_sequences_per_segment"])
    diversity = float(settings["sequence_diversity_weight"])
    penalties, paths, signatures = {}, [], set()
    for _ in range(count * 3):
        try:
            if source_tangents is None:
                path = corridor_astar(
                    vertices, tetrahedra, neighbors, allowed, coordinates,
                    start, end, float(settings["maximum_backward_s"]),
                    penalties)
            else:
                path = directional_corridor_astar(
                    vertices, tetrahedra, neighbors, allowed, coordinates,
                    source_tangents, start, end, settings, penalties)
        except ValueError:
            break
        signature = tuple(path)
        if signature not in signatures:
            paths.append(path)
            signatures.add(signature)
        for first, second in zip(path[:-1], path[1:]):
            edge = (min(first, second), max(first, second))
            penalties[edge] = penalties.get(edge, 0.) + diversity
        if len(paths) == count:
            break
    return paths


def portal_point(triangle, logits, margin):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    weights /= weights.sum()
    beta = margin + (1. - 3. * margin) * weights
    return beta @ np.asarray(triangle), beta


def portal_points(vertices, faces, variables, margin):
    points, barycentric = [], []
    for face, value in zip(faces, np.asarray(variables).reshape((-1, 3))):
        point, beta = portal_point(np.asarray(vertices)[face], value, margin)
        points.append(point)
        barycentric.append(beta)
    return np.asarray(points), np.asarray(barycentric)


def _portal_point_jacobian(triangle, logits, margin):
    shifted = logits - np.max(logits)
    softmax = np.exp(shifted)
    softmax /= softmax.sum()
    scale = 1. - 3. * margin
    beta = margin + scale * softmax
    jacobian_beta = scale * (
        np.diag(softmax) - np.outer(softmax, softmax))
    # Columns differentiate the 3-D point with respect to each logit.
    jacobian_point = np.asarray(triangle).T @ jacobian_beta
    return beta @ np.asarray(triangle), beta, jacobian_point


def route_length_gradient(points):
    points = np.asarray(points, dtype=float)
    gradient = np.zeros_like(points)
    delta = np.diff(points, axis=0)
    unit = delta / np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1e-30)
    gradient[:-1] -= unit
    gradient[1:] += unit
    return gradient


def turn_energy(points, preferred_degrees):
    points = np.asarray(points)
    direction = np.diff(points, axis=0)
    direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-30)
    angle = np.arccos(np.clip(np.einsum(
        "ij,ij->i", direction[:-1], direction[1:]), -1., 1.))
    excess = np.maximum(angle - np.radians(preferred_degrees), 0.)
    return float(np.dot(excess, excess))


def ordered_source_targets(source_curve, count):
    sampled, _ = sample_polyline(
        source_curve, max(route_length(source_curve) / max(8 * count, 8), 1e-6))
    indices = np.linspace(0, len(sampled) - 1, count + 2).round().astype(int)
    return sampled[indices[1:-1]], indices[1:-1] / max(len(sampled) - 1, 1)


def source_segment_prior(points, segment, samples=65):
    """Cubic Hermite prior for one source segment using neighboring anchors."""
    points = np.asarray(points, dtype=float)
    tangents = polyline_tangents(points)
    first, second = points[segment], points[segment + 1]
    scale = np.linalg.norm(second - first)
    result = []
    for u in np.linspace(0., 1., samples):
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        result.append(
            h00 * first + h10 * scale * tangents[segment]
            + h01 * second + h11 * scale * tangents[segment + 1])
    return np.asarray(result)


@dataclass
class PortalOptimization:
    points: np.ndarray
    barycentric: np.ndarray
    initial_points: np.ndarray
    source_parameters: np.ndarray
    objective: float
    iterations: int
    success: bool
    minimum_clearance: float


def optimize_portals(vertices, tetrahedra, tet_path, start, end,
                     source_curve, clearance_field, settings):
    faces = [shared_face(tetrahedra, a, b)
             for a, b in zip(tet_path[:-1], tet_path[1:])]
    if not faces:
        points = np.vstack((start, end))
        return PortalOptimization(points, np.empty((0, 3)), points.copy(),
                                  np.empty(0), route_length(points), 0, True,
                                  np.inf)
    margin = float(settings["portal_margin"])
    targets, parameters = ordered_source_targets(source_curve, len(faces))
    initial_beta = np.asarray([
        np.maximum(face_barycentric(np.asarray(vertices)[face], target), 1e-4)
        for face, target in zip(faces, targets)])
    initial_beta /= initial_beta.sum(axis=1)[:, None]
    initial = np.log(np.maximum(
        (initial_beta - margin) / max(1. - 3. * margin, 1e-12), 1e-8))
    initial_points, _ = portal_points(vertices, faces, initial, margin)
    source_length = route_length(source_curve)
    start_tangent, end_tangent = polyline_tangents(source_curve)[[0, -1]]
    deadline = time.monotonic() + float(
        settings.get("maximum_seconds_per_sequence", 5.))

    def objective_gradient(flat):
        if time.monotonic() > deadline:
            raise TimeoutError
        values = np.asarray(flat).reshape((-1, 3))
        portals, jacobians = [], []
        for face, value in zip(faces, values):
            point, _, jacobian = _portal_point_jacobian(
                np.asarray(vertices)[face], value, margin)
            portals.append(point)
            jacobians.append(jacobian)
        portals = np.asarray(portals)
        points = np.vstack((start, portals, end))
        length = route_length(points)
        length_weight = float(settings["length_weight"])
        point_gradient = length_weight * route_length_gradient(points)
        source_deviation = np.sum((portals - targets) ** 2)
        point_gradient[1:-1] += (
            2. * float(settings["source_deviation_weight"])
            * (portals - targets))
        directions = np.diff(points, axis=0)
        segment_lengths = np.maximum(
            np.linalg.norm(directions, axis=1), 1e-30)
        directions /= segment_lengths[:, None]
        tangent = (1. - np.dot(directions[0], start_tangent)
                   + 1. - np.dot(directions[-1], end_tangent))
        identity = np.eye(3)
        start_gradient = -(
            identity - np.outer(directions[0], directions[0])
        ) @ start_tangent / segment_lengths[0]
        end_gradient = -(
            identity - np.outer(directions[-1], directions[-1])
        ) @ end_tangent / segment_lengths[-1]
        tangent_weight = float(settings["tangent_weight"])
        point_gradient[0] -= tangent_weight * start_gradient
        point_gradient[1] += tangent_weight * start_gradient
        point_gradient[-2] -= tangent_weight * end_gradient
        point_gradient[-1] += tangent_weight * end_gradient
        # A differentiable curvature surrogate with the same minimum as zero
        # turning. Scaling by mean edge length makes it mesh independent.
        residual = points[:-2] - 2. * points[1:-1] + points[2:]
        scale_length = max(float(np.mean(segment_lengths)), 1e-8)
        bending = float(np.sum(residual ** 2) / scale_length ** 2)
        bend_weight = float(settings["turn_weight"])
        scaled_residual = 2. * bend_weight * residual / scale_length ** 2
        point_gradient[:-2] += scaled_residual
        point_gradient[1:-1] -= 2. * scaled_residual
        point_gradient[2:] += scaled_residual
        cosine = np.einsum(
            "ij,ij->i", directions[:-1], directions[1:])
        preferred_cosine = np.cos(np.radians(
            float(settings["preferred_turn_degrees"])))
        angular_excess = np.maximum(preferred_cosine - cosine, 0.)
        angular_energy = float(np.dot(angular_excess, angular_excess))
        for joint, excess in enumerate(angular_excess):
            if excess <= 0.:
                continue
            first_direction = directions[joint]
            second_direction = directions[joint + 1]
            derivative_cosine_first = (
                identity - np.outer(first_direction, first_direction)
            ) @ second_direction / segment_lengths[joint]
            derivative_cosine_second = (
                identity - np.outer(second_direction, second_direction)
            ) @ first_direction / segment_lengths[joint + 1]
            coefficient = -2. * bend_weight * excess
            gradient_first = coefficient * derivative_cosine_first
            gradient_second = coefficient * derivative_cosine_second
            point_gradient[joint] -= gradient_first
            point_gradient[joint + 1] += (
                gradient_first - gradient_second)
            point_gradient[joint + 2] += gradient_second
        if hasattr(clearance_field, "tree"):
            approximate, nearest = clearance_field.tree.query(portals)
        else:
            approximate = clearance_field.approximate(portals)
            nearest = None
        preferred = float(settings["preferred_clearance_m"])
        clearance = np.maximum(preferred - approximate, 0.) / preferred
        clearance_weight = float(settings["clearance_weight"])
        active = approximate < preferred
        if np.any(active) and nearest is not None:
            away = portals[active] - clearance_field.tree.data[nearest[active]]
            away /= np.maximum(
                np.linalg.norm(away, axis=1)[:, None], 1e-30)
            point_gradient[1:-1][active] += (
                -2. * clearance_weight
                * (preferred - approximate[active])[:, None]
                / preferred ** 2 * away)
        length_prior_weight = float(settings["length_prior_weight"])
        relative = (length - source_length) / max(source_length, 1e-30)
        point_gradient += (
            2. * length_prior_weight * relative
            / max(source_length, 1e-30) * route_length_gradient(points))
        value = (
            length_weight * length
            + length_prior_weight * relative ** 2
            + float(settings["source_deviation_weight"]) * source_deviation
            + tangent_weight * tangent
            + bend_weight * (bending + angular_energy)
            + clearance_weight * np.dot(clearance, clearance))
        gradient = np.asarray([
            jacobian.T @ point_gradient[index + 1]
            for index, jacobian in enumerate(jacobians)])
        return float(value), gradient.ravel()

    try:
        result = minimize(
            objective_gradient, initial.ravel(), method="L-BFGS-B", jac=True,
            options={"maxiter": int(settings["maximum_iterations"]),
                     "ftol": 1e-12, "maxls": 20})
        values, success, iterations, value = (
            result.x, bool(result.success), int(result.nit), float(result.fun))
    except TimeoutError:
        values, success, iterations = initial.ravel(), False, 0
        value = np.inf
    portals, barycentric = portal_points(vertices, faces, values, margin)
    points = np.vstack((start, portals, end))
    segment_samples = np.vstack([
        (1. - alpha) * first + alpha * second
        for first, second in zip(points[:-1], points[1:])
        for alpha in np.linspace(0., 1., 5)])
    minimum_clearance = float(np.min(
        clearance_field.approximate(segment_samples)))
    success &= minimum_clearance >= float(settings["hard_clearance_m"])
    return PortalOptimization(
        points, barycentric, np.vstack((start, initial_points, end)),
        parameters, value, iterations, success, minimum_clearance)


def simplify_tet_path(vertices, tetrahedra, neighbors, locator, points,
                      tet_path, allowed, clearance_field, hard_clearance,
                      fiber_id=0, segment_id=0, maximum_skip=64):
    """Deterministically replace the longest feasible nonlocal shortcuts."""
    points, path = np.asarray(points), list(map(int, tet_path))
    changed = True
    contractions = 0
    while changed and len(points) > 2 and contractions < 128:
        changed = False
        for first in range(len(points) - 2):
            farthest = min(len(points) - 1, first + int(maximum_skip))
            spans = sorted(set(
                [farthest - first, 32, 16, 8, 4, 2]), reverse=True)
            for span in spans:
                last = first + span
                if last > farthest or last <= first + 1:
                    continue
                try:
                    route, _, _ = traverse_straight_segment(
                        vertices, tetrahedra, neighbors, locator,
                        points[first], points[last], fiber_id, segment_id)
                except ValueError:
                    continue
                if not all(allowed[tet] for tet in route.tet_path):
                    continue
                samples = np.linspace(0., 1., 17)[:, None] * (
                    points[last] - points[first]) + points[first]
                if np.min(clearance_field.approximate(samples)) < hard_clearance:
                    continue
                replacement = route.points
                points = np.vstack((points[:first], replacement,
                                    points[last + 1:]))
                path = path[:first] + route.tet_path + path[last:]
                changed = True
                contractions += 1
                break
            if changed:
                break
    return points, path


def classify_near_crossing(source_distance, anchor_distance, route_distance,
                           threshold, intersects=False):
    if intersects:
        return "TRUE_ROUTE_INTERSECTION"
    if source_distance < threshold:
        return "PREEXISTING_SOURCE_PROXIMITY"
    if anchor_distance < threshold:
        return "INTRODUCED_BY_ANCHOR_REPAIR"
    if route_distance < threshold:
        return "INTRODUCED_BY_ROUTING"
    return None


def deterministic_candidate_ranking(candidates):
    return sorted(candidates, key=lambda row: (
        not row.get("passes_hard_gates", False),
        float(row["score"]), tuple(row["tet_path"])))
