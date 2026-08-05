"""Continuous-ish navigation inside an exact tetrahedral corridor.

The refined graph is only a sequence initializer. Every graph edge is owned by
one convex tet; the returned owner sequence is subsequently converted to exact
shared-face portals and optimized by :mod:`muscle_sim.fiber_portals`.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import time

import numpy as np
from scipy.spatial import cKDTree

from muscle_sim.fiber_portals import sample_polyline
from muscle_sim.fiber_routing import LOCAL_FACES, shared_face


LOCAL_EDGES = np.asarray([
    [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int8)


@dataclass
class NavigationDomain:
    tetrahedra: np.ndarray
    points: np.ndarray
    clearance: np.ndarray
    source_parameter: np.ndarray
    preferred_direction: np.ndarray
    adjacency: list
    tet_nodes: dict
    start_node: int
    end_node: int


def _face_samples(triangle, count):
    if count <= 1:
        return [np.mean(triangle, axis=0)]
    result = [np.mean(triangle, axis=0)]
    for vertex in triangle:
        result.append(.6 * np.mean(triangle, axis=0) + .4 * vertex)
        if len(result) >= count:
            break
    return result


def navigation_candidates(vertices, tetrahedra, corridor_tets, settings):
    """Generate vertices, edge points, face points, and tet Steiner points."""
    vertices = np.asarray(vertices)
    tetrahedra = np.asarray(tetrahedra)
    edge_count = int(settings["edge_samples_per_tet_edge"])
    face_count = int(settings["face_samples_per_portal"])
    interior_count = int(settings["interior_samples_per_tet"])
    raw = {}
    for tet_id in sorted(map(int, corridor_tets)):
        tet = tetrahedra[tet_id]
        q = vertices[tet]
        points = list(q)
        for local_edge in LOCAL_EDGES:
            for index in range(1, edge_count + 1):
                alpha = index / (edge_count + 1)
                points.append(
                    (1. - alpha) * q[local_edge[0]]
                    + alpha * q[local_edge[1]])
        for local_face in LOCAL_FACES:
            points.extend(_face_samples(q[local_face], face_count))
        if interior_count:
            centroid = q.mean(axis=0)
            points.append(centroid)
            for vertex in q[:max(0, interior_count - 1)]:
                points.append(.75 * centroid + .25 * vertex)
        raw[tet_id] = np.asarray(points)
    return raw


def build_navigation_domain(vertices, tetrahedra, corridor_tets,
                            source_curve, start, end, start_tet, end_tet,
                            clearance_query, settings):
    raw = navigation_candidates(
        vertices, tetrahedra, corridor_tets, settings)
    maximum_nodes = int(settings["maximum_navigation_nodes"])
    key_to_id, points, tet_nodes = {}, [], {}

    def add(point):
        key = tuple(np.round(np.asarray(point), 12))
        if key not in key_to_id:
            if len(points) >= maximum_nodes:
                raise ValueError("NAVIGATION_NODE_BUDGET_EXHAUSTED")
            key_to_id[key] = len(points)
            points.append(np.asarray(point, dtype=float))
        return key_to_id[key]

    for tet_id, candidates in raw.items():
        tet_nodes[tet_id] = [add(point) for point in candidates]
    start_node, end_node = add(start), add(end)
    tet_nodes.setdefault(int(start_tet), []).append(start_node)
    tet_nodes.setdefault(int(end_tet), []).append(end_node)
    points = np.asarray(points)
    clearance = np.asarray(clearance_query(points), dtype=float)
    hard = float(settings["hard_clearance_m"])
    valid = clearance >= hard
    valid[start_node] = valid[end_node] = True
    sampled, parameter = sample_polyline(
        source_curve, max(
            np.sum(np.linalg.norm(np.diff(source_curve, axis=0), axis=1))
            / 512., 1e-6))
    source_direction = np.gradient(sampled, axis=0)
    source_direction /= np.maximum(
        np.linalg.norm(source_direction, axis=1)[:, None], 1e-30)
    _, nearest = cKDTree(sampled).query(points)
    node_parameter = parameter[nearest]
    preferred = source_direction[nearest]
    node_parameter[start_node], node_parameter[end_node] = 0., 1.
    preferred[start_node], preferred[end_node] = (
        source_direction[0], source_direction[-1])
    adjacency = [[] for _ in points]
    edge_count = 0
    maximum_edges = int(settings["maximum_navigation_edges"])
    for tet_id in sorted(tet_nodes):
        nodes = sorted(set(tet_nodes[tet_id]))
        for position, first in enumerate(nodes):
            if not valid[first]:
                continue
            for second in nodes[position + 1:]:
                if not valid[second]:
                    continue
                midpoint = .5 * (points[first] + points[second])
                if float(clearance_query(midpoint[None])[0]) < hard:
                    continue
                delta = points[second] - points[first]
                length = float(np.linalg.norm(delta))
                if length <= 1e-14:
                    continue
                adjacency[first].append((second, int(tet_id), length))
                adjacency[second].append((first, int(tet_id), length))
                edge_count += 2
                if edge_count > maximum_edges:
                    raise ValueError("NAVIGATION_EDGE_BUDGET_EXHAUSTED")
    for rows in adjacency:
        rows.sort(key=lambda row: (row[0], row[1]))
    return NavigationDomain(
        np.asarray(tetrahedra), points, clearance, node_parameter, preferred,
        adjacency, tet_nodes,
        start_node, end_node)


def transition_cost(domain, previous, current, adjacent, length, settings):
    direction = domain.points[adjacent] - domain.points[current]
    direction /= max(np.linalg.norm(direction), 1e-30)
    alignment = 1. - np.clip(
        np.dot(direction, domain.preferred_direction[current]), -1., 1.)
    turn = 0.
    if previous >= 0:
        incoming = domain.points[current] - domain.points[previous]
        incoming /= max(np.linalg.norm(incoming), 1e-30)
        cosine = np.clip(np.dot(incoming, direction), -1., 1.)
        angle = np.degrees(np.arccos(cosine))
        if (settings.get("hard_turn_rejection", True)
                and angle > float(settings["navigation_hard_turn_degrees"])):
            return np.inf
        turn = (1. - cosine) ** 2
    backward = max(
        0., domain.source_parameter[current]
        - domain.source_parameter[adjacent]
        - float(settings["backward_parameter_tolerance"]))
    preferred_clearance = float(settings["preferred_clearance_m"])
    boundary = max(
        0., preferred_clearance - domain.clearance[adjacent]
    ) / max(preferred_clearance, 1e-30)
    return length * (
        1. + float(settings["direction_weight"]) * alignment
        + float(settings["turn_weight"]) * turn
        + float(settings["backward_parameter_weight"]) * backward
        + float(settings["clearance_weight"]) * boundary ** 2)


def directional_navigation_search(domain, settings):
    """Deterministic A* on quantized incoming-direction states."""
    if settings.get("exact_direction_state", False):
        return exact_directional_navigation_search(domain, settings)
    start_time = time.monotonic()
    budget = float(settings["maximum_seconds_per_segment"])
    maximum_states = int(settings["maximum_direction_states"])
    bins = []
    for x in (-1., 0., 1.):
        for y in (-1., 0., 1.):
            for z in (-1., 0., 1.):
                if x == y == z == 0.:
                    continue
                value = np.asarray([x, y, z])
                bins.append(value / np.linalg.norm(value))
    bins = np.asarray(bins)

    def direction_bin(direction):
        direction = np.asarray(direction)
        direction /= max(np.linalg.norm(direction), 1e-30)
        return int(np.argmax(bins @ direction))

    start_bin = direction_bin(
        domain.preferred_direction[domain.start_node])
    start = (domain.start_node, start_bin)
    best, parent, owner = {start: 0.}, {}, {}
    heap = [(float(np.linalg.norm(
        domain.points[domain.start_node]
        - domain.points[domain.end_node])), 0.,
        domain.start_node, start_bin)]
    final = None
    while heap:
        if time.monotonic() - start_time > budget:
            raise ValueError("CONTINUOUS_SEQUENCE_RESOURCE_BUDGET")
        _, cost, current, incoming_bin = heapq.heappop(heap)
        state = (current, incoming_bin)
        if cost != best.get(state):
            continue
        if current == domain.end_node:
            final = state
            break
        if len(best) > maximum_states:
            raise ValueError("DIRECTION_STATE_BUDGET_EXHAUSTED")
        for adjacent, tet_id, length in domain.adjacency[current]:
            delta = domain.points[adjacent] - domain.points[current]
            direction = delta / max(np.linalg.norm(delta), 1e-30)
            cosine = np.clip(np.dot(bins[incoming_bin], direction), -1., 1.)
            angle = np.degrees(np.arccos(cosine))
            if (settings.get("hard_turn_rejection", True)
                    and angle > float(
                        settings["navigation_hard_turn_degrees"])):
                continue
            alignment = 1. - np.clip(np.dot(
                direction, domain.preferred_direction[current]), -1., 1.)
            backward = max(
                0., domain.source_parameter[current]
                - domain.source_parameter[adjacent]
                - float(settings["backward_parameter_tolerance"]))
            preferred_clearance = float(settings["preferred_clearance_m"])
            boundary = max(
                0., preferred_clearance - domain.clearance[adjacent]
            ) / max(preferred_clearance, 1e-30)
            step = length * (
                1. + float(settings["direction_weight"]) * alignment
                + float(settings["turn_weight"]) * (1. - cosine) ** 2
                + float(settings["backward_parameter_weight"]) * backward
                + float(settings["clearance_weight"]) * boundary ** 2)
            candidate = cost + step
            outgoing_bin = direction_bin(direction)
            next_state = (adjacent, outgoing_bin)
            if candidate < best.get(next_state, np.inf) - 1e-15:
                best[next_state] = candidate
                parent[next_state] = state
                owner[next_state] = tet_id
                heuristic = np.linalg.norm(
                    domain.points[adjacent]
                    - domain.points[domain.end_node])
                heapq.heappush(
                    heap, (candidate + heuristic, candidate,
                           adjacent, outgoing_bin))
    if final is None:
        raise ValueError("NO_CONTINUOUS_TURN_FEASIBLE_PATH")
    states = [final]
    while states[-1] != start:
        states.append(parent[states[-1]])
    states.reverse()
    nodes = [state[0] for state in states]
    owners = [owner[state] for state in states[1:]]
    return nodes, owners, float(best[final])


def exact_directional_navigation_search(domain, settings):
    start_time = time.monotonic()
    budget = float(settings["maximum_seconds_per_segment"])
    maximum_states = int(settings["maximum_direction_states"])
    track_owner = bool(settings.get(
        "require_face_adjacent_owner_transitions", False))
    start = (-1, domain.start_node, -1)
    best, parent, owner = {start: 0.}, {}, {}
    heap = [(float(np.linalg.norm(
        domain.points[domain.start_node] - domain.points[domain.end_node])),
        0., -1, domain.start_node, -1)]
    final = None
    while heap:
        if time.monotonic() - start_time > budget:
            raise ValueError("CONTINUOUS_SEQUENCE_RESOURCE_BUDGET")
        _, cost, previous, current, incoming_tet = heapq.heappop(heap)
        state = (previous, current, incoming_tet)
        if cost != best.get(state):
            continue
        if current == domain.end_node:
            final = state
            break
        if len(best) > maximum_states:
            raise ValueError("DIRECTION_STATE_BUDGET_EXHAUSTED")
        for adjacent, tet_id, length in domain.adjacency[current]:
            if adjacent == previous:
                continue
            if (settings.get("require_face_adjacent_owner_transitions", False)
                    and incoming_tet >= 0 and tet_id != incoming_tet
                    and len(set(domain.tetrahedra[incoming_tet]).intersection(
                        domain.tetrahedra[tet_id])) != 3):
                continue
            step = transition_cost(
                domain, previous, current, adjacent, length, settings)
            if not np.isfinite(step):
                continue
            candidate = cost + step
            next_incoming_tet = tet_id if track_owner else -1
            next_state = (current, adjacent, next_incoming_tet)
            if candidate < best.get(next_state, np.inf) - 1e-15:
                best[next_state] = candidate
                parent[next_state] = state
                owner[next_state] = tet_id
                heuristic = np.linalg.norm(
                    domain.points[adjacent]
                    - domain.points[domain.end_node])
                heapq.heappush(
                    heap, (candidate + heuristic, candidate,
                           current, adjacent, next_incoming_tet))
    if final is None:
        raise ValueError("NO_CONTINUOUS_TURN_FEASIBLE_PATH")
    states = [final]
    while states[-1] != start:
        states.append(parent[states[-1]])
    states.reverse()
    nodes = [states[0][1]] + [state[1] for state in states[1:]]
    owners = [owner[state] for state in states[1:]]
    return nodes, owners, float(best[final])


def remove_owner_loops(owners):
    """Chronological loop erasure with stable first-occurrence ordering."""
    result, positions = [], {}
    for owner in map(int, owners):
        if result and owner == result[-1]:
            continue
        if owner in positions:
            cut = positions[owner]
            for removed in result[cut + 1:]:
                positions.pop(removed, None)
            result = result[:cut + 1]
        else:
            positions[owner] = len(result)
            result.append(owner)
    return result


def _incident_bridge(tetrahedra, neighbors, first, second, point,
                     tolerance=1e-9):
    if first == second:
        return [first]
    vertices_in_point = None
    # Candidate incident tets are discovered by shared mesh vertices of the
    # endpoint tets; exact point incidence is checked later by barycentrics in
    # route validation.
    common = set(map(int, tetrahedra[first])).intersection(
        map(int, tetrahedra[second]))
    candidates = {first, second}
    for tet in range(len(tetrahedra)):
        if common and common.issubset(set(map(int, tetrahedra[tet]))):
            candidates.add(tet)
    queue, parent = [first], {first: None}
    for current in queue:
        if current == second:
            break
        for adjacent in sorted(int(x) for x in neighbors[current] if x >= 0):
            if adjacent in candidates and adjacent not in parent:
                parent[adjacent] = current
                queue.append(adjacent)
    if second not in parent:
        raise ValueError("EDGE_VERTEX_CROSSING_DEGENERACY")
    path, current = [], second
    while current is not None:
        path.append(current)
        current = parent[current]
    return path[::-1]


def guide_to_tet_sequence(tetrahedra, neighbors, guide_points, edge_owners):
    sequence = []
    for index, owner in enumerate(map(int, edge_owners)):
        if not sequence:
            sequence.append(owner)
            continue
        if owner == sequence[-1]:
            continue
        if len(set(tetrahedra[owner]).intersection(
                tetrahedra[sequence[-1]])) == 3:
            sequence.append(owner)
            continue
        bridge = _incident_bridge(
            tetrahedra, neighbors, sequence[-1], owner,
            guide_points[index])
        sequence.extend(bridge[1:])
    return remove_owner_loops(sequence)


def guide_to_contracted_route(
        tetrahedra, neighbors, guide_points, edge_owners):
    """Collapse same-tet guide runs and encode edge/vertex fans explicitly."""
    guide_points = np.asarray(guide_points)
    owners = list(map(int, edge_owners))
    if not owners:
        return guide_points.copy(), [], []
    runs = []
    first = 0
    for index in range(1, len(owners)):
        if owners[index] != owners[index - 1]:
            runs.append((owners[first], first, index - 1))
            first = index
    runs.append((owners[first], first, len(owners) - 1))
    points = [guide_points[0]]
    route_owners, events = [], []
    previous_owner = None
    for owner, run_first, run_last in runs:
        junction = guide_points[run_first]
        if previous_owner is not None and owner != previous_owner:
            bridge = _incident_bridge(
                tetrahedra, neighbors, previous_owner, owner, junction)
            for intermediate in bridge[1:-1]:
                route_owners.append(int(intermediate))
                points.append(junction.copy())
            events.append({
                "guide_point_index": run_first,
                "position": junction,
                "from_tet": int(previous_owner), "to_tet": int(owner),
                "bridge_tets": bridge,
                "zero_length_transition_count": max(0, len(bridge) - 2)})
        route_owners.append(owner)
        points.append(guide_points[run_last + 1])
        previous_owner = owner
    return np.asarray(points), route_owners, events


def maximum_physical_turn(points, zero_tolerance=1e-12):
    direction = np.diff(np.asarray(points), axis=0)
    length = np.linalg.norm(direction, axis=1)
    direction = direction[length > zero_tolerance]
    if len(direction) < 2:
        return 0.
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return float(np.max(np.degrees(np.arccos(np.clip(np.einsum(
        "ij,ij->i", direction[:-1], direction[1:]), -1., 1.)))))


def extract_face_fans(tetrahedra, tet_path, maximum_size):
    faces = [set(map(int, shared_face(tetrahedra, a, b)))
             for a, b in zip(tet_path[:-1], tet_path[1:])]
    fans = []
    for first in range(len(faces)):
        common = set(faces[first])
        for last in range(first + 1, min(len(faces), first + maximum_size)):
            common &= faces[last]
            if not common:
                break
            if last - first >= 1:
                fans.append({
                    "first_tet_index": first,
                    "last_tet_index": last + 1,
                    "common_vertex_ids": sorted(common)})
    return fans


def enumerate_local_tet_paths(neighbors, allowed_tets, start, end,
                              maximum_paths, maximum_length):
    allowed = set(map(int, allowed_tets))
    paths, stack = [], [(int(start), [int(start)])]
    while stack and len(paths) < maximum_paths:
        current, path = stack.pop()
        if current == end:
            paths.append(path)
            continue
        if len(path) >= maximum_length:
            continue
        for adjacent in sorted(
                (int(x) for x in neighbors[current]
                 if x >= 0 and int(x) in allowed and int(x) not in path),
                reverse=True):
            stack.append((adjacent, path + [adjacent]))
    return paths


def portal_cone_turn_lower_bound(entry, first_triangle, second_triangle, exit,
                                 samples=5):
    """Sampled conservative proxy for a two-portal minimum maximum turn."""
    barycentric = []
    for i in range(samples + 1):
        for j in range(samples + 1 - i):
            barycentric.append(
                [i / samples, j / samples, 1. - (i + j) / samples])
    first_points = np.asarray(barycentric) @ np.asarray(first_triangle)
    second_points = np.asarray(barycentric) @ np.asarray(second_triangle)
    best = np.inf
    for first in first_points:
        for second in second_points:
            directions = np.diff(np.vstack((entry, first, second, exit)), axis=0)
            lengths = np.linalg.norm(directions, axis=1)
            if np.min(lengths) < 1e-14:
                continue
            directions /= lengths[:, None]
            angle = np.degrees(np.arccos(np.clip(np.einsum(
                "ij,ij->i", directions[:-1], directions[1:]), -1., 1.)))
            best = min(best, float(np.max(angle)))
    return best


def cluster_guide_paths(paths, overlap_threshold=.9):
    representatives = []
    for path in paths:
        current = set(path)
        if any(len(current.intersection(other)) / max(
                len(current.union(other)), 1) >= overlap_threshold
               for other in map(set, representatives)):
            continue
        representatives.append(path)
    return representatives
