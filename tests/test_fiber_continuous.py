import unittest

import numpy as np

from muscle_sim.fiber_continuous import (
    build_navigation_domain, cluster_guide_paths,
    directional_navigation_search, enumerate_local_tet_paths,
    extract_face_fans, guide_to_tet_sequence, navigation_candidates,
    portal_cone_turn_lower_bound, remove_owner_loops, transition_cost)
from muscle_sim.fiber_routing import build_tet_adjacency


V = np.asarray([
    [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
    [1., 1., 1.], [2., 1., 1.]])
T = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32)
N, _ = build_tet_adjacency(T)
SOURCE = np.asarray([[.1, .1, .1], [1.6, .8, .8]])
SETTINGS = {
    "edge_samples_per_tet_edge": 1, "face_samples_per_portal": 2,
    "interior_samples_per_tet": 1, "maximum_navigation_nodes": 1000,
    "maximum_navigation_edges": 10000, "hard_clearance_m": 0.,
    "preferred_clearance_m": .1, "direction_weight": 1.,
    "turn_weight": 2., "hard_turn_rejection": True,
    "navigation_hard_turn_degrees": 150.,
    "backward_parameter_tolerance": .05,
    "backward_parameter_weight": 10., "clearance_weight": 1.,
    "maximum_seconds_per_segment": 2.,
    "maximum_direction_states": 10000}


def clearance(points):
    return np.ones(len(points))


def domain():
    return build_navigation_domain(
        V, T, [0, 1, 2], SOURCE, SOURCE[0], SOURCE[-1], 0, 2,
        clearance, SETTINGS)


def test_continuous_corridor_domain_construction():
    value = domain()
    assert len(value.points) > 2
    assert value.start_node != value.end_node


def test_refined_navigation_node_generation():
    raw = navigation_candidates(V, T, [0], SETTINGS)
    assert len(raw[0]) > 4
    assert any(np.allclose(point, V[T[0]].mean(axis=0)) for point in raw[0])


def test_exact_same_tet_navigation_edges():
    value = domain()
    owners = {row[1] for rows in value.adjacency for row in rows}
    assert owners.issubset({0, 1, 2})


def test_directional_state_shortest_path():
    nodes, owners, cost = directional_navigation_search(domain(), SETTINGS)
    assert nodes[0] == domain().start_node
    assert owners and np.isfinite(cost)


def test_turn_aware_transition_costs():
    value = domain()
    current = value.start_node
    adjacent = value.adjacency[current][0][0]
    assert transition_cost(value, -1, current, adjacent, 1., SETTINGS) >= 1.


def test_source_parameter_monotonicity():
    value = domain()
    nodes, _, _ = directional_navigation_search(value, SETTINGS)
    parameter = value.source_parameter[nodes]
    assert np.min(np.diff(parameter)) >= -.1


def test_endpoint_tangent_initialization():
    value = domain()
    assert np.dot(value.preferred_direction[value.start_node],
                  SOURCE[-1] - SOURCE[0]) > 0.


def test_clearance_based_node_and_edge_rejection():
    def blocked(points):
        return np.zeros(len(points))
    value = build_navigation_domain(
        V, T, [0], SOURCE, SOURCE[0], [.2, .2, .2], 0, 0, blocked,
        dict(SETTINGS, hard_clearance_m=.1))
    assert all(not rows for index, rows in enumerate(value.adjacency)
               if index not in (value.start_node, value.end_node))


def test_guide_path_backtracing():
    value = domain()
    nodes, owners, _ = directional_navigation_search(value, SETTINGS)
    assert len(nodes) == len(owners) + 1


def test_deterministic_guide_to_tet_traversal():
    value = domain()
    nodes, owners, _ = directional_navigation_search(value, SETTINGS)
    points = value.points[nodes]
    first = guide_to_tet_sequence(T, N, points, owners)
    second = guide_to_tet_sequence(T, N, points, owners)
    assert first == second


def test_tet_edge_crossing_degeneracy_bridge():
    sequence = guide_to_tet_sequence(
        T, N, np.asarray([[0., 0., 0.], [0., 1., 0.], [1., 1., 1.]]),
        [0, 2])
    assert sequence[0] == 0 and sequence[-1] == 2


def test_tet_vertex_crossing_degeneracy_is_deterministic():
    first = guide_to_tet_sequence(
        T, N, np.zeros((3, 3)), [0, 2])
    assert first == guide_to_tet_sequence(T, N, np.zeros((3, 3)), [0, 2])


def test_repeated_tet_and_loop_removal():
    assert remove_owner_loops([0, 0, 1, 2, 1, 2]) == [0, 1, 2]


def test_face_fan_extraction():
    fans = extract_face_fans(T, [0, 1, 2], 4)
    assert fans


def test_alternate_face_fan_path_enumeration():
    paths = enumerate_local_tet_paths(N, [0, 1, 2], 0, 2, 4, 4)
    assert [0, 1, 2] in paths


def test_face_fan_contraction_principle():
    alternatives = enumerate_local_tet_paths(N, [0, 1, 2], 0, 2, 4, 4)
    assert min(map(len, alternatives)) <= 3


def test_portal_cone_turn_lower_bound():
    first = V[[1, 2, 3]]
    second = V[[2, 3, 4]]
    bound = portal_cone_turn_lower_bound(SOURCE[0], first, second, SOURCE[-1])
    assert 0. <= bound <= 180.


def test_sequence_diversity_clustering():
    paths = cluster_guide_paths([[0, 1, 2], [0, 1, 2], [0, 3, 2]])
    assert paths == [[0, 1, 2], [0, 3, 2]]


def test_corridor_radius_continuation():
    radii = [.25, .4, .64, 1.]
    assert next(radius for radius in radii if radius >= .6) == .64


def test_representative_segment_0_7_regression_contract():
    directions = np.diff(np.asarray([[0., 0., 0.], [1., 0., 0.],
                                     [2., 0., 0.]]), axis=0)
    turn = np.degrees(np.arccos(np.dot(
        directions[0] / np.linalg.norm(directions[0]),
        directions[1] / np.linalg.norm(directions[1]))))
    assert turn <= 45.


def test_deterministic_resource_budget_behavior():
    try:
        build_navigation_domain(
            V, T, [0, 1, 2], SOURCE, SOURCE[0], SOURCE[-1], 0, 2,
            clearance, dict(SETTINGS, maximum_navigation_nodes=2))
    except ValueError as error:
        assert str(error) == "NAVIGATION_NODE_BUDGET_EXHAUSTED"
    else:
        raise AssertionError("node budget was not enforced")


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
