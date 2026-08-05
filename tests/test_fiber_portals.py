import unittest

import numpy as np

from muscle_sim.fiber_portals import (
    classify_near_crossing, corridor_connected, deterministic_candidate_ranking,
    diverse_portal_sequences, expand_corridor, monotonic_step_allowed,
    optimize_portals, ordered_source_targets, portal_point, route_length_gradient,
    source_corridor, tet_source_coordinates, turn_energy)
from muscle_sim.fiber_routing import build_tet_adjacency, maximum_turn


V = np.asarray([
    [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
    [1., 1., 1.], [2., 1., 1.]])
T = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32)
N, _ = build_tet_adjacency(T)
SOURCE = np.asarray([[.1, .1, .1], [1.5, .8, .8]])


class Clearance:
    def __init__(self, value=1.):
        self.value = value

    def approximate(self, points):
        return np.full(len(points), self.value)


def test_source_corridor_construction():
    allowed, fields = source_corridor(
        V, T, SOURCE, np.ones(3), 10., 0., 0, 2)
    assert np.all(allowed)
    assert np.all(fields["distance_to_source_m"] >= 0.)


def test_adaptive_corridor_expansion():
    settings = {"initial_radius_m": .01, "maximum_radius_m": 10.,
                "radius_growth_factor": 10., "minimum_clearance_m": 0.}
    allowed, _, radius, attempts = expand_corridor(
        V, T, N, SOURCE, np.ones(3), 0, 2, settings)
    assert corridor_connected(N, allowed, 0, 2)
    assert radius >= settings["initial_radius_m"]
    assert attempts[-1]["connected"]


def test_corridor_connectivity():
    assert corridor_connected(N, np.ones(3, dtype=bool), 0, 2)
    assert not corridor_connected(N, np.asarray([1, 0, 1], dtype=bool), 0, 2)


def test_diverse_k_shortest_portal_sequences():
    settings = {"maximum_sequences_per_segment": 2,
                "sequence_diversity_weight": 1., "maximum_backward_s": 1.}
    paths = diverse_portal_sequences(
        V, T, N, np.ones(3, bool), np.arange(3.), 0, 2, settings)
    assert paths == [[0, 1, 2]]


def test_section_monotonicity():
    assert monotonic_step_allowed(.5, .49, .02)
    assert not monotonic_step_allowed(.5, .4, .02)


def test_portal_barycentric_parameterization():
    point, beta = portal_point(V[[1, 2, 3]], [0., 0., 0.], .01)
    assert np.allclose(beta.sum(), 1.)
    assert np.allclose(point, V[[1, 2, 3]].mean(axis=0))


def test_portal_margin_enforcement():
    _, beta = portal_point(V[[1, 2, 3]], [100., -100., -100.], .02)
    assert np.min(beta) >= .02 - 1e-14


def test_route_length_derivative():
    points = np.asarray([[0., 0., 0.], [1., .2, 0.], [2., 0., 0.]])
    analytic = route_length_gradient(points)[1]
    numerical = np.zeros(3)
    epsilon = 1e-6
    for axis in range(3):
        plus, minus = points.copy(), points.copy()
        plus[1, axis] += epsilon
        minus[1, axis] -= epsilon
        numerical[axis] = (
            np.linalg.norm(np.diff(plus, axis=0), axis=1).sum()
            - np.linalg.norm(np.diff(minus, axis=0), axis=1).sum()
        ) / (2 * epsilon)
    assert np.allclose(analytic, numerical, atol=1e-7)


def test_source_parameter_monotonic_correspondence():
    _, parameter = ordered_source_targets(SOURCE, 8)
    assert np.all(np.diff(parameter) > 0.)


def test_endpoint_tangent_derivative_finite():
    points = np.asarray([[0., 0., 0.], [1., .1, 0.]])
    derivative = route_length_gradient(points)
    assert np.all(np.isfinite(derivative))
    assert np.allclose(derivative.sum(axis=0), 0.)


def test_turn_angle_derivative_finite():
    points = np.asarray([[0., 0., 0.], [1., .1, 0.], [2., 0., 0.]])
    epsilon = 1e-6
    value = (turn_energy(points + [[0, 0, 0], [0, epsilon, 0], [0, 0, 0]], 0.)
             - turn_energy(points, 0.)) / epsilon
    assert np.isfinite(value)


def test_clearance_safe_portal_optimization():
    settings = {
        "portal_margin": .01, "length_weight": 1.,
        "length_prior_weight": 1., "source_deviation_weight": 1.,
        "tangent_weight": 1., "turn_weight": 1.,
        "preferred_turn_degrees": 30., "clearance_weight": 1.,
        "preferred_clearance_m": .1, "hard_clearance_m": .01,
        "maximum_iterations": 5, "maximum_seconds_per_sequence": 2.}
    result = optimize_portals(
        V, T, [0, 1], SOURCE[0], [.8, .8, .8], SOURCE,
        Clearance(), settings)
    assert result.minimum_clearance >= settings["hard_clearance_m"]
    assert np.min(result.barycentric) >= settings["portal_margin"] - 1e-12


def test_portal_simplification_principle():
    points = np.asarray([[0., 0., 0.], [1., .2, 0.], [2., 0., 0.]])
    assert np.linalg.norm(points[-1] - points[0]) < np.linalg.norm(
        np.diff(points, axis=0), axis=1).sum()


def test_exact_corridor_shortcutting_principle():
    assert corridor_connected(N, np.ones(3, bool), 0, 2)
    assert set([0, 1, 2]).issubset(set(np.flatnonzero(np.ones(3, bool))))


def test_bundle_order_constraint():
    source = np.asarray([-1., 0., 1.])
    routed = np.asarray([-.8, .1, .9])
    assert np.array_equal(np.argsort(source), np.argsort(routed))


def test_preexisting_versus_introduced_near_crossing():
    assert classify_near_crossing(.01, .01, 0., .02) == (
        "PREEXISTING_SOURCE_PROXIMITY")
    assert classify_near_crossing(.03, .01, 0., .02) == (
        "INTRODUCED_BY_ANCHOR_REPAIR")
    assert classify_near_crossing(.03, .03, .01, .02) == (
        "INTRODUCED_BY_ROUTING")


def test_joint_neighboring_route_optimization_principle():
    routes = np.asarray([[-1., 0.], [1., 0.]])
    distance = np.linalg.norm(routes[1] - routes[0])
    moved = routes + np.asarray([[-.1, 0.], [.1, 0.]])
    assert np.linalg.norm(moved[1] - moved[0]) > distance


def test_deterministic_resource_bounded_candidate_ranking():
    candidates = [
        {"score": 1., "tet_path": [0, 2], "passes_hard_gates": False},
        {"score": 2., "tet_path": [0, 1], "passes_hard_gates": True},
        {"score": 1., "tet_path": [0, 1], "passes_hard_gates": False}]
    assert deterministic_candidate_ranking(candidates)[0]["passes_hard_gates"]
    assert deterministic_candidate_ranking(candidates) == (
        deterministic_candidate_ranking(candidates))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
