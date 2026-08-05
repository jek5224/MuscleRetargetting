import json
import unittest

import numpy as np
import trimesh

from muscle_sim.fiber_routing import (
    ClearanceField, arc_length_weighted_tet_directions, astar_tet_path,
    build_tet_adjacency, classify_straight_segment, crossing_points,
    edge_cost, endpoint_tangent_errors, face_barycentric,
    introduced_crossings, maximum_turn, refine_route, route_length,
    segment_segment_distance, shared_face, source_curve_prior,
    tet_barycentric, validate_exact_route)


VERTICES = np.asarray([
    [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
    [1., 1., 1.]])
TETS = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
SETTINGS = {
    "interior_margin_m": 0.,
    "preferred_clearance_m": .2,
    "boundary_penalty_weight": 2.,
    "source_deviation_weight": 1.,
    "tangent_weight": 1.,
    "maximum_route_corridor_radius_m": 10.,
    "maximum_astar_expansions": 100}


class ConstantClearance:
    def __init__(self, value=1.):
        self.value = value

    def approximate(self, points):
        return np.full(len(points), self.value)


def test_invalid_straight_chord_classification():
    surface = trimesh.creation.box(extents=(2., 2., 2.))
    result = classify_straight_segment(
        surface, np.asarray([0., 0., 0.]), np.asarray([2., 0., 0.]), 0.)
    assert result["classification"] != "STRAIGHT_SEGMENT_VALID"
    assert result["maximum_outside_depth_m"] > 0


def test_tet_adjacency_graph_construction():
    neighbors, local = build_tet_adjacency(TETS)
    assert 1 in neighbors[0] and 0 in neighbors[1]
    assert np.count_nonzero(neighbors >= 0) == 2
    assert np.count_nonzero(local >= 0) == 2


def test_deterministic_tet_face_astar():
    neighbors, _ = build_tet_adjacency(TETS)
    args = (VERTICES, TETS, neighbors, 0, 1, VERTICES[0], VERTICES[4],
            ConstantClearance(), SETTINGS)
    assert astar_tet_path(*args) == astar_tet_path(*args) == [0, 1]


def test_clearance_weighted_graph_cost():
    args = (np.asarray([0., 0., 0.]), np.asarray([1., 0., 0.]),
            np.asarray([0., 0., 0.]), np.asarray([1., 0., 0.]),
            np.asarray([1., 0., 0.]))
    assert edge_cost(*args, .01, SETTINGS) > edge_cost(*args, 1., SETTINGS)


def test_shared_face_crossing_generation():
    points, rows = crossing_points(
        VERTICES, TETS, [0, 1], [.1, .1, .1], [.8, .8, .8],
        ConstantClearance(), SETTINGS)
    assert len(rows) == 1
    assert np.allclose(np.sum(rows[0]["face_barycentric"]), 1.)
    assert np.min(rows[0]["face_barycentric"]) >= 0.
    assert points.shape == (3, 3)


def test_exact_per_tet_subsegment_containment():
    points, _ = crossing_points(
        VERTICES, TETS, [0, 1], [.1, .1, .1], [.8, .8, .8],
        ConstantClearance(), SETTINGS)
    assert validate_exact_route(VERTICES, TETS, [0, 1], points)["valid"]


def test_source_curve_prior_interpolates_anchors():
    anchors = np.asarray([[0., 0., 0.], [1., .2, 0.], [2., 0., 0.]])
    prior = source_curve_prior(anchors, 4)
    assert np.allclose(prior[[0, 4, 8]], anchors)


def test_endpoint_tangent_preservation_metric():
    points = np.asarray([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    assert np.allclose(endpoint_tangent_errors(points[:2], points, 0), 0.)


def test_adaptive_inserted_sample_generation_honors_cap():
    points, owners, parameters = refine_route(
        [[0., 0., 0.], [10., 0., 0.]], [0], [0., 1.], .1, 30., 7)
    assert len(points) - 2 <= 7
    assert len(owners) == len(points) - 1
    assert np.all(np.diff(parameters) > 0)


def test_complete_routed_curve_containment():
    crossing, _ = crossing_points(
        VERTICES, TETS, [0, 1], [.1, .1, .1], [.8, .8, .8],
        ConstantClearance(), SETTINGS)
    check = validate_exact_route(VERTICES, TETS, [0, 1], crossing)
    assert all(row["valid"] for row in check["subsegments"])


def test_fiber_fiber_crossing_detection():
    routes = [
        {"fiber_index": 0, "points": np.asarray([[-1., 0., 0.], [1., 0., 0.]]),
         "source_sample_parameter": np.asarray([0., 1.])},
        {"fiber_index": 1, "points": np.asarray([[0., -1., 0.], [0., 1., 0.]]),
         "source_sample_parameter": np.asarray([0., 1.])}]
    assert introduced_crossings(routes, 1e-6)
    assert segment_segment_distance(
        routes[0]["points"], routes[1]["points"]) == 0.


def test_bundle_order_preservation_by_section_sign():
    before = np.asarray([[-1., 0.], [0., 0.], [1., 0.]])
    after = np.asarray([[-1., .1], [0., -.1], [1., .1]])
    assert np.array_equal(np.argsort(before[:, 0]), np.argsort(after[:, 0]))


def test_route_length_gate_metric():
    assert route_length([[0., 0., 0.], [1., 0., 0.]]) == 1.
    assert abs(route_length([[0., 0., 0.], [1., 0., 0.]]) - 1.) <= .01


def test_route_curvature_gate_metric():
    assert maximum_turn([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]) == 0.
    assert maximum_turn([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]]) == 90.


def test_zero_prestress_routed_fiber():
    points = np.asarray([[0., 0., 0.], [.5, .1, 0.], [1., 0., 0.]])
    rest = np.linalg.norm(np.diff(points, axis=0), axis=1)
    stretch = np.linalg.norm(np.diff(points, axis=0), axis=1) / rest
    assert np.allclose(stretch, 1.)


def test_routed_deformation_continuity_across_tet_faces():
    face = shared_face(TETS, 0, 1)
    point = VERTICES[face].mean(axis=0)
    first = tet_barycentric(VERTICES, TETS, 0, point)
    second = tet_barycentric(VERTICES, TETS, 1, point)
    deformation = VERTICES + np.asarray([.2, -.1, .3])
    assert np.allclose(
        first @ deformation[TETS[0]], second @ deformation[TETS[1]])


def test_arc_length_weighted_anisotropy_reconstruction():
    route = {"points": np.asarray([[0., 0., 0.], [.25, 0., 0.],
                                   [1., 0., 0.]]),
             "subsegment_owner_tets": np.asarray([0, 0])}
    direction, covered = arc_length_weighted_tet_directions(
        [route], VERTICES, TETS)
    assert np.allclose(direction[0], [1., 0., 0.])
    assert np.isclose(covered[0], 1.)


def test_deterministic_routing_provenance_values():
    neighbors, _ = build_tet_adjacency(TETS)
    path = astar_tet_path(
        VERTICES, TETS, neighbors, 0, 1, [.1, .1, .1], [.8, .8, .8],
        ConstantClearance(), SETTINGS)
    text_a = json.dumps({"tet_path": path}, sort_keys=True)
    text_b = json.dumps({"tet_path": path}, sort_keys=True)
    assert text_a == text_b


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
