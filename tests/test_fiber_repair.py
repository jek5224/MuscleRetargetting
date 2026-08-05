import numpy as np
import trimesh
import unittest

from muscle_sim.fiber_repair import (
    SURFACE_CLASSES, classify_run, contiguous_runs, endpoint_run_targets,
    fiber_shape_metrics, medial_interior_target, polyline_tangents,
    ray_polygon_radius, sample_complete_segments, signed_clearance,
    surface_comparison_class)


def box():
    return trimesh.creation.box(extents=(2., 2., 2.))


def test_outside_run_classification():
    assert contiguous_runs([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert classify_run(2, 2, 6, np.ones(6), 2, .1) == (
        "SHORT_CONTIGUOUS_OUTSIDE_RUN")
    assert classify_run(0, 1, 6, np.ones(6), 2, .1) == (
        "ENDPOINT_OUTSIDE_NEAR_ATTACHMENT")


def test_multi_surface_comparison_classes():
    assert surface_comparison_class(
        [1, 1, 1, 0], [1.] * 4, 1e-4) == "OUTSIDE_ONLY_C3"
    assert surface_comparison_class(
        [0, 1, 1, 0], [1.] * 4, 1e-4) == "OUTSIDE_ORIGINAL_REFERENCE"
    assert set(SURFACE_CLASSES)


def test_signed_clearance_inside_positive():
    signed, _, _ = signed_clearance(
        box(), np.asarray([[0., 0., 0.], [2., 0., 0.]]))
    assert signed[0] > 0
    assert signed[1] < 0


def test_polyline_tangent_order():
    tangent = polyline_tangents([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert np.allclose(tangent, [1., 0., 0.])


def test_ray_polygon_radius():
    polygon = np.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    assert np.isclose(ray_polygon_radius([0, 0], [1, 0], polygon), 1.)


def test_endpoint_side_preservation():
    surface = box()
    points = np.asarray([[1.1, 0, 0], [.5, 0, 0], [0, 0, 0.]])
    repaired, meta = endpoint_run_targets(points, 0, 0, surface, .01)
    assert signed_clearance(surface, repaired[:1])[0][0] >= .01
    assert meta[0]["anchor_sample"] > 0
    assert repaired[0, 0] > 0


def test_segment_subdivision_preserves_order():
    samples, owners = sample_complete_segments(
        np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), 4)
    assert np.all(np.diff(samples[:, 0]) > 0)
    assert owners.tolist() == [0, 0, 0, 1, 1, 1]


def test_medial_target_is_inside():
    surface = box()
    frames = {
        "origin": np.asarray([[0., 0., 0.]]),
        "tangent": np.asarray([[1., 0., 0.]]),
        "u": np.asarray([[0., 1., 0.]]),
        "v": np.asarray([[0., 0., 1.]])}
    repaired = medial_interior_target(
        np.asarray([1.1, 0., 0.]), surface, frames, .01)
    assert signed_clearance(surface, repaired[None])[0][0] >= .01


def test_rest_length_metric_uses_repaired_rest():
    class Fiber:
        fiber_index = 0
        rest_points = np.asarray([[0., 0, 0], [.5, 0, 0]])
        rest_segment_lengths = np.asarray([.5])
    repaired = [{
        "fiber_index": 0, "points": Fiber.rest_points.copy(),
        "source_sample_parameter": np.asarray([0., 1.]),
        "simulation_repaired_rest_lengths": np.asarray([.5])}]
    config = {
        "interior_margin_m": 1e-4,
        "maximum_sample_displacement_m": .1,
        "maximum_total_length_relative_error": .1,
        "maximum_segment_length_relative_error": .1,
        "maximum_tangent_change_degrees": 10.}
    result = fiber_shape_metrics([Fiber()], repaired, box(), config)
    assert result["fiber_rows"][0]["total_length_relative_error"] == 0


def test_displacement_gate_rejects_large_change():
    class Fiber:
        fiber_index = 0
        rest_points = np.asarray([[0., 0, 0], [.5, 0, 0]])
        rest_segment_lengths = np.asarray([.5])
    repaired = [{
        "fiber_index": 0, "points": np.asarray([[0., 0, 0], [.5, .2, 0]]),
        "source_sample_parameter": np.asarray([0., 1.]),
        "simulation_repaired_rest_lengths": np.asarray([.54])}]
    config = {
        "interior_margin_m": 1e-4,
        "maximum_sample_displacement_m": .1,
        "maximum_total_length_relative_error": 1.,
        "maximum_segment_length_relative_error": 1.,
        "maximum_tangent_change_degrees": 90.}
    assert not fiber_shape_metrics(
        [Fiber()], repaired, box(), config)["gates"][
            "maximum_sample_displacement"]


def test_sample_role_vocabulary():
    assert {"MUSCLE_FIBER_INTERIOR", "FIBER_ENDPOINT",
            "APONEUROSIS_ENDPOINT", "TENDON_PATH",
            "ATTACHMENT_PATH", "UNKNOWN"}


def test_crossing_detector_samples_segment_interior():
    samples, _ = sample_complete_segments(
        np.asarray([[-2., 0, 0], [2., 0, 0]]), 8)
    assert len(samples) == 7
    assert np.any(signed_clearance(box(), samples)[0] > 0)


def test_ordering_source_parameters_monotone():
    values = np.r_[0., np.linspace(0., 1., 9)[1:-1], 1.]
    assert np.all(np.diff(values) > 0)


def test_repaired_length_is_current_geometry():
    points = np.asarray([[0., 0, 0], [.2, .1, 0], [.4, 0, 0]])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.isclose(lengths.sum(), 2. * np.sqrt(.05))


def test_surface_class_ambiguity_has_priority():
    assert surface_comparison_class(
        [False] * 4, [1e-7, 1., 1., 1.], 1e-6
    ) == "AMBIGUOUS_NEAR_BOUNDARY"


def test_endpoint_repair_deterministic():
    surface = box()
    points = np.asarray([[1.1, 0, 0], [.5, 0, 0], [0, 0, 0.]])
    first = endpoint_run_targets(points, 0, 0, surface, .01)[0]
    second = endpoint_run_targets(points, 0, 0, surface, .01)[0]
    assert np.array_equal(first, second)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(
                value, description=name))
    return suite
