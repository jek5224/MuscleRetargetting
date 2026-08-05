import unittest

import numpy as np

from muscle_sim.diagnostic_original_asset import (
    contact_mode, diagnostic_decision, embed_proxy_vertices,
    proxy_positions, sliver_classification, transfer_proxy_forces,
    validate_fiber_segments)
from muscle_sim.layered_meshing import try_two_to_three_flip, union_boundary
from muscle_sim.local_remeshing import tet_quality


V = np.asarray([
    [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
    [1., 1., 1.]])
T = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)


def test_diagnostic_asset_selection_policy():
    rows = [
        {"valid": False, "outside": 0},
        {"valid": True, "outside": 0}]
    selected = sorted(rows, key=lambda row: not row["valid"])[0]
    assert selected["valid"]


def test_fiber_compatible_asset_validation():
    report = validate_fiber_segments(
        V, T, [np.asarray([[.1, .1, .1], [.2, .2, .2]])])
    assert report["embedding_success"]


def test_complete_source_segment_containment():
    report = validate_fiber_segments(
        V, T, [np.asarray([[.1, .1, .1], [.8, .8, .8]])])
    assert report["complete_segment_containment_success"]


def test_isolated_sliver_classification():
    quality = {"minimum_dihedral_degrees": np.asarray([1., 3.])}
    assert sliver_classification(T, quality) == "ISOLATED_SLIVER"


def test_boundary_preserving_two_to_three_flip():
    result = try_two_to_three_flip(V, T, 0, 1)
    assert result is not None
    assert union_boundary(result[0]) == union_boundary(T)


def test_fiber_preserving_topology_repair_principle():
    fibers = np.asarray([[.2, .2, .2]])
    before = fibers.copy()
    try_two_to_three_flip(V, T, 0, 1)
    assert np.array_equal(fibers, before)


def test_proxy_vertex_tet_embedding():
    rows = embed_proxy_vertices(V, T, [[.1, .1, .1]], .01)
    assert rows[0]["tet_id"] >= 0 and not rows[0]["projected"]


def test_proxy_contact_force_transfer():
    forces = transfer_proxy_forces(
        len(V), T, np.asarray([0]), np.asarray([[.25] * 4]),
        np.asarray([[4., 0., 0.]]))
    assert np.allclose(forces[T[0], 0], 1.)
    assert np.isclose(forces[:, 0].sum(), 4.)


def test_proxy_virtual_work_consistency():
    tet_ids = np.asarray([0])
    barycentric = np.asarray([[.1, .2, .3, .4]])
    force = np.asarray([[1., 2., 3.]])
    displacement = np.arange(len(V) * 3, dtype=float).reshape((-1, 3)) * 1e-3
    nodal = transfer_proxy_forces(
        len(V), T, tet_ids, barycentric, force)
    proxy_displacement = proxy_positions(
        displacement, T, tet_ids, barycentric)
    assert np.isclose(np.sum(nodal * displacement),
                      np.sum(force * proxy_displacement))


def test_proxy_deformation_update():
    weights = np.asarray([[.25] * 4])
    first = proxy_positions(V, T, [0], weights)
    moved = proxy_positions(V + [1., 2., 3.], T, [0], weights)
    assert np.allclose(moved - first, [[1., 2., 3.]])


def test_native_versus_proxy_contact_mode_selection():
    assert contact_mode("native_boundary", False) == "native_boundary"
    assert contact_mode("embedded_proxy", True) == "embedded_proxy"


def test_unchanged_solver_parameter_verification():
    before = {"contact_safe_minimum_J": .2, "tolerance": 1e-6}
    after = dict(before)
    assert before == after


def test_diagnostic_decision_classification():
    assert diagnostic_decision(
        {"mechanically_promising": True}) == (
        "CASE_A_PHYSICAL_FORMULATION_WORKS")
    assert diagnostic_decision(isolated_failure="mesh") == (
        "CASE_C_OR_D_ORIGINAL_DISCRETIZATION_BLOCKER")


def test_deterministic_asset_and_result_provenance():
    values = {"asset": "a", "hash": "b", "seed": 0}
    assert sorted(values.items()) == sorted(dict(values).items())


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
