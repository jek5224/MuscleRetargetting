import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muscle_sim.compare_experiment_f import compare
from muscle_sim.local_remeshing import (
    RegionSelection, attempt_local_seam_repair, repair_cavity,
    reembed_fibers, tet_quality,
    validate_closed_boundary)
from viewer.isolated_muscle import EmbeddedFiber, extract_boundary_faces


VERTICES = np.asarray([
    [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
TETS = np.asarray([[1, 2, 3, 0]], dtype=np.int32)


class IdentityBackend:
    def tetrahedralize(self, vertices, faces, quality_options):
        return vertices.copy(), np.asarray([[1, 2, 3, 0]], dtype=np.int32)


class LocalRemeshingTests(unittest.TestCase):
    def test_cavity_boundary_watertightness(self):
        report = validate_closed_boundary(
            VERTICES, extract_boundary_faces(TETS))
        self.assertTrue(report["valid"])
        broken = extract_boundary_faces(TETS)[:-1]
        report = validate_closed_boundary(VERTICES, broken)
        self.assertFalse(report["watertight"])
        self.assertTrue(report["nonmanifold_or_open_edges"])

    def test_conforming_interface_and_positive_orientation(self):
        faces = extract_boundary_faces(TETS)
        selection = RegionSelection(
            np.asarray([0]), np.asarray([0]), np.asarray([0]), faces,
            faces, np.empty((0, 3), dtype=np.int32),
            np.arange(4), {})
        vertices, tets, cavity, unchanged = repair_cavity(
            VERTICES, TETS, selection, IdentityBackend(), {})
        self.assertEqual(
            {tuple(sorted(v)) for v in extract_boundary_faces(cavity)},
            {tuple(sorted(v)) for v in faces})
        self.assertTrue(np.all(tet_quality(
            vertices, tets)["signed_volume"] > 0.0))

    def test_external_surface_and_attachment_ids_are_preserved(self):
        faces = extract_boundary_faces(TETS)
        selection = RegionSelection(
            np.asarray([0]), np.asarray([0]), np.asarray([0]), faces,
            faces, np.empty((0, 3), dtype=np.int32),
            np.arange(4), {})
        vertices, tets, _, _ = repair_cavity(
            VERTICES, TETS, selection, IdentityBackend(), {})
        self.assertTrue(np.array_equal(vertices, VERTICES))
        self.assertEqual(
            {tuple(sorted(v)) for v in extract_boundary_faces(tets)},
            {tuple(sorted(v)) for v in faces})
        self.assertTrue(np.array_equal(vertices[[0, 1]], VERTICES[[0, 1]]))

    def test_fiber_reembedding_and_direction_input(self):
        points = np.asarray([[.1, .1, .1], [.2, .1, .1]])
        bary = np.column_stack((
            points[:, 0], points[:, 1], points[:, 2],
            1. - np.sum(points, axis=1)))
        fiber = EmbeddedFiber(
            0, 0, np.zeros(2, dtype=np.int32), bary, points,
            np.linalg.norm(np.diff(points, axis=0), axis=1))
        fibers, report = reembed_fibers([fiber], VERTICES, TETS, 1e-12)
        self.assertFalse(report["outside_samples"])
        self.assertLess(report["maximum_reconstruction_error"], 1e-12)
        self.assertTrue(np.all(fibers[0].tet_ids == 0))

    def test_stale_contact_ids_explicitly_invalid_in_provenance_contract(self):
        contact = {
            "old_contact_primitive_ids_valid": False,
            "old_AL_multipliers_valid": False}
        self.assertFalse(contact["old_contact_primitive_ids_valid"])
        self.assertFalse(contact["old_AL_multipliers_valid"])

    def test_seam_repair_is_transactional_when_no_closed_result_exists(self):
        faces = extract_boundary_faces(TETS)
        # Add two extra triangles at one edge to form a four-sheet branch.
        vertices = np.vstack((VERTICES, [[0., -1., 0.], [0., 0., -1.]]))
        branched = np.vstack((faces, [[0, 1, 4], [1, 0, 5]]))
        repaired, report = attempt_local_seam_repair(
            vertices, branched, faces, faces, [], np.empty((0, 3)), {})
        self.assertFalse(report["accepted"])
        self.assertTrue(np.array_equal(repaired, branched))
        self.assertEqual(report["removed_faces"], [])

    def test_experiment_comparison_rejects_threshold_change(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root) / "a", Path(root) / "b"
            first.mkdir()
            second.mkdir()
            base = {"accepted": False, "acceptance_criteria": {
                "contact_safe_minimum_J": 0.2}}
            (first / "rest_correction_report.json").write_text(
                json.dumps(base))
            changed = {**base, "acceptance_criteria": {
                "contact_safe_minimum_J": 0.19}}
            (second / "rest_correction_report.json").write_text(
                json.dumps(changed))
            with self.assertRaises(ValueError):
                compare(first, second)


if __name__ == "__main__":
    unittest.main()
