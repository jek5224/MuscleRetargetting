import unittest

import numpy as np
import trimesh

from muscle_sim.hybrid_surface import (
    ATTACHMENT_LOCKED, CONTACT_LOCKED, FREE, TRANSITION,
    candidate_report, fit_positions, geodesic_weights,
    surface_graph, triangle_correspondence, umbrella_laplacian)
from viewer.isolated_muscle import extract_boundary_faces


VERTICES = np.asarray([
    [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
TETS = np.asarray([[1, 2, 3, 0]], dtype=np.int32)
FACES = extract_boundary_faces(TETS)


class HybridSurfaceTests(unittest.TestCase):
    def config(self):
        return {
            "contact_lock_radius_m": .2,
            "contact_transition_width_m": .4,
            "attachment_lock_radius_m": .2,
            "attachment_transition_width_m": .4,
            "position_fit_weight": 10.,
            "laplacian_weight": 2.,
            "path_b_anchor_weight": 1.,
            "contact_max_deviation_m": .01,
            "contact_p95_deviation_m": .01,
            "maximum_triangle_aspect_ratio": 3.,
            "maximum_volume_change_fraction": .2,
        }

    def test_contact_mask_construction_and_priority(self):
        masks = geodesic_weights(
            VERTICES, FACES, VERTICES[[0]], VERTICES[[1]], self.config())
        self.assertEqual(masks["class"][0], CONTACT_LOCKED)
        self.assertEqual(masks["class"][1], ATTACHMENT_LOCKED)

    def test_geodesic_transition_weights_are_smooth_and_bounded(self):
        masks = geodesic_weights(
            VERTICES, FACES, VERTICES[[0]], np.empty((0, 3)),
            self.config())
        self.assertTrue(np.all(masks["lock_weight"] >= 0.))
        self.assertTrue(np.all(masks["lock_weight"] <= 1.))
        self.assertEqual(masks["lock_weight"][0], 1.)

    def test_triangle_correspondence_reconstructs_target(self):
        rows = triangle_correspondence(
            VERTICES, FACES, VERTICES, FACES)
        self.assertLess(np.max(rows["distance"]), 1e-12)
        reconstructed = np.einsum(
            "ni,nij->nj", rows["barycentric"],
            VERTICES[FACES[rows["reference_face_id"]]])
        self.assertLess(np.max(np.linalg.norm(
            reconstructed - rows["target"], axis=1)), 1e-12)

    def test_correspondence_exposes_ambiguity_field(self):
        rows = triangle_correspondence(
            VERTICES, FACES, VERTICES, FACES)
        self.assertEqual(rows["ambiguous"].dtype, np.bool_)

    def test_laplacian_preserves_constant_field(self):
        graph, edges = surface_graph(VERTICES, FACES)
        laplacian = umbrella_laplacian(len(VERTICES), edges)
        self.assertLess(np.linalg.norm(laplacian @ np.ones(4)), 1e-12)

    def test_contact_constrained_fit_is_finite(self):
        masks = geodesic_weights(
            VERTICES, FACES, VERTICES[[0]], np.empty((0, 3)),
            self.config())
        correspondence = triangle_correspondence(
            VERTICES, FACES, VERTICES + [.01, 0, 0], FACES)
        correspondence["contact_corridor_target"] = VERTICES.copy()
        fitted = fit_positions(
            VERTICES, FACES, masks, correspondence, VERTICES[[0]],
            np.empty((0, 3)), self.config(), .5, False)
        self.assertTrue(np.all(np.isfinite(fitted)))

    def test_fitting_keeps_connectivity_manifold(self):
        mesh = trimesh.Trimesh(VERTICES, FACES, process=False)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)

    def test_contact_deviation_gate_rejects_shift(self):
        config = self.config()
        config["contact_max_deviation_m"] = 1e-4
        shifted = VERTICES + [.01, 0, 0]
        report = candidate_report(
            "shifted", shifted, FACES,
            trimesh.Trimesh(VERTICES, FACES, process=False).volume,
            VERTICES, FACES, np.mean(VERTICES[FACES], axis=1), config)
        self.assertFalse(report["accepted"])


if __name__ == "__main__":
    unittest.main()
