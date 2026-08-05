import unittest

import numpy as np

from muscle_sim.full_retetrahedralization import (
    contact_correspondence, render_embedding, transfer_attachment_faces)
from muscle_sim.full_surface import (
    boundary_loops, classify_loops, complete_topology,
    geometric_comparison)
from muscle_sim.local_remeshing import tet_quality
from viewer.isolated_muscle import (
    AttachmentPatch, MuscleData, extract_boundary_faces)


VERTICES = np.asarray([
    [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
TETS = np.asarray([[1, 2, 3, 0]], dtype=np.int32)
FACES = extract_boundary_faces(TETS)


class FullRetetrahedralizationTests(unittest.TestCase):
    def test_manifoldness_and_orientation(self):
        report = complete_topology(VERTICES, FACES)
        self.assertTrue(report["watertight"])
        self.assertTrue(report["two_manifold"])
        self.assertTrue(report["consistently_oriented"])

    def test_unresolved_topology_rejected(self):
        report = complete_topology(VERTICES, FACES[:-1])
        self.assertFalse(report["valid_topology"])
        self.assertTrue(report["boundary_edges"])

    def test_boundary_loop_and_attachment_classification(self):
        faces = FACES[:-1]
        loops = boundary_loops(faces)
        patch = AttachmentPatch(
            "bone", np.asarray(loops[0]), end_type=0)
        rows = classify_loops(VERTICES, loops, [patch])
        self.assertEqual(rows[0]["classification"],
                         "ORIGIN_ATTACHMENT_OPENING")

    def test_four_sheet_branch_detected(self):
        vertices = np.vstack((VERTICES, [[0., -1., 0.], [0., 0., -1.]]))
        faces = np.vstack((FACES, [[0, 1, 4], [1, 0, 5]]))
        report = complete_topology(vertices, faces)
        self.assertIn([0, 1], report["nonmanifold_edges"])

    def test_geometric_deviation_is_measured(self):
        shifted = VERTICES + [1e-3, 0., 0.]
        report = geometric_comparison(
            VERTICES, FACES, shifted, FACES)
        self.assertGreater(report["maximum_bidirectional_distance"], 0.)

    def test_attachment_transfer_uses_geometry(self):
        patch = AttachmentPatch("bone", np.asarray([0, 1, 2]), end_type=0)
        muscle = MuscleData(
            "m", VERTICES, TETS, FACES, [patch], [], {})
        rows, face_ids = transfer_attachment_faces(
            VERTICES, FACES, muscle, 2.0)
        self.assertEqual(rows[0]["bone_id"], "bone")
        self.assertTrue(len(face_ids))

    def test_contact_surface_correspondence(self):
        rows = contact_correspondence(
            VERTICES, FACES, VERTICES, FACES,
            np.mean(VERTICES[FACES[:1]], axis=1), 1e-6)
        self.assertEqual(len(rows), 1)
        self.assertLess(rows[0]["distance"], 1e-12)

    def test_render_embedding_is_barycentric(self):
        rows = render_embedding(VERTICES, VERTICES, TETS, 1e-12)
        self.assertEqual(len(rows), len(VERTICES))
        self.assertLess(max(row["error"] for row in rows), 1e-12)

    def test_tet_quality_hard_rejection_metric(self):
        sliver = VERTICES.copy()
        sliver[3, 2] = 1e-6
        quality = tet_quality(sliver, TETS)
        self.assertLess(quality["minimum_dihedral_degrees"][0], 2.0)


if __name__ == "__main__":
    unittest.main()
