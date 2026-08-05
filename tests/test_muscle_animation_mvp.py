"""Focused reliability tests for the independent animation MVP."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muscle_sim.mvp.core import (
    attachment_update, boundary_faces, centerline_from_endpoints,
    closest_rotation, orient_tets, pathological_tets, pose_substeps,
    project_bone_collision, project_global_volume, project_pair_collision,
    signed_tet_volumes, sweep_deform, wrap_polyline_capsule)
from muscle_sim.mvp.io import write_obj


V = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
              [0., 0., 1.]])
T = np.array([[0, 1, 2, 3]])


class MuscleAnimationMVPTests(unittest.TestCase):
    def test_bvh_loading_fixture(self):
        text = Path("data/motion/left_thigh_quasistatic_5pose.bvh").read_text()
        self.assertIn("MOTION", text)
        self.assertIn("Frames:\t5", text)

    def test_skeleton_joint_mapping_fixture(self):
        text = Path("data/zygote_skel.xml").read_text()
        self.assertIn("L_Femur0", text)
        self.assertIn("L_Tibia_Fibula0", text)

    def test_bone_local_attachment_update(self):
        result = attachment_update([[1., 0., 0.]], np.eye(3), [0., 2., 0.])
        np.testing.assert_allclose(result, [[1., 2., 0.]])

    def test_tet_arap_local_projection(self):
        rotation = closest_rotation(np.diag([1.2, .9, 1.]))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)

    def test_global_deformation_solve(self):
        guide = np.array([[0., 0., 0.], [0., 0., 1.]])
        posed = guide + [1., 0., 0.]
        result = sweep_deform(V, guide, posed)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_volume_projection(self):
        scaled = V * 2.
        corrected = project_global_volume(scaled, V)
        self.assertLess(np.linalg.norm(corrected - V), np.linalg.norm(scaled - V))

    def test_pathological_tet_regularization(self):
        bad, _ = pathological_tets(V * [1., 1., 1e-8], T)
        self.assertEqual(bad.tolist(), [0])

    def test_automatic_centerline_generation(self):
        guide = centerline_from_endpoints(V, [0, 1], [2, 3], 7)
        self.assertEqual(guide.shape, (7, 3))

    def test_centerline_endpoint_attachment(self):
        guide = centerline_from_endpoints(V, [0], [3], 5)
        np.testing.assert_allclose(guide[[0, -1]], V[[0, 3]])

    def test_bone_wrapping(self):
        line = np.array([[-2., 0., 0.], [0., 0., 0.], [2., 0., 0.]])
        wrapped = wrap_polyline_capsule(line, np.zeros(3), 1., [0., 1., 0.])
        self.assertGreater(np.linalg.norm(wrapped[1]), 0.)

    def test_centerline_to_volume_coupling(self):
        guide = np.array([[0., 0., 0.], [0., 0., 1.]])
        result = sweep_deform(V, guide, guide + [1., 0., 0.])
        self.assertGreater(result[:, 0].mean(), V[:, 0].mean())

    def test_bone_collision_projection(self):
        result, count, _ = project_bone_collision(
            np.array([[.01, 0., 0.]]), [0], np.array([[0., 0., 0.]]), .02)
        self.assertEqual(count, 1)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_muscle_collision_projection(self):
        a, b, count, _ = project_pair_collision(
            [[0., 0., 0.]], [0], [[0.0001, 0., 0.]], [0], .001)
        self.assertEqual(count, 1)
        self.assertGreater(np.linalg.norm(a[0] - b[0]), .0001)

    def test_temporal_continuation(self):
        previous, target = V, V + 1.
        intermediate = .5 * previous + .5 * target
        self.assertTrue(np.all(intermediate > previous))

    def test_deep_flexion_substepping(self):
        self.assertGreater(pose_substeps(90., deep_count=6),
                           pose_substeps(0., deep_count=6))

    def test_output_cache_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.obj"
            write_obj(path, V, boundary_faces(T))
            self.assertEqual(sum(row.startswith("v ") for row in
                                 path.read_text().splitlines()), len(V))

    def test_orientation_helper(self):
        oriented = orient_tets(V, T[:, [0, 2, 1, 3]])
        self.assertGreater(signed_tet_volumes(V, oriented)[0], 0.)


if __name__ == "__main__":
    unittest.main()
