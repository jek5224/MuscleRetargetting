import unittest

import numpy as np

from muscle_sim.layered_meshing import (
    optimize_isolated_sliver, try_two_to_three_flip, union_boundary)
from muscle_sim.local_remeshing import tet_quality
from viewer.isolated_muscle import extract_boundary_faces, orient_tetrahedra


class LayeredMeshingTests(unittest.TestCase):
    def bipyramid(self):
        vertices = np.asarray([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            [.3, .3, .01], [.3, .3, -1.]])
        tets = orient_tetrahedra(
            vertices, np.asarray([[0, 1, 2, 3], [0, 2, 1, 4]]))
        return vertices, tets

    def test_two_to_three_flip_preserves_union_boundary(self):
        vertices, tets = self.bipyramid()
        result = try_two_to_three_flip(vertices, tets, 0, 1)
        self.assertIsNotNone(result)
        replacement, _ = result
        self.assertEqual(union_boundary(tets), union_boundary(replacement))

    def test_flip_tets_have_positive_orientation(self):
        vertices, tets = self.bipyramid()
        replacement, quality = try_two_to_three_flip(
            vertices, tets, 0, 1)
        self.assertTrue(np.all(quality["signed_volume"] > 0.))

    def test_outer_boundary_is_exact_not_tolerance_merged(self):
        vertices, tets = self.bipyramid()
        replacement, _ = try_two_to_three_flip(vertices, tets, 0, 1)
        before = {tuple(sorted(face))
                  for face in extract_boundary_faces(tets)}
        after = {tuple(sorted(face))
                 for face in extract_boundary_faces(replacement)}
        self.assertEqual(before, after)

    def test_quality_metrics_detect_sliver(self):
        vertices, tets = self.bipyramid()
        quality = tet_quality(vertices, tets)
        self.assertLess(
            np.min(quality["minimum_dihedral_degrees"]), 2.)


if __name__ == "__main__":
    unittest.main()
