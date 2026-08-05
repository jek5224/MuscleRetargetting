import unittest

import numpy as np

from muscle_sim.cross_sections import (
    assign_bundle_coordinates, covariance_map, cyclic_correspondence,
    polygon_descriptors, representative_bundle_path,
    resample_closed_contour, resample_polyline,
    rotation_minimizing_frames, section_error, thickness_pairs)
from viewer.isolated_muscle import EmbeddedFiber


class CrossSectionTests(unittest.TestCase):
    def circle(self, count=64, radii=(2., 1.)):
        angle = np.linspace(0., 2. * np.pi, count, endpoint=False)
        return np.column_stack((
            radii[0] * np.cos(angle), radii[1] * np.sin(angle)))

    def test_polyline_resampling(self):
        points = np.asarray([[0., 0., 0.], [0., 0., 2.]])
        result = resample_polyline(points, 5)
        self.assertTrue(np.allclose(result[:, 2], np.linspace(0., 2., 5)))

    def test_representative_bundle_path(self):
        first = np.column_stack((
            np.zeros(4), np.zeros(4), np.linspace(0., 1., 4)))
        fibers = [
            EmbeddedFiber(0, i, np.zeros(4, dtype=np.int32),
                          np.zeros((4, 4)), first + [i * .1, 0, 0],
                          np.ones(3)) for i in range(3)]
        path = representative_bundle_path(fibers, count=8)
        self.assertEqual(path.shape, (8, 3))

    def test_rotation_minimizing_frames_are_orthonormal(self):
        t = np.linspace(0., 1., 20)
        path = np.column_stack((.1 * np.sin(t), t, .1 * np.cos(t)))
        frames = rotation_minimizing_frames(path)
        self.assertLess(np.max(np.abs(np.einsum(
            "ij,ij->i", frames["u"], frames["tangent"]))), 1e-12)
        self.assertTrue(np.allclose(
            np.linalg.norm(frames["v"], axis=1), 1.))

    def test_surface_to_section_coordinate_assignment(self):
        path = np.column_stack((
            np.zeros(10), np.linspace(0., 1., 10), np.zeros(10)))
        frames = rotation_minimizing_frames(path)
        result = assign_bundle_coordinates(
            np.asarray([[.2, .5, .1]]), frames)
        self.assertEqual(len(result["s"]), 1)

    def test_closed_contour_resampling(self):
        points = np.column_stack((self.circle(12), np.zeros(12)))
        sampled = resample_closed_contour(points, 32)
        self.assertEqual(sampled.shape, (32, 3))

    def test_cyclic_contour_correspondence(self):
        reference = self.circle()
        current = np.roll(reference, 13, axis=0)
        mapped, report = cyclic_correspondence(reference, current)
        self.assertLess(np.max(np.linalg.norm(mapped - reference, axis=1)),
                        1e-12)

    def test_section_area_centroid_covariance(self):
        descriptor = polygon_descriptors(self.circle(256))
        self.assertAlmostEqual(descriptor["area"], 2. * np.pi, places=2)
        self.assertLess(np.linalg.norm(descriptor["centroid"]), 1e-12)
        self.assertGreater(descriptor["principal_values"][0],
                           descriptor["principal_values"][1])

    def test_near_circular_orientation_is_not_required(self):
        descriptor = polygon_descriptors(self.circle(radii=(1., 1.)))
        ratio = descriptor["principal_values"][0] / (
            descriptor["principal_values"][1])
        self.assertAlmostEqual(ratio, 1., places=10)

    def test_thickness_pairs_do_not_cross_index_order(self):
        contour = self.circle()
        pairs, lengths = thickness_pairs(contour, pair_count=12)
        self.assertTrue(np.all(np.diff(pairs[:, 0]) >= 0))
        self.assertTrue(np.all(pairs[:, 1] > pairs[:, 0]))
        self.assertTrue(np.all(lengths > 0.))

    def test_covariance_map_matches_target(self):
        current = polygon_descriptors(self.circle(radii=(2., 1.)))
        reference = polygon_descriptors(self.circle(radii=(1., 3.)))
        transform = covariance_map(current, reference)
        mapped = transform @ current["covariance"] @ transform.T
        self.assertTrue(np.allclose(mapped, reference["covariance"],
                                    atol=1e-10))

    def test_section_error_detects_inflation(self):
        reference = polygon_descriptors(self.circle(radii=(1., 1.)))
        current = polygon_descriptors(self.circle(radii=(2., 2.)))
        error = section_error(reference, current)
        self.assertGreater(error["relative_area_error"], 1.)


if __name__ == "__main__":
    unittest.main()
