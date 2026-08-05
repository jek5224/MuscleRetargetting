import unittest

import numpy as np

from viewer.multimuscle import (
    directed_deformable_contact_energy_gradient,
    normal_cohesion_energy_gradient,
    point_rigid_contact_energy_gradient,
    symmetric_deformable_contact_energy_gradient,
)


TRIANGLE_VERTICES = np.array([
    [-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]])
TRIANGLE_FACES = np.array([[0, 1, 2]], dtype=np.int32)


def directional_check(function, variables, gradients, seed=3, epsilon=1e-7):
    rng = np.random.default_rng(seed)
    directions = [rng.normal(size=value.shape) for value in variables]
    scale = np.sqrt(sum(np.sum(direction ** 2) for direction in directions))
    directions = [direction / scale for direction in directions]
    plus = function(*[
        value + epsilon * direction
        for value, direction in zip(variables, directions)])
    minus = function(*[
        value - epsilon * direction
        for value, direction in zip(variables, directions)])
    numerical = (plus - minus) / (2.0 * epsilon)
    analytical = sum(
        np.sum(gradient * direction)
        for gradient, direction in zip(gradients, directions))
    return abs(numerical - analytical) / max(
        1.0, abs(numerical), abs(analytical))


class MultiMuscleContactTests(unittest.TestCase):
    def test_inside_deformable_contact_gradient(self):
        target = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        faces = np.array([
            [0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
            dtype=np.int32)
        source = np.array([[0.2, 0.2, 0.08]])
        inside = np.array([True])
        energy, gs, gt, _ = directed_deformable_contact_energy_gradient(
            source, np.array([0]), target, faces, 0.04, 11.0, inside)
        error = directional_check(
            lambda a, b: directed_deformable_contact_energy_gradient(
                a, np.array([0]), b, faces, 0.04, 11.0, inside)[0],
            [source, target], [gs, gt])
        self.assertLess(error, 2e-6)
        self.assertGreater(energy, 0.0)

    def test_rigid_contact_gradient(self):
        points = np.array([[0.1, 0.2, 0.03], [-0.2, 0.1, 0.04]])
        args = (TRIANGLE_VERTICES, TRIANGLE_FACES, 0.08, 12.0)
        energy, gradient, diagnostics = (
            point_rigid_contact_energy_gradient(points, *args))
        self.assertEqual(diagnostics.active_count, 2)
        error = directional_check(
            lambda x: point_rigid_contact_energy_gradient(x, *args)[0],
            [points], [gradient])
        self.assertLess(error, 1e-7)
        self.assertGreater(energy, 0.0)

    def test_symmetric_deformable_contact_gradient(self):
        first = TRIANGLE_VERTICES.copy()
        second = TRIANGLE_VERTICES.copy() + [0.12, 0.05, 0.06]
        ids = np.arange(3)
        args = (ids, TRIANGLE_FACES, ids, TRIANGLE_FACES, 0.1, 9.0)
        energy, g1, g2, diagnostics = (
            symmetric_deformable_contact_energy_gradient(
                first, args[0], args[1], second, args[2], args[3],
                args[4], args[5]))
        error = directional_check(
            lambda a, b: symmetric_deformable_contact_energy_gradient(
                a, ids, TRIANGLE_FACES, b, ids, TRIANGLE_FACES,
                0.1, 9.0)[0],
            [first, second], [g1, g2])
        self.assertLess(error, 2e-6)
        self.assertGreater(diagnostics.active_count, 0)
        # Translation invariance is the discrete symmetry/conservation check.
        self.assertLess(np.linalg.norm(np.sum(g1, axis=0)
                                       + np.sum(g2, axis=0)), 1e-10)
        self.assertGreater(energy, 0.0)

    def test_tangential_motion_has_no_contact_force(self):
        points = np.array([[0.0, 0.0, 0.04]])
        _, gradient, _ = point_rigid_contact_energy_gradient(
            points, TRIANGLE_VERTICES, TRIANGLE_FACES, 0.08, 10.0)
        self.assertLess(abs(gradient[0, 0]), 1e-12)
        self.assertLess(abs(gradient[0, 1]), 1e-12)
        self.assertGreater(abs(gradient[0, 2]), 0.0)

    def test_fascia_contact_gradient_inside_side(self):
        points = np.array([[0.0, 0.0, -0.03]])
        inside = np.array([True])
        energy, gradient, diagnostics = point_rigid_contact_energy_gradient(
            points, TRIANGLE_VERTICES, TRIANGLE_FACES, 0.02, 8.0,
            forbidden_mask=inside)
        error = directional_check(
            lambda x: point_rigid_contact_energy_gradient(
                x, TRIANGLE_VERTICES, TRIANGLE_FACES, 0.02, 8.0,
                forbidden_mask=inside)[0],
            [points], [gradient])
        self.assertLess(error, 1e-7)
        self.assertAlmostEqual(diagnostics.maximum_penetration, 0.05)
        self.assertGreater(energy, 0.0)

    def test_normal_cohesion_gradient_and_tangential_freedom(self):
        first = np.array([[0.0, 0.0, 0.0]])
        second = np.array([[0.3, -0.2, 0.08]])
        ids = np.array([0])
        normals = np.array([[0.0, 0.0, 1.0]])
        args = (ids, ids, normals, 0.03, 1.0, 7.0)
        energy, g1, g2, active = normal_cohesion_energy_gradient(
            first, second, *args)
        error = directional_check(
            lambda a, b: normal_cohesion_energy_gradient(
                a, b, *args)[0],
            [first, second], [g1, g2])
        self.assertLess(error, 1e-7)
        self.assertEqual(active, 1)
        self.assertEqual(g1[0, 0], 0.0)
        self.assertEqual(g1[0, 1], 0.0)
        self.assertGreater(energy, 0.0)


if __name__ == "__main__":
    unittest.main()
