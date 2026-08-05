import copy
import unittest

import numpy as np
import yaml

from viewer.isolated_muscle import (
    AttachmentPatch, EmbeddedFiber, MuscleData, attachment_targets,
    bind_attachment_patches, build_tet_fiber_directions,
    extract_boundary_faces, harmonic_attachment_update, precompute_energy,
    reconstruct_fiber,
    total_energy_gradient,
)


def synthetic_problem():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.int32)
    points = np.array([
        [0.15, 0.15, 0.15],
        [0.30, 0.15, 0.15],
        [0.42, 0.18, 0.15],
    ])
    barycentric = np.column_stack((
        points[:, 0], points[:, 1], points[:, 2],
        1.0 - np.sum(points, axis=1)))
    # This tet ordering maps beta[0:3] to x/y/z unit vertices only after
    # moving the origin vertex to the fourth slot.
    tetrahedra = np.array([[1, 2, 3, 0]], dtype=np.int32)
    fiber = EmbeddedFiber(
        0, 0, np.zeros(3, dtype=np.int32), barycentric,
        points, np.linalg.norm(np.diff(points, axis=0), axis=1))
    patches = [
        AttachmentPatch("A", np.array([0], dtype=np.int32), end_type=0),
        AttachmentPatch("B", np.array([1], dtype=np.int32), end_type=1),
    ]
    muscle = MuscleData(
        "synthetic", vertices, tetrahedra,
        extract_boundary_faces(tetrahedra), patches, [fiber], {})
    with open("config/isolated_muscle.yaml") as handle:
        config = yaml.safe_load(handle)
    return muscle, config


class IsolatedMuscleTests(unittest.TestCase):
    def test_attachment_transform_correctness(self):
        muscle, _ = synthetic_problem()
        rest = {
            "A": np.eye(4),
            "B": np.eye(4),
        }
        bind_attachment_patches(
            muscle.attachment_patches, muscle.vertices, rest)
        angle = 0.31
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [0.2, -0.1, 0.4]
        targets = attachment_targets(
            muscle.attachment_patches, {"A": transform, "B": transform})
        for vertex, target in targets.items():
            expected = rotation @ muscle.vertices[vertex] + transform[:3, 3]
            self.assertLess(np.linalg.norm(target - expected), 1e-12)

    def test_rest_energy_and_fiber_embedding(self):
        muscle, config = synthetic_problem()
        precomputed = precompute_energy(muscle)
        energy, gradient, terms = total_energy_gradient(
            muscle.vertices, muscle, precomputed, config)
        self.assertLess(abs(energy), 1e-10)
        self.assertLess(np.linalg.norm(gradient), 1e-8)
        reconstructed = reconstruct_fiber(
            muscle.fibers[0], muscle.vertices, muscle.tetrahedra)
        self.assertLess(
            np.max(np.linalg.norm(
                reconstructed - muscle.fibers[0].rest_points, axis=1)),
            1e-12)
        self.assertAlmostEqual(terms["volume_ratio"], 1.0)

    def test_rigid_transform_invariance(self):
        muscle, config = synthetic_problem()
        precomputed = precompute_energy(muscle)
        angle = 0.47
        rotation = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
        moved = muscle.vertices @ rotation.T + [0.3, -0.2, 0.4]
        energy, _, terms = total_energy_gradient(
            moved, muscle, precomputed, config)
        self.assertLess(abs(energy), 1e-9)
        self.assertAlmostEqual(terms["volume_ratio"], 1.0, places=12)

    def test_energy_gradient_finite_difference(self):
        muscle, config = synthetic_problem()
        precomputed = precompute_energy(muscle)
        moved = muscle.vertices.copy()
        moved[1] += [0.03, 0.01, -0.005]
        moved[2] += [-0.01, 0.02, 0.007]
        energy, gradient, _ = total_energy_gradient(
            moved, muscle, precomputed, config)
        self.assertTrue(np.isfinite(energy))
        direction = np.random.default_rng(4).normal(size=moved.shape)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-7
        plus = total_energy_gradient(
            moved + epsilon * direction,
            muscle, precomputed, config)[0]
        minus = total_energy_gradient(
            moved - epsilon * direction,
            muscle, precomputed, config)[0]
        numerical = (plus - minus) / (2.0 * epsilon)
        analytical = float(np.sum(gradient * direction))
        relative = abs(numerical - analytical) / max(
            1.0, abs(numerical), abs(analytical))
        self.assertLess(relative, 2e-6)

    def test_each_new_energy_gradient_finite_difference(self):
        muscle, base_config = synthetic_problem()
        precomputed = precompute_energy(muscle)
        moved = muscle.vertices.copy()
        moved[1] += [0.025, 0.008, -0.003]
        moved[2] += [-0.006, 0.016, 0.004]
        direction = np.random.default_rng(8).normal(size=moved.shape)
        direction /= np.linalg.norm(direction)
        for term in ("matrix_volume", "fiber", "bending"):
            config = copy.deepcopy(base_config)
            if term == "matrix_volume":
                config["fiber"]["anisotropy_enabled"] = False
                config["fiber"]["bending_enabled"] = False
            elif term == "fiber":
                config["material"]["shear_modulus"] = 0.0
                config["material"]["bulk_modulus"] = 0.0
                config["fiber"]["bending_enabled"] = False
            else:
                config["material"]["shear_modulus"] = 0.0
                config["material"]["bulk_modulus"] = 0.0
                config["fiber"]["anisotropy_enabled"] = False
            _, gradient, _ = total_energy_gradient(
                moved, muscle, precomputed, config)
            epsilon = 1e-7
            plus = total_energy_gradient(
                moved + epsilon * direction,
                muscle, precomputed, config)[0]
            minus = total_energy_gradient(
                moved - epsilon * direction,
                muscle, precomputed, config)[0]
            numerical = (plus - minus) / (2.0 * epsilon)
            analytical = float(np.sum(gradient * direction))
            relative = abs(numerical - analytical) / max(
                1.0, abs(numerical), abs(analytical))
            self.assertLess(relative, 3e-6, term)

    def test_small_attachment_displacement_is_exact_and_valid(self):
        muscle, config = synthetic_problem()
        precomputed = precompute_energy(muscle)
        targets = {
            0: muscle.vertices[0],
            1: muscle.vertices[1] + np.array([0.002, 0.001, 0.0]),
        }
        moved = harmonic_attachment_update(
            muscle.vertices, targets, precomputed)
        self.assertLess(np.linalg.norm(moved[0] - targets[0]), 1e-12)
        self.assertLess(np.linalg.norm(moved[1] - targets[1]), 1e-12)
        energy, _, terms = total_energy_gradient(
            moved, muscle, precomputed, config)
        self.assertTrue(np.isfinite(energy))
        self.assertGreater(terms["minimum_J"], config["material"]["minimum_J"])

    def test_fiber_direction_comes_from_embedded_curve(self):
        muscle, _ = synthetic_problem()
        direction, valid = build_tet_fiber_directions(
            muscle.vertices, muscle.tetrahedra, muscle.fibers)
        expected = (
            muscle.fibers[0].rest_points[-1]
            - muscle.fibers[0].rest_points[0])
        expected /= np.linalg.norm(expected)
        self.assertTrue(valid[0])
        self.assertGreater(np.dot(direction[0], expected), 0.95)


if __name__ == "__main__":
    unittest.main()
