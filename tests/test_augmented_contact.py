import unittest

import numpy as np

from viewer.augmented_contact import (
    ActiveContact, PersistentActiveSet,
    al_inequality_value_derivatives, diagnose_infeasible_contacts,
    fischer_burmeister, update_multiplier)


class AugmentedContactTests(unittest.TestCase):
    def test_experiment_a_sign_and_multiplier(self):
        rho, multiplier, gap = 20.0, 0.3, -0.04
        value, derivative, curvature = (
            al_inequality_value_derivatives(gap, multiplier, rho))
        epsilon = 1e-7
        plus = al_inequality_value_derivatives(
            gap + epsilon, multiplier, rho)[0]
        minus = al_inequality_value_derivatives(
            gap - epsilon, multiplier, rho)[0]
        self.assertAlmostEqual(
            derivative, (plus - minus) / (2.0 * epsilon), places=8)
        self.assertEqual(curvature, rho)
        updated = update_multiplier(multiplier, gap, rho)
        self.assertGreater(updated, multiplier)
        self.assertGreater(value, 0.0)

    def test_contact_geometry_gradient(self):
        positions = [
            np.array([[0.2, 0.2, -0.03]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])]
        contact = ActiveContact(
            "p-t", 0, 0, 1, (0, 1, 2),
            np.array([0.6, 0.2, 0.2]),
            np.array([0.0, 0.0, 1.0]), multiplier=0.4)
        value, gradients, _, _ = contact.energy_gradient(positions, 15.0)
        directions = [
            np.random.default_rng(2).normal(size=x.shape)
            for x in positions]
        norm = np.sqrt(sum(np.sum(x * x) for x in directions))
        directions = [x / norm for x in directions]
        epsilon = 1e-7
        plus = contact.energy_gradient([
            x + epsilon * d for x, d in zip(positions, directions)],
            15.0)[0]
        minus = contact.energy_gradient([
            x - epsilon * d for x, d in zip(positions, directions)],
            15.0)[0]
        analytical = sum(
            np.sum(g * d) for g, d in zip(gradients, directions))
        self.assertLess(abs(
            analytical - (plus - minus) / (2.0 * epsilon)), 1e-7)
        self.assertGreater(value, 0.0)

    def test_experiment_b_block_plane_scalar_equilibrium(self):
        # One normal block coordinate with spring k and plane g=x.
        rest, stiffness, rho, multiplier = -0.08, 10.0, 5.0, 0.0
        x = rest
        for _ in range(40):
            _, al_gradient, al_hessian = (
                al_inequality_value_derivatives(x, multiplier, rho))
            gradient = stiffness * (x - rest) + al_gradient
            hessian = stiffness + al_hessian
            x -= gradient / hessian
            multiplier = update_multiplier(multiplier, x, rho)
        self.assertGreaterEqual(x, -1e-5)
        self.assertLess(
            abs(fischer_burmeister(multiplier, x)), 1e-4)
        self.assertGreater(multiplier, 0.0)

    def test_experiment_c_two_free_bodies_are_symmetric(self):
        # Two spring-supported normal coordinates, g=b-a.
        rest_a, rest_b = 0.02, -0.02
        a, b, multiplier, rho, stiffness = (
            rest_a, rest_b, 0.0, 10.0, 8.0)
        for _ in range(20):
            gap = b - a
            _, derivative, curvature = (
                al_inequality_value_derivatives(gap, multiplier, rho))
            gradient = np.array([
                stiffness * (a - rest_a) - derivative,
                stiffness * (b - rest_b) + derivative])
            hessian = np.array([
                [stiffness + curvature, -curvature],
                [-curvature, stiffness + curvature]])
            step = np.linalg.solve(hessian, -gradient)
            a, b = np.array([a, b]) + step
            multiplier = update_multiplier(multiplier, b - a, rho)
        self.assertGreaterEqual(b - a, -1e-5)
        self.assertAlmostEqual(
            a - rest_a, -(b - rest_b), places=8)

    def test_experiment_d_nonconflicting_hard_attachment(self):
        # a is fixed away from contact, b is free and feasible.
        a, b = -0.2, 0.1
        multiplier = update_multiplier(0.0, b - a, 10.0)
        self.assertEqual(multiplier, 0.0)
        self.assertGreater(b - a, 0.0)

    def test_experiment_e_infeasible_fixed_overlap(self):
        positions = [
            np.array([[0.0, 0.0, -0.02]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])]
        contact = ActiveContact(
            "fixed-overlap", 0, 0, 1, (0, 1, 2),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]), last_gap=-0.02)
        conflicts = diagnose_infeasible_contacts(
            [contact], [{0}, {0, 1, 2}], 1e-4)
        self.assertEqual(len(conflicts), 1)
        multiplier = 0.0
        for _ in range(5):
            multiplier = update_multiplier(
                multiplier, contact.last_gap, 10.0)
        self.assertGreater(multiplier, 0.0)
        self.assertNotEqual(
            fischer_burmeister(multiplier, contact.last_gap), 0.0)

    def test_active_set_hysteresis(self):
        active = PersistentActiveSet(0.001, 0.002)
        contact = ActiveContact(
            "c", 0, 0, 1, (0, 1, 2), np.ones(3) / 3.0,
            np.array([0.0, 0.0, 1.0]), last_gap=0.0005)
        self.assertEqual(active.update([contact]), 1)
        contact.last_gap = 0.0015
        self.assertEqual(active.update([contact]), 0)
        contact.last_gap = 0.0025
        self.assertEqual(active.update([contact]), 1)


if __name__ == "__main__":
    unittest.main()
