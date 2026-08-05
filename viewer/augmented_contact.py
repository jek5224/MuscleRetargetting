"""Augmented-Lagrangian normal contact and persistent active-set utilities.

Sign convention:
    gap g >= 0 is feasible
    multiplier lambda >= 0
    lambda * g = 0

For c=-g <= 0 the Powell-Hestenes-Rockafellar inequality term is

    (max(0, lambda-rho*g)^2-lambda^2)/(2*rho).

Its active derivative with respect to g is rho*g-lambda and the update is
lambda <- max(0, lambda-rho*g).
"""
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np


def al_inequality_value_derivatives(gap, multiplier, rho):
    """Return AL value, d/dgap, and generalized d2/dgap2."""
    shifted = float(multiplier) - float(rho) * float(gap)
    if shifted <= 0.0:
        return (
            -0.5 * float(multiplier) ** 2 / float(rho),
            0.0, 0.0)
    value = (
        0.5 * shifted * shifted / float(rho)
        - 0.5 * float(multiplier) ** 2 / float(rho))
    return value, -shifted, float(rho)


def update_multiplier(multiplier, gap, rho):
    return max(0.0, float(multiplier) - float(rho) * float(gap))


def fischer_burmeister(multiplier, gap, epsilon=0.0):
    """FB residual for lambda>=0, gap>=0, lambda*gap=0."""
    return (
        np.sqrt(float(multiplier) ** 2 + float(gap) ** 2
                + float(epsilon) ** 2)
        - float(multiplier) - float(gap))


@dataclass
class ActiveContact:
    constraint_id: Hashable
    source_body: int
    source_vertex: int
    target_body: int
    target_triangle: tuple
    barycentric: np.ndarray
    normal: np.ndarray
    multiplier: float = 0.0
    last_gap: float = np.inf
    weight: float = 1.0

    def gap(self, positions):
        source = positions[self.source_body][self.source_vertex]
        target = np.sum(
            positions[self.target_body][
                np.asarray(self.target_triangle, dtype=np.int32)]
            * self.barycentric[:, None], axis=0)
        return float(np.dot(source - target, self.normal))

    def energy_gradient(self, positions, rho):
        gap = self.gap(positions)
        value, derivative, curvature = al_inequality_value_derivatives(
            gap, self.multiplier, rho)
        value *= self.weight
        derivative *= self.weight
        gradients = [
            np.zeros_like(current, dtype=np.float64)
            for current in positions]
        gradients[self.source_body][self.source_vertex] += (
            derivative * self.normal)
        target = np.asarray(self.target_triangle, dtype=np.int32)
        gradients[self.target_body][target] -= (
            derivative * self.barycentric[:, None] * self.normal)
        return value, gradients, gap, curvature * self.weight


class PersistentActiveSet:
    def __init__(self, activation_distance, release_distance):
        if release_distance <= activation_distance:
            raise ValueError(
                "release_distance must exceed activation_distance")
        self.activation_distance = float(activation_distance)
        self.release_distance = float(release_distance)
        self.contacts = {}

    def update(self, candidates: Iterable[ActiveContact]):
        previous_ids = set(self.contacts)
        candidate_map = {
            candidate.constraint_id: candidate for candidate in candidates}
        result = {}
        for constraint_id, candidate in candidate_map.items():
            old = self.contacts.get(constraint_id)
            if old is not None:
                candidate.multiplier = old.multiplier
            if (candidate.last_gap <= self.activation_distance
                    or (old is not None
                        and candidate.last_gap <= self.release_distance)):
                result[constraint_id] = candidate
        self.contacts = result
        return len(previous_ids.symmetric_difference(result))

    def multiplier_update(self, rho):
        maximum_change = 0.0
        for contact in self.contacts.values():
            old = contact.multiplier
            contact.multiplier = update_multiplier(
                old, contact.last_gap, rho)
            maximum_change = max(
                maximum_change, abs(contact.multiplier - old))
        return maximum_change

    def complementarity_norm(self):
        if not self.contacts:
            return 0.0
        residuals = [
            fischer_burmeister(contact.multiplier, contact.last_gap)
            for contact in self.contacts.values()]
        return float(np.linalg.norm(residuals))

    def maximum_penetration(self):
        return max(
            [max(0.0, -contact.last_gap)
             for contact in self.contacts.values()] or [0.0])


def diagnose_infeasible_contacts(contacts, fixed_by_body,
                                 penetration_tolerance):
    conflicts = []
    for contact in contacts:
        source_fixed = (
            contact.source_vertex in fixed_by_body[contact.source_body])
        target_fixed = all(
            vertex in fixed_by_body[contact.target_body]
            for vertex in contact.target_triangle)
        if (source_fixed and target_fixed
                and contact.last_gap < -penetration_tolerance):
            conflicts.append({
                "constraint_id": str(contact.constraint_id),
                "gap": contact.last_gap,
                "reason": "both contact primitives are hard constrained",
            })
    return conflicts


def _triangle_barycentric(points, triangles):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1, v2 = b - a, c - a, points - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = np.maximum(d00 * d11 - d01 * d01, 1e-20)
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.column_stack((1.0 - v - w, v, w))


def build_deformable_candidates(
        positions, bodies, first_id, second_id, release_distance,
        previous=None, symmetric_weight=0.5):
    """Build exact signed vertex-triangle candidates in both directions."""
    import trimesh
    contacts = []
    previous = previous or {}
    for source_id, target_id in (
            (first_id, second_id), (second_id, first_id)):
        source_body, target_body = bodies[source_id], bodies[target_id]
        source, target = positions[source_id], positions[target_id]
        source_ids = np.unique(source_body.muscle.surface_faces)
        target_faces = target_body.muscle.surface_faces
        mesh = trimesh.Trimesh(
            vertices=target, faces=target_faces, process=False)
        closest, distances, face_ids = trimesh.proximity.closest_point(
            mesh, source[source_ids])
        inside = mesh.contains(source[source_ids])
        gaps = np.where(inside, -distances, distances)
        keep = gaps <= release_distance
        kept_source = source_ids[keep]
        kept_faces = face_ids[keep]
        kept_closest = closest[keep]
        triangles = target[target_faces[kept_faces]]
        barycentric = _triangle_barycentric(kept_closest, triangles)
        delta = source[kept_source] - kept_closest
        lengths = np.maximum(
            np.linalg.norm(delta, axis=1), 1e-12)
        # The normal points toward increasing signed gap. For a buried point,
        # closest-source is outward; outside it is source-closest.
        normals = (
            np.where(inside[keep, None], -1.0, 1.0)
            * delta / lengths[:, None])
        for vertex, face_id, bary, normal, gap in zip(
                kept_source, kept_faces, barycentric, normals, gaps[keep]):
            constraint_id = (
                int(source_id), int(vertex),
                int(target_id), int(face_id))
            old = previous.get(constraint_id)
            contacts.append(ActiveContact(
                constraint_id, int(source_id), int(vertex),
                int(target_id),
                tuple(int(value) for value in target_faces[int(face_id)]),
                bary, normal,
                multiplier=(old.multiplier if old else 0.0),
                last_gap=float(gap), weight=float(symmetric_weight)))
    return contacts
