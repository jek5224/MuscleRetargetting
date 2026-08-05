"""Coupled multi-muscle state and differentiable frictionless contact kernels.

The isolated constitutive implementation remains in ``viewer.isolated_muscle``.
This module only supplies global indexing and contact/confinement terms.
Closest features are rebuilt on every evaluation; within one feature region the
point-triangle energy uses the envelope-theorem gradient.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree, ConvexHull

from viewer.isolated_muscle import (
    MuscleData, deformation_gradients, total_energy_gradient)


@dataclass
class MuscleBody:
    muscle: MuscleData
    precomputed: dict
    config: dict
    offset: int = 0
    fixed_vertices: set = field(default_factory=set)
    attachment_exclusion: set = field(default_factory=set)

    @property
    def name(self):
        return self.muscle.name

    @property
    def vertex_count(self):
        return len(self.muscle.vertices)


@dataclass
class RigidSurface:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    parent_bone: Optional[str] = None


@dataclass
class ContactDiagnostics:
    energy: float = 0.0
    active_count: int = 0
    minimum_gap: float = np.inf
    maximum_penetration: float = 0.0
    points: list = field(default_factory=list)
    closest_points: list = field(default_factory=list)
    normals: list = field(default_factory=list)

    def merge(self, other):
        self.energy += other.energy
        self.active_count += other.active_count
        self.minimum_gap = min(self.minimum_gap, other.minimum_gap)
        self.maximum_penetration = max(
            self.maximum_penetration, other.maximum_penetration)
        self.points.extend(other.points)
        self.closest_points.extend(other.closest_points)
        self.normals.extend(other.normals)


@dataclass
class MultiMuscleSystem:
    bodies: List[MuscleBody]
    contact_pairs: List[Tuple[int, int]]
    bones: Dict[str, RigidSurface] = field(default_factory=dict)
    fascia: Optional[RigidSurface] = None
    wrapping_objects: Dict[str, RigidSurface] = field(default_factory=dict)
    body_bones: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        offset = 0
        for body in self.bodies:
            body.offset = offset
            offset += body.vertex_count
            body.fixed_vertices = {
                int(vertex)
                for patch in body.muscle.attachment_patches
                for vertex in patch.vertex_ids}
        self.vertex_count = offset

    def pack(self, positions):
        return np.vstack(positions)

    def unpack(self, global_positions):
        return [
            global_positions[b.offset:b.offset + b.vertex_count]
            for b in self.bodies]


def closest_point_triangle(point, triangle):
    """Return closest point and barycentrics (Ericson region tests)."""
    a, b, c = np.asarray(triangle, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array([1.0, 0.0, 0.0])
    bp = p - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1.0 - v, v, 0.0])
    cp = p - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w, np.array([1.0 - v - w, v, w])


def _closest_triangle(point, vertices, faces, tree=None, candidates=12):
    triangles = vertices[faces]
    if tree is None:
        tree = cKDTree(np.mean(triangles, axis=1))
    count = min(candidates, len(faces))
    indices = np.atleast_1d(tree.query(point, k=count)[1])
    best = None
    for face_id in indices:
        closest, bary = closest_point_triangle(
            point, triangles[int(face_id)])
        distance2 = float(np.dot(point - closest, point - closest))
        if best is None or distance2 < best[0]:
            best = distance2, int(face_id), closest, bary
    return best


def point_rigid_contact_energy_gradient(
        points, rigid_vertices, rigid_faces, thickness, stiffness,
        forbidden_mask=None, point_ids=None):
    """Smooth quadratic normal contact against a rigid triangle surface.

    ``forbidden_mask`` supplies the signed side for a closed surface. A point
    on the forbidden side has negative gap. For bones this means "inside";
    for a containing fascia shell this means "outside".
    """
    points = np.asarray(points, dtype=np.float64)
    gradient = np.zeros_like(points)
    diagnostics = ContactDiagnostics()
    if forbidden_mask is None:
        forbidden_mask = np.zeros(len(points), dtype=bool)
    if point_ids is None:
        point_ids = np.arange(len(points))
    point_ids = np.asarray(point_ids, dtype=np.int32)
    if not len(point_ids):
        return 0.0, gradient, diagnostics
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=rigid_vertices, faces=rigid_faces, process=False)
    closest, distances, _ = trimesh.proximity.closest_point(
        mesh, points[point_ids])
    distances = np.maximum(distances, 1e-12)
    signs = np.where(forbidden_mask[point_ids], -1.0, 1.0)
    gaps = signs * distances
    penetrations = np.maximum(0.0, thickness - gaps)
    diagnostics.minimum_gap = float(np.min(gaps))
    diagnostics.maximum_penetration = float(np.max(penetrations))
    active = penetrations > 0.0
    if np.any(active):
        active_ids = point_ids[active]
        normals = (
            signs[active, None]
            * (points[active_ids] - closest[active])
            / distances[active, None])
        diagnostics.energy = float(
            0.5 * stiffness * np.sum(penetrations[active] ** 2))
        gradient[active_ids] -= (
            stiffness * penetrations[active, None] * normals)
        diagnostics.active_count = int(np.sum(active))
        diagnostics.points.extend(points[active_ids].copy())
        diagnostics.closest_points.extend(closest[active].copy())
        diagnostics.normals.extend(normals.copy())
    return diagnostics.energy, gradient, diagnostics


def directed_deformable_contact_energy_gradient(
        source_vertices, source_ids, target_vertices, target_faces,
        thickness, stiffness, source_inside_target=None):
    """One directed half of symmetric vertex-triangle sliding contact."""
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    target_vertices = np.asarray(target_vertices, dtype=np.float64)
    source_gradient = np.zeros_like(source_vertices)
    target_gradient = np.zeros_like(target_vertices)
    diagnostics = ContactDiagnostics()
    if source_inside_target is None:
        source_inside_target = np.zeros(len(source_vertices), dtype=bool)
    source_ids = np.asarray(source_ids, dtype=np.int32)
    if not len(source_ids):
        return 0.0, source_gradient, target_gradient, diagnostics
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=target_vertices, faces=target_faces, process=False)
    closest, distances, face_ids = trimesh.proximity.closest_point(
        mesh, source_vertices[source_ids])
    distances = np.maximum(distances, 1e-12)
    signs = np.where(source_inside_target[source_ids], -1.0, 1.0)
    gaps = signs * distances
    penetrations = np.maximum(0.0, thickness - gaps)
    diagnostics.minimum_gap = float(np.min(gaps))
    diagnostics.maximum_penetration = float(np.max(penetrations))
    active = penetrations > 0.0
    if np.any(active):
        active_source = source_ids[active]
        active_faces = target_faces[face_ids[active]]
        triangles = target_vertices[active_faces]
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        v0, v1, v2 = b - a, c - a, closest[active] - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denominator = np.maximum(d00 * d11 - d01 * d01, 1e-20)
        bary_v = (d11 * d20 - d01 * d21) / denominator
        bary_w = (d00 * d21 - d01 * d20) / denominator
        bary = np.column_stack((1.0 - bary_v - bary_w,
                                bary_v, bary_w))
        normals = (
            signs[active, None]
            * (source_vertices[active_source] - closest[active])
            / distances[active, None])
        force_scale = 0.5 * stiffness * penetrations[active]
        diagnostics.energy = float(
            0.25 * stiffness * np.sum(penetrations[active] ** 2))
        source_gradient[active_source] -= force_scale[:, None] * normals
        contributions = (
            force_scale[:, None, None] * bary[:, :, None]
            * normals[:, None, :])
        np.add.at(target_gradient, active_faces, contributions)
        diagnostics.active_count = int(np.sum(active))
        diagnostics.points.extend(source_vertices[active_source].copy())
        diagnostics.closest_points.extend(closest[active].copy())
        diagnostics.normals.extend(normals.copy())
    return diagnostics.energy, source_gradient, target_gradient, diagnostics


def symmetric_deformable_contact_energy_gradient(
        first_vertices, first_surface_ids, first_faces,
        second_vertices, second_surface_ids, second_faces,
        thickness, stiffness, first_inside_second=None,
        second_inside_first=None):
    e1, g1a, g2a, d1 = directed_deformable_contact_energy_gradient(
        first_vertices, first_surface_ids, second_vertices, second_faces,
        thickness, stiffness, first_inside_second)
    e2, g2b, g1b, d2 = directed_deformable_contact_energy_gradient(
        second_vertices, second_surface_ids, first_vertices, first_faces,
        thickness, stiffness, second_inside_first)
    d1.merge(d2)
    return e1 + e2, g1a + g1b, g2a + g2b, d1


def normal_cohesion_energy_gradient(
        first, second, first_ids, second_ids, reference_normals,
        allowed_gap, break_distance, stiffness):
    """Normal-only separation energy for fixed rest-neighbor pairs."""
    gradient_first = np.zeros_like(first)
    gradient_second = np.zeros_like(second)
    energy = 0.0
    active = 0
    for ia, ib, normal in zip(first_ids, second_ids, reference_normals):
        delta = second[ib] - first[ia]
        distance = np.linalg.norm(delta)
        separation = float(np.dot(delta, normal))
        excess = separation - allowed_gap
        if excess <= 0.0 or distance >= break_distance:
            continue
        energy += 0.5 * stiffness * excess * excess
        force = stiffness * excess * normal
        gradient_first[ia] -= force
        gradient_second[ib] += force
        active += 1
    return energy, gradient_first, gradient_second, active


def multi_energy_gradient(global_vertices, system, contact_config,
                          rigid_forbidden_masks=None):
    """Assemble all independent constitutive terms and coupled contacts."""
    positions = system.unpack(global_vertices)
    gradient = np.zeros_like(global_vertices)
    terms = {"elastic": 0.0, "bone_contact": 0.0,
             "muscle_contact": 0.0, "fascia_contact": 0.0}
    diagnostics = {}
    for body, current in zip(system.bodies, positions):
        energy, local_gradient, local_terms = total_energy_gradient(
            current, body.muscle, body.precomputed, body.config)
        if not np.isfinite(energy):
            return np.inf, np.full_like(global_vertices, np.nan), terms, {}
        terms["elastic"] += energy
        gradient[body.offset:body.offset + body.vertex_count] += local_gradient
        diagnostics[body.name] = {"material": local_terms}

    if contact_config.get("bone_muscle_enabled", True):
        import trimesh
        for body_id, (body, current) in enumerate(
                zip(system.bodies, positions)):
            surface_ids = np.unique(body.muscle.surface_faces)
            contact_ids = np.asarray([
                vertex for vertex in surface_ids
                if vertex not in body.attachment_exclusion], dtype=np.int32)
            for bone_name in system.body_bones.get(body.name, []):
                bone = system.bones[bone_name]
                key = f"{body.name}|{bone_name}"
                if rigid_forbidden_masks and key in rigid_forbidden_masks:
                    forbidden = rigid_forbidden_masks[key]
                else:
                    mesh = trimesh.Trimesh(
                        vertices=bone.vertices, faces=bone.faces,
                        process=False)
                    forbidden = mesh.contains(current)
                energy, local_gradient, diag = (
                    point_rigid_contact_energy_gradient(
                        current, bone.vertices, bone.faces,
                        float(contact_config["bone_muscle_thickness"]),
                        float(contact_config["stiffness"]),
                        forbidden_mask=forbidden, point_ids=contact_ids))
                terms["bone_contact"] += energy
                gradient[body.offset:body.offset + body.vertex_count] += (
                    local_gradient)
                diagnostics[key] = diag

    if contact_config.get("muscle_muscle_enabled", True):
        import trimesh
        for first_id, second_id in system.contact_pairs:
            first, second = system.bodies[first_id], system.bodies[second_id]
            x1, x2 = positions[first_id], positions[second_id]
            ids1 = np.unique(first.muscle.surface_faces)
            ids2 = np.unique(second.muscle.surface_faces)
            mesh1 = trimesh.Trimesh(
                vertices=x1, faces=first.muscle.surface_faces, process=False)
            mesh2 = trimesh.Trimesh(
                vertices=x2, faces=second.muscle.surface_faces, process=False)
            first_inside_second = mesh2.contains(x1)
            second_inside_first = mesh1.contains(x2)
            energy, g1, g2, diag = (
                symmetric_deformable_contact_energy_gradient(
                    x1, ids1, first.muscle.surface_faces,
                    x2, ids2, second.muscle.surface_faces,
                    float(contact_config["muscle_muscle_thickness"]),
                    float(contact_config["stiffness"]),
                    first_inside_second, second_inside_first))
            terms["muscle_contact"] += energy
            gradient[first.offset:first.offset + first.vertex_count] += g1
            gradient[second.offset:second.offset + second.vertex_count] += g2
            diagnostics[f"{first.name}|{second.name}"] = diag

    if system.fascia is not None and contact_config.get(
            "fascia_muscle_enabled", True):
        import trimesh
        shell = trimesh.Trimesh(
            vertices=system.fascia.vertices, faces=system.fascia.faces,
            process=False)
        for body, current in zip(system.bodies, positions):
            surface_ids = np.unique(body.muscle.surface_faces)
            # Fascia contains the muscles: outside, rather than inside, is the
            # forbidden side.
            forbidden = ~shell.contains(current)
            energy, local_gradient, diag = (
                point_rigid_contact_energy_gradient(
                    current, system.fascia.vertices, system.fascia.faces,
                    float(contact_config["fascia_muscle_thickness"]),
                    float(contact_config["stiffness"]),
                    forbidden_mask=forbidden, point_ids=surface_ids))
            terms["fascia_contact"] += energy
            gradient[body.offset:body.offset + body.vertex_count] += (
                local_gradient)
            diagnostics[f"{body.name}|fascia"] = diag
    return sum(terms.values()), gradient, terms, diagnostics


def compactness_metrics(positions, bodies, neighbor_distance,
                        fascia_volume=None):
    """Reproducible geometric compactness diagnostics (not anatomy scores)."""
    surfaces = [
        x[np.unique(body.muscle.surface_faces)]
        for x, body in zip(positions, bodies)]
    nearest = []
    near_count = total_count = 0
    for index, points in enumerate(surfaces):
        others = np.vstack([
            other for j, other in enumerate(surfaces) if j != index])
        distances = cKDTree(others).query(points)[0]
        nearest.extend(distances.tolist())
        near_count += int(np.sum(distances < neighbor_distance))
        total_count += len(distances)
    all_points = np.vstack(surfaces)
    bbox_volume = float(np.prod(np.ptp(all_points, axis=0)))
    try:
        hull_volume = float(ConvexHull(all_points).volume)
    except Exception:
        hull_volume = np.nan
    total_volume = 0.0
    for current, body in zip(positions, bodies):
        F = deformation_gradients(
            current, body.muscle, body.precomputed)
        total_volume += float(np.sum(
            body.precomputed["volumes"] * np.linalg.det(F)))
    return {
        "neighbor_surface_fraction": near_count / max(total_count, 1),
        "mean_internal_gap": float(np.mean(nearest)),
        "union_bounding_box_volume": bbox_volume,
        "convex_hull_volume": hull_volume,
        "total_muscle_volume": total_volume,
        "fascia_fill_ratio": (
            total_volume / fascia_volume
            if fascia_volume is not None and fascia_volume > 0 else np.nan),
    }
