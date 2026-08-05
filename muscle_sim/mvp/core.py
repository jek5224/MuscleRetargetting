"""Small robust primitives used by the muscle animation MVP.

This module deliberately favors bounded, visually stable projections over the
strict publication/FEM acceptance machinery.
"""
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree


def boundary_faces(tets):
    count, oriented = Counter(), {}
    for a, b, c, d in np.asarray(tets, dtype=np.int64):
        for face in ((a, c, b), (a, b, d), (b, c, d), (a, d, c)):
            key = tuple(sorted(map(int, face)))
            count[key] += 1
            oriented.setdefault(key, tuple(map(int, face)))
    return np.asarray([oriented[k] for k, n in count.items() if n == 1],
                      dtype=np.int32)


def signed_tet_volumes(vertices, tets):
    q = np.asarray(vertices)[np.asarray(tets)]
    return np.einsum("ij,ij->i", np.cross(q[:, 1] - q[:, 0],
                     q[:, 2] - q[:, 0]), q[:, 3] - q[:, 0]) / 6.


def orient_tets(vertices, tets):
    result = np.asarray(tets, dtype=np.int32).copy()
    negative = signed_tet_volumes(vertices, result) < 0.
    result[negative, 1], result[negative, 2] = (
        result[negative, 2].copy(), result[negative, 1].copy())
    return result


def pathological_tets(vertices, tets, relative_threshold=1e-5):
    q = np.asarray(vertices)[np.asarray(tets)]
    edges = [np.linalg.norm(q[:, i] - q[:, j], axis=1)
             for i in range(4) for j in range(i + 1, 4)]
    scale = np.mean(edges, axis=0) ** 3 + 1e-30
    quality = np.abs(signed_tet_volumes(vertices, tets)) / scale
    return np.flatnonzero(quality < relative_threshold), quality


def attachment_update(local_points, rotation, translation):
    return (np.asarray(rotation) @ np.asarray(local_points).T).T + translation


def closest_rotation(deformation_gradient):
    u, _, vt = np.linalg.svd(np.asarray(deformation_gradient))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.:
        u[:, -1] *= -1.
        rotation = u @ vt
    return rotation


def project_global_volume(vertices, rest_vertices, fixed=(), strength=.8):
    """Radially correct volume while leaving hard attachments untouched."""
    vertices = np.asarray(vertices).copy()
    fixed = np.asarray(sorted(set(map(int, fixed))), dtype=np.int64)
    center = vertices.mean(axis=0)
    rest_center = np.asarray(rest_vertices).mean(axis=0)
    rest_radius = np.sqrt(np.mean(np.sum(
        (np.asarray(rest_vertices) - rest_center) ** 2, axis=1)))
    radius = np.sqrt(np.mean(np.sum((vertices - center) ** 2, axis=1)))
    scale = np.clip(rest_radius / max(radius, 1e-12), .75, 1.35)
    corrected = center + (vertices - center) * (
        1. + strength * (scale - 1.))
    if len(fixed):
        corrected[fixed] = vertices[fixed]
    return corrected


def pca_centerline(vertices, count=17):
    points = np.asarray(vertices)
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    axis = vt[0]
    coordinate = (points - center) @ axis
    lo, hi = np.quantile(coordinate, [.02, .98])
    return center + np.linspace(lo, hi, count)[:, None] * axis


def centerline_from_endpoints(vertices, start_ids, end_ids, count=17):
    points = np.asarray(vertices)
    start = points[np.asarray(start_ids, dtype=int)].mean(axis=0)
    end = points[np.asarray(end_ids, dtype=int)].mean(axis=0)
    u = np.linspace(0., 1., count)
    return (1. - u[:, None]) * start + u[:, None] * end


def wrap_polyline_capsule(points, center, radius, side_direction):
    """Project interior guide points outside a spherical knee proxy."""
    result = np.asarray(points).copy()
    side = np.asarray(side_direction, dtype=float)
    side /= max(np.linalg.norm(side), 1e-12)
    for i in range(1, len(result) - 1):
        delta = result[i] - center
        distance = np.linalg.norm(delta)
        if distance < radius:
            normal = delta / max(distance, 1e-12)
            normal = normal + .35 * side
            normal /= max(np.linalg.norm(normal), 1e-12)
            result[i] = center + radius * normal
    for _ in range(2):
        result[1:-1] = (
            .25 * result[:-2] + .5 * result[1:-1] + .25 * result[2:])
    return result


def guide_volume(vertices, rest_vertices, rest_guide, posed_guide,
                 weight=.25, fixed=()):
    """Couple vertices to corresponding longitudinal guide locations."""
    vertices = np.asarray(vertices).copy()
    rest = np.asarray(rest_vertices)
    guide = np.asarray(rest_guide)
    segments = guide[1:] - guide[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    total = max(arc[-1], 1e-12)
    axis = guide[-1] - guide[0]
    denom = max(np.dot(axis, axis), 1e-12)
    u = np.clip(((rest - guide[0]) @ axis) / denom, 0., 1.)
    sample_u = arc / total
    target = np.column_stack([
        np.interp(u, sample_u, posed_guide[:, k]) for k in range(3)])
    rest_axis_points = guide[0] + u[:, None] * axis
    desired = target + (rest - rest_axis_points)
    result = (1. - weight) * vertices + weight * desired
    fixed = np.asarray(sorted(set(map(int, fixed))), dtype=np.int64)
    if len(fixed):
        result[fixed] = vertices[fixed]
    return result


def _rotation_between(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = np.clip(np.dot(a, b), -1., 1.)
    if sine < 1e-10:
        if cosine > 0.:
            return np.eye(3)
        trial = np.array([1., 0., 0.])
        if abs(np.dot(trial, a)) > .8:
            trial = np.array([0., 1., 0.])
        axis = np.cross(a, trial)
        axis /= np.linalg.norm(axis)
        return 2. * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.array([[0., -axis[2], axis[1]],
                     [axis[2], 0., -axis[0]],
                     [-axis[1], axis[0], 0.]])
    return np.eye(3) + sine * skew + (1. - cosine) * (skew @ skew)


def sweep_deform(rest_vertices, rest_guide, posed_guide, attachment_target=None,
                 fixed=()):
    """Sweep rest cross-sections along a posed guide.

    This is more robust than vertex-wise blended skinning for long muscles:
    longitudinal ordering is preserved and radial cross-sections rotate as a
    coherent unit. LBS is blended only near attachment ends.
    """
    rest = np.asarray(rest_vertices)
    rest_guide = np.asarray(rest_guide)
    posed_guide = np.asarray(posed_guide)
    rest_axis = rest_guide[-1] - rest_guide[0]
    denom = max(np.dot(rest_axis, rest_axis), 1e-12)
    u = np.clip(((rest - rest_guide[0]) @ rest_axis) / denom, 0., 1.)
    parameter = np.linspace(0., 1., len(posed_guide))
    centers = np.column_stack([
        np.interp(u, parameter, posed_guide[:, k]) for k in range(3)])
    segment = np.clip(
        np.searchsorted(parameter, u, side="right") - 1,
        0, len(posed_guide) - 2)
    tangents = posed_guide[segment + 1] - posed_guide[segment]
    rest_centers = rest_guide[0] + u[:, None] * rest_axis
    radial = rest - rest_centers
    result = np.empty_like(rest)
    for i in range(len(rest)):
        result[i] = centers[i] + _rotation_between(
            rest_axis, tangents[i]) @ radial[i]
    if attachment_target is not None:
        target = np.asarray(attachment_target)
        # Broad transition avoids a discontinuity between rigid cap motion and
        # the centerline sweep, especially for long bi-articular muscles.
        end_weight = np.clip((.42 - np.minimum(u, 1. - u)) / .42, 0., 1.)
        end_weight = end_weight * end_weight * (3. - 2. * end_weight)
        result = ((1. - end_weight[:, None]) * result
                  + end_weight[:, None] * target)
    fixed = np.asarray(sorted(set(map(int, fixed))), dtype=np.int64)
    if len(fixed) and attachment_target is not None:
        result[fixed] = np.asarray(attachment_target)[fixed]
    return result


def project_bone_collision(vertices, surface_ids, bone_vertices, margin=.0015,
                           fixed=()):
    """Unsigned closest-vertex fallback suitable for imperfect bone surfaces."""
    result = np.asarray(vertices).copy()
    if not len(bone_vertices) or not len(surface_ids):
        return result, 0, 0.
    tree = cKDTree(np.asarray(bone_vertices))
    ids = np.asarray(surface_ids, dtype=np.int64)
    distance, nearest = tree.query(result[ids])
    active = distance < margin
    fixed = set(map(int, fixed))
    moved, maximum = 0, 0.
    bone_center = np.asarray(bone_vertices).mean(axis=0)
    for local in np.flatnonzero(active):
        vi = int(ids[local])
        if vi in fixed:
            continue
        base = np.asarray(bone_vertices)[nearest[local]]
        normal = base - bone_center
        normal /= max(np.linalg.norm(normal), 1e-12)
        result[vi] = base + margin * normal
        moved += 1
        maximum = max(maximum, float(margin - distance[local]))
    return result, moved, maximum


def project_pair_collision(a, a_ids, b, b_ids, spacing=.001):
    """Symmetric normal-only proximity projection with no tangential glue."""
    a, b = np.asarray(a).copy(), np.asarray(b).copy()
    if not len(a_ids) or not len(b_ids):
        return a, b, 0, 0.
    ia, ib = np.asarray(a_ids), np.asarray(b_ids)
    tree = cKDTree(b[ib])
    distance, nearest = tree.query(a[ia])
    active = np.flatnonzero(distance < spacing)
    maximum = 0.
    for local in active:
        va, vb = int(ia[local]), int(ib[nearest[local]])
        direction = a[va] - b[vb]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = a.mean(axis=0) - b.mean(axis=0)
            norm = np.linalg.norm(direction)
        direction /= max(norm, 1e-12)
        correction = .5 * (spacing - distance[local]) * direction
        a[va] += correction
        b[vb] -= correction
        maximum = max(maximum, float(spacing - distance[local]))
    return a, b, len(active), maximum


def pose_substeps(angle_degrees, moderate=45., deep=80.,
                  base=2, deep_count=6):
    if angle_degrees >= deep:
        return deep_count
    if angle_degrees >= moderate:
        return max(base + 1, deep_count // 2)
    return base
