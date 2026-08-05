"""Validated building blocks for an isolated quasistatic tet-muscle solve.

This module deliberately does not depend on the multi-muscle contact/cage
pipeline.  It consumes the project's existing pickled ``*_tet.npz`` files and
keeps explicit curved waypoint fibers embedded in their tetrahedral volume.
All positions are SI metres.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree


@dataclass
class AttachmentPatch:
    bone_name: str
    vertex_ids: np.ndarray
    bone_local_positions: Optional[np.ndarray] = None
    stream_index: int = 0
    end_type: int = 0


@dataclass
class EmbeddedFiber:
    stream_index: int
    fiber_index: int
    tet_ids: np.ndarray
    barycentric: np.ndarray
    rest_points: np.ndarray
    rest_segment_lengths: np.ndarray

    @property
    def rest_total_length(self):
        return float(np.sum(self.rest_segment_lengths))


@dataclass
class MuscleData:
    name: str
    vertices: np.ndarray
    tetrahedra: np.ndarray
    surface_faces: np.ndarray
    attachment_patches: list
    fibers: list
    raw: dict


def bind_attachment_patches(patches, rest_vertices, bone_rest_transforms):
    """Populate bone-local attachment positions from 4x4 rest transforms."""
    for patch in patches:
        transform = np.asarray(
            bone_rest_transforms[patch.bone_name], dtype=np.float64)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        patch.bone_local_positions = (
            np.asarray(rest_vertices)[patch.vertex_ids] - translation
        ) @ rotation


def attachment_targets(patches, bone_world_transforms):
    targets = {}
    for patch in patches:
        if patch.bone_local_positions is None:
            raise ValueError(f"patch for {patch.bone_name} is not bound")
        transform = np.asarray(
            bone_world_transforms[patch.bone_name], dtype=np.float64)
        world = (
            patch.bone_local_positions @ transform[:3, :3].T
            + transform[:3, 3])
        for vertex, point in zip(patch.vertex_ids, world):
            if int(vertex) in targets and not np.allclose(
                    targets[int(vertex)], point, atol=1e-10):
                raise ValueError(
                    f"conflicting attachment target at vertex {vertex}")
            targets[int(vertex)] = point
    return targets


def extract_boundary_faces(tetrahedra):
    """Return consistently indexed faces occurring on exactly one tet."""
    t = np.asarray(tetrahedra, dtype=np.int32)
    faces = np.concatenate((
        t[:, [0, 2, 1]], t[:, [0, 1, 3]],
        t[:, [0, 3, 2]], t[:, [1, 2, 3]]), axis=0)
    key = np.sort(faces, axis=1)
    _, inverse, count = np.unique(
        key, axis=0, return_inverse=True, return_counts=True)
    return faces[count[inverse] == 1]


def orient_tetrahedra(vertices, tetrahedra):
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32).copy()
    p = np.asarray(vertices)[tetrahedra]
    det = np.einsum(
        "ij,ij->i", p[:, 0] - p[:, 3],
        np.cross(p[:, 1] - p[:, 3], p[:, 2] - p[:, 3]))
    negative = det < 0.0
    tmp = tetrahedra[negative, 1].copy()
    tetrahedra[negative, 1] = tetrahedra[negative, 2]
    tetrahedra[negative, 2] = tmp
    return tetrahedra


def filter_tetrahedra(vertices, tetrahedra, minimum_quality=1e-4):
    q = np.asarray(vertices)[tetrahedra]
    determinant = np.einsum(
        "ij,ij->i", q[:, 0] - q[:, 3],
        np.cross(q[:, 1] - q[:, 3], q[:, 2] - q[:, 3]))
    edge_lengths = []
    for first in range(4):
        for second in range(first + 1, 4):
            edge_lengths.append(np.linalg.norm(
                q[:, first] - q[:, second], axis=1))
    mean_edge = np.mean(edge_lengths, axis=0)
    quality = np.abs(determinant) / (
        6.0 * np.maximum(mean_edge, 1e-12) ** 3)
    retained = quality > minimum_quality
    return np.asarray(tetrahedra)[retained], {
        "input_tet_count": int(len(tetrahedra)),
        "retained_tet_count": int(np.sum(retained)),
        "removed_tet_count": int(np.sum(~retained)),
        "minimum_input_tet_quality": float(np.min(quality)),
        "minimum_retained_tet_quality": float(np.min(quality[retained])),
        "quality_threshold": float(minimum_quality),
    }


def _cap_components(sim_faces, cap_face_indices):
    faces = np.asarray(sim_faces, dtype=np.int32)[
        np.asarray(cap_face_indices, dtype=np.int32)]
    vertex_faces = {}
    for fi, face in enumerate(faces):
        for vertex in face:
            vertex_faces.setdefault(int(vertex), []).append(fi)
    unseen = set(range(len(faces)))
    components = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component_faces = {seed}
        while stack:
            fi = stack.pop()
            for vertex in faces[fi]:
                for neighbor in vertex_faces[int(vertex)]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component_faces.add(neighbor)
                        stack.append(neighbor)
        components.append(np.unique(faces[sorted(component_faces)]))
    return components


def attachment_patches_from_tet(data):
    """Expand cap anchors into connected multi-vertex attachment patches."""
    # cap_face_indices are authored against the render boundary, while
    # sim_faces may include a separately regenerated tet boundary.
    sim_faces = data.get("render_faces", data.get("faces"))
    cap_indices = np.asarray(data.get("cap_face_indices", []), dtype=np.int32)
    attachment_rows = np.asarray(
        data.get("cap_attachments", []), dtype=np.int32)
    names = data.get("attach_skeleton_names", [])
    if sim_faces is None or not len(cap_indices) or not len(attachment_rows):
        raise ValueError("tet file has no complete cap attachment metadata")
    components = _cap_components(sim_faces, cap_indices)
    patches = []
    assigned = set()
    for row in attachment_rows:
        anchor, stream, end_type = map(int, row[:3])
        candidates = [c for c in components if anchor in set(c.tolist())]
        if len(candidates) != 1:
            raise ValueError(
                f"attachment anchor {anchor} belongs to {len(candidates)} "
                "cap components")
        vertex_ids = np.asarray(candidates[0], dtype=np.int32)
        key = (stream, end_type, tuple(vertex_ids.tolist()))
        if key in assigned:
            continue
        assigned.add(key)
        if stream >= len(names) or end_type >= len(names[stream]):
            raise ValueError(
                f"missing bone name for stream {stream}, end {end_type}")
        bone_name = str(names[stream][end_type])
        if not bone_name.endswith(("0", "1")):
            bone_name += "0"
        patches.append(AttachmentPatch(
            bone_name=bone_name, vertex_ids=vertex_ids,
            stream_index=stream, end_type=end_type))
    return patches


def explicit_fiber_polylines(data):
    """Convert level-major waypoint storage to explicit curved polylines."""
    result = []
    for stream_index, levels in enumerate(data.get("waypoints", []) or []):
        if not levels:
            continue
        arrays = [np.asarray(level, dtype=np.float64) for level in levels]
        fiber_count = min(len(level) for level in arrays)
        for fiber_index in range(fiber_count):
            result.append((
                stream_index, fiber_index,
                np.asarray([level[fiber_index] for level in arrays])))
    return result


class TetLocator:
    def __init__(self, vertices, tetrahedra):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
        q = self.vertices[self.tetrahedra]
        self.origin = q[:, 3]
        dm = np.stack((
            q[:, 0] - q[:, 3], q[:, 1] - q[:, 3],
            q[:, 2] - q[:, 3]), axis=2)
        self.inverse = np.linalg.inv(dm)
        self.tree = cKDTree(q.mean(axis=1))

    def locate(self, point, tolerance=2e-6, candidates=96):
        count = min(candidates, len(self.tetrahedra))
        _, near = self.tree.query(point, k=count)
        near = np.atleast_1d(near)
        local = np.einsum(
            "nij,nj->ni", self.inverse[near],
            np.asarray(point) - self.origin[near])
        bary = np.column_stack((local, 1.0 - np.sum(local, axis=1)))
        score = np.min(bary, axis=1)
        best = int(np.argmax(score))
        if score[best] < -tolerance:
            return -1, bary[best], float(score[best])
        return int(near[best]), bary[best], float(score[best])


def orient_fiber(points, origin_centroid, insertion_centroid,
                 ambiguity_tolerance=1e-3):
    forward = (
        np.linalg.norm(points[0] - origin_centroid)
        + np.linalg.norm(points[-1] - insertion_centroid))
    reverse = (
        np.linalg.norm(points[-1] - origin_centroid)
        + np.linalg.norm(points[0] - insertion_centroid))
    ambiguous = abs(forward - reverse) < ambiguity_tolerance
    return (points[::-1].copy() if reverse < forward else points.copy(),
            ambiguous)


def embed_fibers(vertices, tetrahedra, polylines, patches,
                 tolerance=2e-6):
    locator = TetLocator(vertices, tetrahedra)
    origins = [p for p in patches if p.end_type == 0]
    insertions = [p for p in patches if p.end_type == 1]
    if not origins or not insertions:
        raise ValueError("both origin and insertion patches are required")
    origin_centroid = np.mean(
        np.concatenate([vertices[p.vertex_ids] for p in origins]), axis=0)
    insertion_centroid = np.mean(
        np.concatenate([vertices[p.vertex_ids] for p in insertions]), axis=0)
    embedded = []
    outside = []
    ambiguous = []
    bary_min = np.inf
    bary_max = -np.inf
    for stream, fiber_index, raw_points in polylines:
        points, is_ambiguous = orient_fiber(
            raw_points, origin_centroid, insertion_centroid)
        ambiguous.extend([(stream, fiber_index)] if is_ambiguous else [])
        tet_ids = []
        weights = []
        sample_valid = []
        for sample_index, point in enumerate(points):
            tet_id, bary, score = locator.locate(point, tolerance=tolerance)
            if tet_id < 0:
                outside.append((stream, fiber_index, sample_index, score))
                sample_valid.append(False)
            else:
                sample_valid.append(True)
            tet_ids.append(tet_id)
            weights.append(bary)
            bary_min = min(bary_min, float(np.min(bary)))
            bary_max = max(bary_max, float(np.max(bary)))
        # Actual project files store some endpoints as skeleton control points,
        # not volumetric samples. Retain the longest contiguous truly embedded
        # portion; do not invent barycentric coordinates for those controls.
        runs = []
        start = None
        for sample_index, valid in enumerate(sample_valid + [False]):
            if valid and start is None:
                start = sample_index
            elif not valid and start is not None:
                runs.append((start, sample_index))
                start = None
        if runs:
            first, last = max(runs, key=lambda pair: pair[1] - pair[0])
        else:
            first, last = 0, 0
        if last - first >= 2:
            kept_points = points[first:last]
            kept_tets = np.asarray(tet_ids[first:last], dtype=np.int32)
            kept_weights = np.asarray(weights[first:last])
            lengths = np.linalg.norm(np.diff(kept_points, axis=0), axis=1)
            embedded.append(EmbeddedFiber(
                stream, fiber_index, kept_tets, kept_weights,
                kept_points, lengths))
    report = {
        "fiber_count": len(polylines),
        "valid_fiber_count": len(embedded),
        "sample_count": int(sum(len(p[2]) for p in polylines)),
        "embedded_sample_count": int(sum(len(f.tet_ids) for f in embedded)),
        "outside_samples": outside,
        "ambiguous_fibers": ambiguous,
        "minimum_barycentric_weight": float(bary_min),
        "maximum_barycentric_weight": float(bary_max),
        "degenerate_segments": int(sum(
            np.sum(f.rest_segment_lengths < 1e-9) for f in embedded)),
    }
    return embedded, report


def reconstruct_fiber(fiber, vertices, tetrahedra):
    tet_vertices = np.asarray(tetrahedra)[fiber.tet_ids]
    return np.einsum(
        "ni,nij->nj", fiber.barycentric,
        np.asarray(vertices)[tet_vertices])


def build_tet_fiber_directions(vertices, tetrahedra, fibers,
                               nearby_radius=0.02):
    """Average embedded segment directions, then fill only from nearby data."""
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    centroids = vertices[tetrahedra].mean(axis=1)
    sums = np.zeros((len(tetrahedra), 3))
    weights = np.zeros(len(tetrahedra))
    segment_midpoints = []
    segment_directions = []
    segment_lengths = []
    for fiber in fibers:
        points = reconstruct_fiber(fiber, vertices, tetrahedra)
        delta = np.diff(points, axis=0)
        length = np.linalg.norm(delta, axis=1)
        good = length > 1e-9
        direction = delta[good] / length[good, None]
        containing = fiber.tet_ids[:-1][good]
        np.add.at(sums, containing, direction * length[good, None])
        np.add.at(weights, containing, length[good])
        segment_midpoints.extend((0.5 * (points[:-1] + points[1:]))[good])
        segment_directions.extend(direction)
        segment_lengths.extend(length[good])
    direct = weights > 0.0
    directions = np.zeros_like(sums)
    directions[direct] = sums[direct] / weights[direct, None]
    missing = ~direct
    if np.any(missing) and segment_midpoints:
        tree = cKDTree(np.asarray(segment_midpoints))
        distance, index = tree.query(centroids[missing])
        near = distance <= nearby_radius
        missing_ids = np.where(missing)[0]
        directions[missing_ids[near]] = np.asarray(segment_directions)[
            index[near]]
        weights[missing_ids[near]] = np.asarray(segment_lengths)[index[near]]
    valid = weights > 0.0
    directions[valid] /= np.linalg.norm(
        directions[valid], axis=1, keepdims=True)
    return directions, valid


def load_muscle_data(path):
    path = Path(path)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    tetrahedra = orient_tetrahedra(vertices, data["tetrahedra"])
    tetrahedra, quality_report = filter_tetrahedra(vertices, tetrahedra)
    surface = np.asarray(
        data.get("sim_faces")
        if data.get("sim_faces") is not None
        else extract_boundary_faces(tetrahedra), dtype=np.int32)
    patches = attachment_patches_from_tet(data)
    transferred = data.get("_simulation_fiber_embeddings")
    if transferred is not None:
        fibers = list(transferred)
        report = {
            "fiber_count": len(fibers),
            "valid_fiber_count": len(fibers),
            "sample_count": int(sum(len(f.rest_points) for f in fibers)),
            "embedded_sample_count": int(sum(len(f.tet_ids) for f in fibers)),
            "outside_samples": [],
            "ambiguous_fibers": [],
            "minimum_barycentric_weight": float(min(
                np.min(f.barycentric) for f in fibers)),
            "maximum_barycentric_weight": float(max(
                np.max(f.barycentric) for f in fibers)),
            "degenerate_segments": int(sum(np.sum(
                f.rest_segment_lengths < 1e-9) for f in fibers)),
            "simulation_fiber_override": True,
        }
    else:
        fibers, report = embed_fibers(
            vertices, tetrahedra, explicit_fiber_polylines(data), patches)
    report.update(quality_report)
    muscle = MuscleData(
        (path.stem[:-4] if path.stem.endswith("_tet") else path.stem),
        vertices, tetrahedra, surface,
        patches, fibers, data)
    return muscle, report


def fiber_embedding_matrix(fiber, tetrahedra, vertex_count):
    """Sparse matrix mapping flattened tet vertices to fiber point positions."""
    rows, columns, values = [], [], []
    tet_vertices = np.asarray(tetrahedra)[fiber.tet_ids]
    for sample in range(len(fiber.tet_ids)):
        for corner in range(4):
            vertex = int(tet_vertices[sample, corner])
            weight = float(fiber.barycentric[sample, corner])
            for axis in range(3):
                rows.append(3 * sample + axis)
                columns.append(3 * vertex + axis)
                values.append(weight)
    return sp.csr_matrix(
        (values, (rows, columns)),
        shape=(3 * len(fiber.tet_ids), 3 * vertex_count))


def precompute_energy(muscle, nearby_radius=0.02):
    q = muscle.vertices[muscle.tetrahedra]
    dm = np.stack((
        q[:, 0] - q[:, 3], q[:, 1] - q[:, 3],
        q[:, 2] - q[:, 3]), axis=2)
    determinant = np.linalg.det(dm)
    directions, fiber_valid = build_tet_fiber_directions(
        muscle.vertices, muscle.tetrahedra, muscle.fibers,
        nearby_radius=nearby_radius)
    edge_set = set()
    for tet in muscle.tetrahedra:
        for first in range(4):
            for second in range(first + 1, 4):
                edge_set.add(tuple(sorted((
                    int(tet[first]), int(tet[second])))))
    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    edge_length = np.linalg.norm(
        muscle.vertices[edges[:, 0]] - muscle.vertices[edges[:, 1]], axis=1)
    adjacency = sp.csr_matrix((
        np.concatenate((1.0 / edge_length, 1.0 / edge_length)),
        (np.concatenate((edges[:, 0], edges[:, 1])),
         np.concatenate((edges[:, 1], edges[:, 0])))),
        shape=(len(muscle.vertices), len(muscle.vertices)))
    laplacian = (
        sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency)
    return {
        "Dm_inverse": np.linalg.inv(dm),
        "volumes": np.abs(determinant) / 6.0,
        "fiber_directions": directions,
        "fiber_valid": fiber_valid,
        "fiber_matrices": [
            fiber_embedding_matrix(
                fiber, muscle.tetrahedra, len(muscle.vertices))
            for fiber in muscle.fibers],
        "laplacian": laplacian,
    }


def harmonic_attachment_update(vertices, target_by_vertex, precomputed):
    """Smooth exact patch displacement through the current tet state."""
    current = np.asarray(vertices, dtype=np.float64)
    fixed = np.asarray(sorted(target_by_vertex), dtype=np.int32)
    fixed_set = set(int(index) for index in fixed)
    free = np.asarray(
        [index for index in range(len(current))
         if index not in fixed_set], dtype=np.int32)
    target = np.asarray([target_by_vertex[int(index)] for index in fixed])
    # Remove the best common rigid motion first. This makes a global skeleton
    # transform exactly energy-free instead of asking a graph Laplacian (which
    # lacks affine precision on irregular tets) to approximate a rotation.
    source_center = np.mean(current[fixed], axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (current[fixed] - source_center).T @ (
        target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    current = (
        (current - source_center) @ rotation.T + target_center)
    displacement = np.zeros_like(current)
    displacement[fixed] = target - current[fixed]
    if len(free):
        laplacian = precomputed["laplacian"]
        Lff = laplacian[free][:, free].tocsc()
        Lfc = laplacian[free][:, fixed].tocsc()
        solver = splu(Lff + 1e-10 * sp.eye(len(free), format="csc"))
        rhs = -(Lfc @ displacement[fixed])
        for axis in range(3):
            displacement[free, axis] = solver.solve(rhs[:, axis])
    result = current + displacement
    result[fixed] = target
    return result


def deformation_gradients(vertices, muscle, precomputed):
    q = np.asarray(vertices)[muscle.tetrahedra]
    ds = np.stack((
        q[:, 0] - q[:, 3], q[:, 1] - q[:, 3],
        q[:, 2] - q[:, 3]), axis=2)
    return np.einsum(
        "nij,njk->nik", ds, precomputed["Dm_inverse"])


def _scatter_F_gradient(P, muscle, precomputed):
    """Map dE/dF blocks back to world-space vertex gradients."""
    d_ds = np.einsum(
        "nij,nkj->nik", P, precomputed["Dm_inverse"])
    gradient = np.zeros_like(muscle.vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(
            gradient, muscle.tetrahedra[:, corner], d_ds[:, :, corner])
    np.add.at(
        gradient, muscle.tetrahedra[:, 3], -np.sum(d_ds, axis=2))
    return gradient


def _fiber_response(stretch, extension_stiffness, compression_ratio,
                    transition_width, target_scale=1.0):
    """C-infinity blend between weak compression and strong extension."""
    strain = stretch - target_scale
    argument = (stretch - 1.0) / transition_width
    tangent = np.tanh(argument)
    blend = 0.5 * (1.0 + tangent)
    k_compression = extension_stiffness * compression_ratio
    stiffness = (
        k_compression
        + (extension_stiffness - k_compression) * blend)
    d_stiffness = (
        (extension_stiffness - k_compression)
        * 0.5 * (1.0 - tangent * tangent) / transition_width)
    energy = 0.5 * stiffness * strain * strain
    derivative = stiffness * strain + 0.5 * d_stiffness * strain * strain
    return energy, derivative


def bending_energy_gradient(vertices, muscle, precomputed, stiffness):
    """Rotation-invariant weak preservation of each fiber's turning angles."""
    if stiffness <= 0.0:
        return 0.0, np.zeros_like(vertices)
    flat = np.asarray(vertices).ravel()
    gradient_flat = np.zeros_like(flat)
    energy = 0.0
    for fiber, matrix in zip(muscle.fibers,
                             precomputed["fiber_matrices"]):
        points = (matrix @ flat).reshape(-1, 3)
        rest = fiber.rest_points
        point_gradient = np.zeros_like(points)
        for index in range(1, len(points) - 1):
            u = points[index] - points[index - 1]
            v = points[index + 1] - points[index]
            ru = rest[index] - rest[index - 1]
            rv = rest[index + 1] - rest[index]
            lu = max(float(np.linalg.norm(u)), 1e-12)
            lv = max(float(np.linalg.norm(v)), 1e-12)
            rlu = max(float(np.linalg.norm(ru)), 1e-12)
            rlv = max(float(np.linalg.norm(rv)), 1e-12)
            cosine = float(np.dot(u, v) / (lu * lv))
            rest_cosine = float(np.dot(ru, rv) / (rlu * rlv))
            difference = cosine - rest_cosine
            energy += 0.5 * stiffness * difference * difference
            dc_du = v / (lu * lv) - cosine * u / (lu * lu)
            dc_dv = u / (lu * lv) - cosine * v / (lv * lv)
            scale = stiffness * difference
            point_gradient[index - 1] -= scale * dc_du
            point_gradient[index] += scale * (dc_du - dc_dv)
            point_gradient[index + 1] += scale * dc_dv
        gradient_flat += matrix.T @ point_gradient.ravel()
    return float(energy), gradient_flat.reshape(-1, 3)


def total_energy_gradient(vertices, muscle, precomputed, config):
    """Matrix + volume + anatomical fiber + optional path energy."""
    vertices = np.asarray(vertices, dtype=np.float64)
    F = deformation_gradients(vertices, muscle, precomputed)
    J = np.linalg.det(F)
    minimum_J = float(config["material"]["minimum_J"])
    if np.any(~np.isfinite(J)) or np.any(J <= minimum_J):
        return np.inf, np.full_like(vertices, np.nan), {
            "matrix": np.inf, "volume": np.inf, "fiber": np.inf,
            "bending": np.inf, "minimum_J": float(np.nanmin(J)),
            "maximum_J": float(np.nanmax(J)),
            "inverted_tets": int(np.sum(J <= 0.0)),
            "volume_ratio": float(np.nan),
        }
    inverse_transpose = np.linalg.inv(F).transpose(0, 2, 1)
    log_J = np.log(J)
    volumes = precomputed["volumes"]
    mu = float(config["material"]["shear_modulus"])
    bulk = float(config["material"]["bulk_modulus"])
    invariant = np.einsum("nij,nij->n", F, F)
    matrix_density = 0.5 * mu * (invariant - 3.0) - mu * log_J
    volume_density = 0.5 * bulk * log_J * log_J
    P = volumes[:, None, None] * (
        mu * (F - inverse_transpose)
        + (bulk * log_J)[:, None, None] * inverse_transpose)
    matrix_energy = float(np.sum(volumes * matrix_density))
    volume_energy = float(np.sum(volumes * volume_density))

    fiber_energy = 0.0
    fiber_cfg = config["fiber"]
    if fiber_cfg.get("enabled", True) and fiber_cfg.get(
            "anisotropy_enabled", True):
        valid = precomputed["fiber_valid"]
        a0 = precomputed["fiber_directions"][valid]
        Fa = np.einsum("nij,nj->ni", F[valid], a0)
        stretch = np.linalg.norm(Fa, axis=1)
        target_scale = (
            float(fiber_cfg.get("target_scale", 1.0))
            if fiber_cfg.get("active_shortening_enabled", False) else 1.0)
        density, derivative = _fiber_response(
            stretch, float(fiber_cfg["extension_stiffness"]),
            float(fiber_cfg["compression_stiffness_ratio"]),
            float(fiber_cfg.get("transition_width", 0.03)),
            target_scale)
        fiber_energy = float(np.sum(volumes[valid] * density))
        direction_gradient = Fa / np.maximum(stretch[:, None], 1e-12)
        P_fiber = (
            volumes[valid] * derivative)[:, None, None] * np.einsum(
                "ni,nj->nij", direction_gradient, a0)
        P[valid] += P_fiber
    gradient = _scatter_F_gradient(P, muscle, precomputed)

    bending_energy = 0.0
    if fiber_cfg.get("bending_enabled", False):
        bending_energy, bending_gradient = bending_energy_gradient(
            vertices, muscle, precomputed,
            float(fiber_cfg.get("bending_stiffness", 0.0)))
        gradient += bending_gradient
    terms = {
        "matrix": matrix_energy,
        "volume": volume_energy,
        "fiber": fiber_energy,
        "bending": bending_energy,
        "minimum_J": float(np.min(J)),
        "maximum_J": float(np.max(J)),
        "inverted_tets": int(np.sum(J <= 0.0)),
        "volume_ratio": float(np.sum(volumes * J) / np.sum(volumes)),
    }
    return (matrix_energy + volume_energy + fiber_energy + bending_energy,
            gradient, terms)


def fiber_diagnostics(vertices, muscle, precomputed):
    diagnostics = []
    for fiber, matrix in zip(muscle.fibers,
                             precomputed["fiber_matrices"]):
        points = (matrix @ np.asarray(vertices).ravel()).reshape(-1, 3)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        ratios = lengths / np.maximum(fiber.rest_segment_lengths, 1e-12)
        turning = []
        for index in range(1, len(points) - 1):
            a = points[index] - points[index - 1]
            b = points[index + 1] - points[index]
            cosine = np.dot(a, b) / max(
                np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
            turning.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
        diagnostics.append({
            "stream": fiber.stream_index,
            "fiber": fiber.fiber_index,
            "rest_length": fiber.rest_total_length,
            "current_length": float(np.sum(lengths)),
            "length_ratio": float(
                np.sum(lengths) / max(fiber.rest_total_length, 1e-12)),
            "minimum_segment_ratio": float(np.min(ratios)),
            "maximum_segment_ratio": float(np.max(ratios)),
            "maximum_turning_angle": float(max(turning, default=0.0)),
            "maximum_turning_sample": (
                int(np.argmax(turning) + 1) if turning else -1),
            "turning_angles": np.asarray(turning, dtype=np.float64),
            "points": points,
        })
    return diagnostics
