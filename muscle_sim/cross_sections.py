"""Fiber-bundle coordinates and joint cross-section preservation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import trimesh

from muscle_sim.full_surface import complete_topology
from muscle_sim.hybrid_surface import contact_metrics
from muscle_sim.local_remeshing import (
    active_contact_points, json_ready, load_raw_mesh)
from viewer.isolated_muscle import (
    extract_boundary_faces, load_muscle_data, orient_tetrahedra)


def resample_polyline(points, count):
    points = np.asarray(points)
    length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
        np.diff(points, axis=0), axis=1))))
    if length[-1] <= 1e-12:
        raise ValueError("degenerate fiber polyline")
    parameter = length / length[-1]
    sample = np.linspace(0.0, 1.0, count)
    return np.column_stack([
        np.interp(sample, parameter, points[:, axis])
        for axis in range(3)])


def representative_bundle_path(fibers, count=96, smoothing_passes=3):
    curves = np.asarray([
        resample_polyline(fiber.rest_points, count) for fiber in fibers])
    path = np.median(curves, axis=0)
    for _ in range(smoothing_passes):
        smooth = path.copy()
        smooth[1:-1] = (
            path[:-2] + 2.0 * path[1:-1] + path[2:]) / 4.0
        path = smooth
    return path


def rotation_minimizing_frames(path):
    path = np.asarray(path)
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(
        np.linalg.norm(tangent, axis=1, keepdims=True), 1e-30)
    seed = np.asarray([1., 0., 0.])
    if abs(np.dot(seed, tangent[0])) > .9:
        seed = np.asarray([0., 0., 1.])
    first = seed - np.dot(seed, tangent[0]) * tangent[0]
    first /= np.linalg.norm(first)
    u = np.zeros_like(path)
    u[0] = first
    for index in range(1, len(path)):
        axis = np.cross(tangent[index - 1], tangent[index])
        sine = np.linalg.norm(axis)
        cosine = np.clip(np.dot(
            tangent[index - 1], tangent[index]), -1., 1.)
        if sine < 1e-12:
            transported = u[index - 1]
        else:
            axis /= sine
            angle = np.arctan2(sine, cosine)
            previous = u[index - 1]
            transported = (
                previous * np.cos(angle)
                + np.cross(axis, previous) * np.sin(angle)
                + axis * np.dot(axis, previous) * (1. - np.cos(angle)))
        transported -= np.dot(transported, tangent[index]) * tangent[index]
        u[index] = transported / max(np.linalg.norm(transported), 1e-30)
    v = np.cross(tangent, u)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)
    arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(
        np.diff(path, axis=0), axis=1))))
    arc /= arc[-1]
    return {"origin": path, "tangent": tangent, "u": u, "v": v, "s": arc}


def assign_bundle_coordinates(vertices, frames):
    tree = cKDTree(frames["origin"])
    _, index = tree.query(vertices)
    offset = vertices - frames["origin"][index]
    return {
        "s": frames["s"][index],
        "u": np.einsum("ij,ij->i", offset, frames["u"][index]),
        "v": np.einsum("ij,ij->i", offset, frames["v"][index]),
        "nearest_bundle_path_sample": index,
    }


def resample_closed_contour(points, count=128):
    points = np.asarray(points)
    if np.linalg.norm(points[0] - points[-1]) < 1e-10:
        points = points[:-1]
    closed = np.vstack((points, points[0]))
    length = np.concatenate(([0.], np.cumsum(np.linalg.norm(
        np.diff(closed, axis=0), axis=1))))
    if length[-1] <= 1e-12:
        raise ValueError("degenerate section contour")
    sample = np.linspace(0., length[-1], count, endpoint=False)
    result = np.column_stack([
        np.interp(sample, length, closed[:, axis]) for axis in range(3)])
    return result


def contour_2d(points, frame):
    offset = points - frame["origin"]
    return np.column_stack((
        offset @ frame["u"], offset @ frame["v"]))


def polygon_descriptors(points):
    points = np.asarray(points)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - (
        following[:, 0] * points[:, 1])
    signed_area = .5 * np.sum(cross)
    if abs(signed_area) <= 1e-14:
        raise ValueError("zero-area section contour")
    if signed_area < 0:
        points = points[::-1].copy()
        following = np.roll(points, -1, axis=0)
        cross = points[:, 0] * following[:, 1] - (
            following[:, 0] * points[:, 1])
        signed_area = .5 * np.sum(cross)
    centroid = np.sum(
        (points + following) * cross[:, None], axis=0) / (
            6. * signed_area)
    centered = points - centroid
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    perimeter = float(np.sum(np.linalg.norm(
        following - points, axis=1)))
    return {
        "points": points, "area": float(signed_area),
        "centroid": centroid, "covariance": covariance,
        "principal_values": values, "principal_vectors": vectors,
        "radius_proxies": np.sqrt(np.maximum(values, 0.)),
        "perimeter": perimeter,
    }


def extract_section(mesh, frame, sample_count=128):
    section = mesh.section(
        plane_origin=frame["origin"], plane_normal=frame["tangent"])
    if section is None:
        raise ValueError("missing section loop")
    loops = section.discrete
    if not loops:
        raise ValueError("missing section loop")
    # Intended external contour is the largest closed loop. Other substantial
    # loops are ambiguous and rejected.
    areas = []
    candidates = []
    for loop in loops:
        sampled = resample_closed_contour(loop, sample_count)
        local = contour_2d(sampled, frame)
        descriptor = polygon_descriptors(local)
        candidates.append((sampled, descriptor))
        areas.append(descriptor["area"])
    order = np.argsort(areas)[::-1]
    if len(order) > 1 and areas[order[1]] > .05 * areas[order[0]]:
        raise ValueError("multiple substantial section loops")
    points, descriptor = candidates[int(order[0])]
    descriptor["points_3d"] = points
    return descriptor


def cyclic_correspondence(reference, current):
    reference, current = np.asarray(reference), np.asarray(current)
    if len(reference) != len(current):
        raise ValueError("contours require equal resampling")
    best = None
    for reversed_curve in (False, True):
        curve = current[::-1] if reversed_curve else current
        for shift in range(len(curve)):
            rolled = np.roll(curve, shift, axis=0)
            error = float(np.mean(np.sum(
                rolled - reference, axis=1) ** 2))
            key = (error, reversed_curve, shift)
            if best is None or key < best[0]:
                best = (key, rolled)
    return best[1], {
        "mean_squared_error": best[0][0],
        "reversed": best[0][1], "cyclic_shift": best[0][2]}


def thickness_pairs(contour, pair_count=16):
    contour = np.asarray(contour)
    count = len(contour)
    ids = np.linspace(0, count // 2 - 1, pair_count).astype(int)
    pairs = np.column_stack((ids, ids + count // 2))
    lengths = np.linalg.norm(
        contour[pairs[:, 0]] - contour[pairs[:, 1]], axis=1)
    return pairs, lengths


def section_stations(frames, config):
    total = np.sum(np.linalg.norm(np.diff(
        frames["origin"], axis=0), axis=1))
    spacing = min(
        float(config["contact_section_spacing_m"]),
        float(config["distal_section_spacing_m"]),
        float(config["belly_section_spacing_m"]))
    count = max(12, int(np.ceil(total / spacing)))
    return np.linspace(.03, .97, count)


def frame_at(frames, value):
    index = int(np.argmin(np.abs(frames["s"] - value)))
    return {key: frames[key][index] for key in (
        "origin", "tangent", "u", "v")} | {"index": index, "s": value}


def build_section_data(reference_mesh, current_mesh, frames, config):
    rows = []
    for section_id, value in enumerate(section_stations(frames, config)):
        frame = frame_at(frames, value)
        reference = extract_section(reference_mesh, frame)
        current = extract_section(current_mesh, frame)
        current_points, mapping = cyclic_correspondence(
            reference["points"], current["points"])
        current = polygon_descriptors(current_points)
        pairs, reference_thickness = thickness_pairs(reference["points"])
        current_thickness = np.linalg.norm(
            current["points"][pairs[:, 0]]
            - current["points"][pairs[:, 1]], axis=1)
        rows.append({
            "section_id": section_id, "s": value, "frame": frame,
            "reference": reference, "current": current,
            "correspondence": mapping, "thickness_pairs": pairs,
            "reference_thickness": reference_thickness,
            "current_thickness": current_thickness,
        })
    return rows


def covariance_map(current, reference, regularization=1e-12):
    def root(matrix, inverse=False):
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, regularization)
        power = -0.5 if inverse else 0.5
        return vectors @ np.diag(values ** power) @ vectors.T
    return root(reference["covariance"]) @ root(
        current["covariance"], inverse=True)


def apply_section_transforms(vertices, frames, sections, strength):
    section_s = np.asarray([row["s"] for row in sections])
    matrices = np.asarray([
        covariance_map(row["current"], row["reference"])
        for row in sections])
    translations = np.asarray([
        row["reference"]["centroid"]
        - matrices[index] @ row["current"]["centroid"]
        for index, row in enumerate(sections)])
    coordinates = assign_bundle_coordinates(vertices, frames)
    result = vertices.copy()
    for vertex in range(len(vertices)):
        s = coordinates["s"][vertex]
        upper = int(np.searchsorted(section_s, s))
        upper = min(max(upper, 1), len(section_s) - 1)
        lower = upper - 1
        blend = np.clip(
            (s - section_s[lower])
            / max(section_s[upper] - section_s[lower], 1e-12), 0., 1.)
        matrix = (1. - blend) * matrices[lower] + blend * matrices[upper]
        translation = (
            (1. - blend) * translations[lower]
            + blend * translations[upper])
        local = np.asarray([coordinates["u"][vertex],
                            coordinates["v"][vertex]])
        fitted = matrix @ local + translation
        fitted = (1. - strength) * local + strength * fitted
        frame_index = coordinates["nearest_bundle_path_sample"][vertex]
        result[vertex] = (
            frames["origin"][frame_index]
            + fitted[0] * frames["u"][frame_index]
            + fitted[1] * frames["v"][frame_index]
            + np.dot(
                vertices[vertex] - frames["origin"][frame_index],
                frames["tangent"][frame_index])
            * frames["tangent"][frame_index])
    return result


def apply_whole_section_contact_translation(
        fitted, base, contact_fitted, frames, contact_points,
        radius=0.006):
    """Transfer contact correction as a smooth rigid translation per section."""
    coordinates = assign_bundle_coordinates(base, frames)
    contact_distance = cKDTree(contact_points).query(base)[0]
    displacement = contact_fitted - base
    translations = np.zeros_like(frames["origin"])
    confidence = np.zeros(len(translations))
    for frame_id in range(len(translations)):
        members = (
            (coordinates["nearest_bundle_path_sample"] == frame_id)
            & (contact_distance <= radius))
        if np.any(members):
            translations[frame_id] = np.median(
                displacement[members], axis=0)
            confidence[frame_id] = 1.0
    known = np.where(confidence > 0)[0]
    if not len(known):
        return fitted.copy(), translations
    for axis in range(3):
        translations[:, axis] = np.interp(
            np.arange(len(translations)), known,
            translations[known, axis],
            left=translations[known[0], axis],
            right=translations[known[-1], axis])
    for _ in range(4):
        translations[1:-1] = (
            translations[:-2] + 2. * translations[1:-1]
            + translations[2:]) / 4.
    result = fitted + translations[
        coordinates["nearest_bundle_path_sample"]]
    return result, translations


def align_whole_sections_to_contact(
        vertices, faces, frames, contact_corridor_points,
        iterations=2):
    """Translate full sections using old-corridor to candidate residuals."""
    result = vertices.copy()
    total = np.zeros_like(frames["origin"])
    frame_tree = cKDTree(frames["origin"])
    for _ in range(iterations):
        mesh = trimesh.Trimesh(
            vertices=result, faces=faces, process=False)
        closest, _, _ = trimesh.proximity.closest_point(
            mesh, contact_corridor_points)
        residual = contact_corridor_points - closest
        _, frame_ids = frame_tree.query(contact_corridor_points)
        translations = np.zeros_like(frames["origin"])
        counts = np.zeros(len(translations))
        for frame_id, value in zip(frame_ids, residual):
            translations[frame_id] += value
            counts[frame_id] += 1.
        known = np.where(counts > 0)[0]
        if not len(known):
            break
        translations[known] /= counts[known, None]
        for axis in range(3):
            translations[:, axis] = np.interp(
                np.arange(len(translations)), known,
                translations[known, axis],
                left=translations[known[0], axis],
                right=translations[known[-1], axis])
        for _smooth in range(4):
            translations[1:-1] = (
                translations[:-2] + 2. * translations[1:-1]
                + translations[2:]) / 4.
        coordinates = assign_bundle_coordinates(result, frames)
        result += translations[
            coordinates["nearest_bundle_path_sample"]]
        total += translations
    return result, total


def section_error(reference, current):
    return {
        "relative_area_error": abs(
            current["area"] - reference["area"]) / reference["area"],
        "centroid_offset": float(np.linalg.norm(
            current["centroid"] - reference["centroid"])),
        "major_radius_relative_error": abs(
            current["radius_proxies"][0]
            - reference["radius_proxies"][0])
        / max(reference["radius_proxies"][0], 1e-12),
        "minor_radius_relative_error": abs(
            current["radius_proxies"][1]
            - reference["radius_proxies"][1])
        / max(reference["radius_proxies"][1], 1e-12),
        "perimeter_relative_error": abs(
            current["perimeter"] - reference["perimeter"])
        / reference["perimeter"],
    }


def validate_sections(reference_mesh, candidate_mesh, frames, config):
    data = build_section_data(reference_mesh, candidate_mesh, frames, config)
    errors, thickness_rows = [], []
    for row in data:
        error = section_error(row["reference"], row["current"])
        error.update({"section_id": row["section_id"], "s": row["s"]})
        errors.append(error)
        for pair_id, (reference, current) in enumerate(zip(
                row["reference_thickness"], row["current_thickness"])):
            thickness_rows.append({
                "section_id": row["section_id"], "pair_id": pair_id,
                "reference": reference, "current": current,
                "absolute_error": abs(current - reference),
                "relative_error": abs(current - reference)
                / max(reference, 1e-12),
            })
    return data, errors, thickness_rows
