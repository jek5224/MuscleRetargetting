"""Generate C0-C5 joint full-contour constrained candidates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml

from muscle_sim.cross_sections import (
    align_whole_sections_to_contact, apply_section_transforms,
    apply_whole_section_contact_translation,
    build_section_data,
    representative_bundle_path, rotation_minimizing_frames,
    validate_sections)
from muscle_sim.full_surface import complete_topology
from muscle_sim.hybrid_surface import contact_metrics
from muscle_sim.local_remeshing import (
    active_contact_points, json_ready, load_raw_mesh)
from viewer.isolated_muscle import (
    extract_boundary_faces, load_muscle_data, orient_tetrahedra)


def triangle_aspect(vertices, faces):
    triangle = vertices[faces]
    lengths = np.stack([
        np.linalg.norm(triangle[:, i] - triangle[:, (i + 1) % 3], axis=1)
        for i in range(3)], axis=1)
    return float(np.max(
        np.max(lengths, axis=1)
        / np.maximum(np.min(lengths, axis=1), 1e-30)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-surface", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--contact-fit-surface", default=(
        "assets/generated/Left_Semitendinosus_contact_constrained_surface/"
        "hybrid_surface.obj"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    local = config["cross_section_constraints"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    muscle, _ = load_muscle_data(config["current_tet_asset"])
    frames = rotation_minimizing_frames(
        representative_bundle_path(muscle.fibers))
    reference_raw = load_raw_mesh(args.reference)
    reference = trimesh.Trimesh(
        reference_raw["vertices"], reference_raw["render_faces"],
        process=False)
    base = trimesh.load(args.base_surface, force="mesh", process=False)
    vertices, faces = np.asarray(base.vertices), np.asarray(base.faces)
    sections = build_section_data(reference, base, frames, local)
    old_raw = load_raw_mesh(config["current_tet_asset"])
    old_vertices = np.asarray(old_raw["vertices"])
    old_faces = extract_boundary_faces(orient_tetrahedra(
        old_vertices, old_raw["tetrahedra"]))
    contact_points = active_contact_points(
        Path(config["diagnostic_dir"]) / "active_contact_multipliers.json",
        config["muscle_name"], old_vertices, old_faces)
    old_centroids = np.mean(old_vertices[old_faces], axis=1)
    critical_centroids = old_centroids[
        __import__("scipy").spatial.cKDTree(contact_points).query(
            old_centroids)[0] <= .005]
    contact_fit = trimesh.load(
        args.contact_fit_surface, force="mesh", process=False)
    if (len(contact_fit.vertices) != len(vertices)
            or not np.array_equal(contact_fit.faces, faces)):
        raise ValueError(
            "contact-fit and Path B surfaces must share topology")
    definitions = [
        ("C0", 0.0), ("C1", .25), ("C2", .5),
        ("C3", .75), ("C4", .9), ("C5", .95),
        ("C6", 1.0)]
    candidates, all_profiles, all_thickness = [], [], []
    for name, strength in definitions:
        fitted = (
            vertices.copy() if strength == 0.
            else apply_section_transforms(
                vertices, frames, sections, strength))
        if strength > 0.:
            fitted, section_translations = (
                apply_whole_section_contact_translation(
                    fitted, vertices, np.asarray(contact_fit.vertices),
                    frames, contact_points))
            fitted, residual_translation = align_whole_sections_to_contact(
                fitted, faces, frames, critical_centroids)
            section_translations += residual_translation
        else:
            section_translations = np.zeros_like(frames["origin"])
        mesh = trimesh.Trimesh(
            vertices=fitted, faces=faces, process=False)
        try:
            _, errors, thickness = validate_sections(
                reference, mesh, frames, local)
            contour_valid = True
        except ValueError as error:
            errors, thickness = [], []
            contour_valid = False
            contour_error = str(error)
        contact = contact_metrics(
            mesh, old_vertices, old_faces, contact_points,
            .005)
        topology = complete_topology(fitted, faces)
        area = np.asarray([
            row["relative_area_error"] for row in errors])
        centroid = np.asarray([
            row["centroid_offset"] for row in errors])
        thick_abs = np.asarray([
            row["absolute_error"] for row in thickness])
        thick_rel = np.asarray([
            row["relative_error"] for row in thickness])
        maximum_area = float(np.max(area)) if len(area) else np.inf
        rms_area = float(np.sqrt(np.mean(area ** 2))) if len(area) else np.inf
        maximum_centroid = (
            float(np.max(centroid)) if len(centroid) else np.inf)
        maximum_thickness = (
            float(np.max(thick_abs)) if len(thick_abs) else np.inf)
        maximum_thickness_rel = (
            float(np.max(thick_rel)) if len(thick_rel) else np.inf)
        aspect = triangle_aspect(fitted, faces)
        accepted = bool(
            contour_valid and topology["valid_topology"]
            and contact["maximum_deviation"]
            <= float(local["contact_max_deviation_m"])
            and contact["percentile_95_deviation"]
            <= float(local["contact_p95_deviation_m"])
            and maximum_area
            <= float(local["maximum_relative_area_error"])
            and rms_area <= float(local["RMS_relative_area_error"])
            and maximum_centroid
            <= float(local["maximum_centroid_offset_m"])
            and maximum_thickness
            <= float(local["maximum_thickness_absolute_error_m"])
            and maximum_thickness_rel
            <= float(local["maximum_thickness_relative_error"])
            and aspect <= float(local["maximum_triangle_aspect_ratio"]))
        surface_path = output / f"{name}_surface.obj"
        mesh.export(surface_path)
        row = {
            "name": name, "strength": strength, "accepted": accepted,
            "contour_valid": contour_valid,
            "contour_error": (
                None if contour_valid else contour_error),
            "topology": topology, "contact": contact,
            "maximum_relative_area_error": maximum_area,
            "RMS_relative_area_error": rms_area,
            "maximum_centroid_offset": maximum_centroid,
            "maximum_thickness_absolute_error": maximum_thickness,
            "maximum_thickness_relative_error": maximum_thickness_rel,
            "maximum_triangle_aspect_ratio": aspect,
            "surface_path": str(surface_path),
            "maximum_whole_section_translation": float(np.max(
                np.linalg.norm(section_translations, axis=1))),
        }
        candidates.append(row)
        all_profiles.extend({"candidate": name, **value}
                            for value in errors)
        all_thickness.extend({"candidate": name, **value}
                             for value in thickness)
    ranking = sorted(candidates, key=lambda row: (
        not row["accepted"], row["maximum_relative_area_error"],
        row["contact"]["maximum_deviation"]))
    accepted = [row for row in ranking if row["accepted"]]
    report = {
        "candidates": ranking,
        "accepted_candidate": accepted[0]["name"] if accepted else None,
        "failure_classification": (
            None if accepted else "LOCAL_SHAPE_PRESERVATION_INFEASIBLE"),
        "PLC_invoked": False, "tetrahedralization_invoked": False,
        "metadata_transfer_performed": False,
    }
    (output / "section_error_summary.json").write_text(json.dumps(
        json_ready(report), indent=2))
    if all_profiles:
        with (output / "cross_section_profiles.csv").open(
                "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=all_profiles[0].keys())
            writer.writeheader()
            writer.writerows(all_profiles)
    if all_thickness:
        with (output / "thickness_pairs.csv").open(
                "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=all_thickness[0].keys())
            writer.writeheader()
            writer.writerows(all_thickness)
    if not accepted:
        raise ValueError("no cross-section candidate passes all gates")
    Path(accepted[0]["surface_path"]).replace(output / "hybrid_surface.obj")


if __name__ == "__main__":
    main()
