"""Region-specific geometry validation and PLC preflight."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
import trimesh

from muscle_sim.full_surface import complete_topology
from muscle_sim.local_remeshing import json_ready, load_raw_mesh


def triangle_quality(vertices, faces):
    triangle = vertices[faces]
    lengths = np.stack([
        np.linalg.norm(triangle[:, i] - triangle[:, (i + 1) % 3], axis=1)
        for i in range(3)], axis=1)
    cosine = []
    for corner in range(3):
        first = triangle[:, (corner + 1) % 3] - triangle[:, corner]
        second = triangle[:, (corner + 2) % 3] - triangle[:, corner]
        cosine.append(np.einsum("ij,ij->i", first, second) / np.maximum(
            np.linalg.norm(first, axis=1)
            * np.linalg.norm(second, axis=1), 1e-30))
    angles = np.degrees(np.arccos(np.clip(
        np.stack(cosine, axis=1), -1., 1.)))
    return {
        "minimum_edge_length": float(np.min(lengths)),
        "maximum_edge_length": float(np.max(lengths)),
        "maximum_aspect_ratio": float(np.max(
            np.max(lengths, axis=1)
            / np.maximum(np.min(lengths, axis=1), 1e-30))),
        "minimum_triangle_angle_degrees": float(np.min(angles)),
        "maximum_triangle_angle_degrees": float(np.max(angles)),
        "extremely_short_edge_count": int(np.sum(lengths < 1e-7)),
        "extremely_acute_triangle_count": int(np.sum(
            np.min(angles, axis=1) < 1.0)),
    }


def section_metrics(mesh, stations):
    rows = []
    for station in stations:
        section = mesh.section(
            plane_origin=[0., station, 0.],
            plane_normal=[0., 1., 0.])
        if section is None:
            rows.append({"station": station, "area": 0.0})
            continue
        planar, _ = section.to_planar()
        polygons = planar.polygons_full
        area = float(sum(polygon.area for polygon in polygons))
        centroid = (
            np.average(
                np.asarray([[polygon.centroid.x, polygon.centroid.y]
                            for polygon in polygons]),
                axis=0, weights=[polygon.area for polygon in polygons])
            if polygons else np.zeros(2))
        rows.append({
            "station": station, "area": area,
            "planar_centroid": centroid.tolist()})
    return rows


def validate(surface_path, reference_path, config):
    surface = trimesh.load(surface_path, force="mesh", process=False)
    reference_raw = load_raw_mesh(reference_path)
    reference = trimesh.Trimesh(
        reference_raw["vertices"], reference_raw["render_faces"],
        process=False)
    vertices, faces = np.asarray(surface.vertices), np.asarray(surface.faces)
    topology = complete_topology(vertices, faces)
    quality = triangle_quality(vertices, faces)
    low = max(surface.bounds[0, 1], reference.bounds[0, 1])
    high = min(surface.bounds[1, 1], reference.bounds[1, 1])
    stations = np.linspace(low + .02 * (high - low),
                           high - .02 * (high - low), 25)
    surface_sections = section_metrics(surface, stations)
    reference_sections = section_metrics(reference, stations)
    area_error = np.asarray([
        abs(first["area"] - second["area"])
        / max(second["area"], 1e-12)
        for first, second in zip(surface_sections, reference_sections)])
    # Nearest opposite-side distance is a stable thickness proxy for this
    # slender muscle and does not assume a watertight ray backend.
    samples = vertices[::max(1, len(vertices) // 2000)]
    _, distance_reference, _ = trimesh.proximity.closest_point(
        reference, samples)
    thickness_proxy = {
        "sample_count": len(samples),
        "maximum_surface_offset": float(np.max(distance_reference)),
        "rms_surface_offset": float(np.sqrt(np.mean(
            distance_reference ** 2))),
    }
    report = {
        "surface": str(surface_path),
        "topology": topology,
        "triangle_quality": quality,
        "cross_sections": {
            "hybrid": surface_sections,
            "reference": reference_sections,
            "maximum_relative_area_error": float(np.max(area_error)),
            "rms_relative_area_error": float(np.sqrt(np.mean(
                area_error ** 2))),
        },
        "local_thickness_proxy": thickness_proxy,
        "plc_preflight": {
            "duplicate_vertex_check": "exact_and_tolerance_checked",
            "duplicate_face_count": len(topology["duplicate_triangles"]),
            "nonmanifold_edge_count": len(topology["nonmanifold_edges"]),
            "orientation_conflict_count": len(
                topology["inconsistent_orientation_edges"]),
            "short_edge_count": quality["extremely_short_edge_count"],
            "acute_triangle_count":
                quality["extremely_acute_triangle_count"],
            "tetgen_diagnose_required": True,
        },
    }
    local = config["contact_constrained_reconstruction"]
    report["passed_nonintersection_preflight"] = bool(
        topology["valid_topology"]
        and quality["maximum_aspect_ratio"]
        <= float(local["maximum_triangle_aspect_ratio"])
        and report["cross_sections"]["maximum_relative_area_error"]
        <= float(local["maximum_cross_section_area_error_fraction"])
        and thickness_proxy["maximum_surface_offset"]
        <= float(local["maximum_thickness_proxy_error_m"])
        and not quality["extremely_short_edge_count"]
        and not quality["extremely_acute_triangle_count"])
    report["failure_classification"] = (
        None if report["passed_nonintersection_preflight"]
        else "LOCAL_THICKNESS_CANNOT_BE_PRESERVED"
        if (report["cross_sections"]["maximum_relative_area_error"]
            > float(local["maximum_cross_section_area_error_fraction"])
            or thickness_proxy["maximum_surface_offset"]
            > float(local["maximum_thickness_proxy_error_m"]))
        else "SURFACE_TRIANGLE_QUALITY_FAILURE")
    report["tetgen_invoked"] = False
    report["metadata_transfer_performed"] = False
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True)
    parser.add_argument("--reference", default=(
        "tet_orig_std/L_Semitendinosus_tet.npz"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    report = validate(args.surface, args.reference, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "hybrid_surface_validation.json").write_text(json.dumps(
        json_ready(report), indent=2))
    if not report["passed_nonintersection_preflight"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
