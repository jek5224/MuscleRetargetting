"""Inspect source fibers against every Semitendinosus surface representation."""
import argparse
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.cross_sections import (
    representative_bundle_path, rotation_minimizing_frames)
from muscle_sim.fiber_repair import (
    diagnose_fibers, load_surface, write_json, write_obj_polylines)
from viewer.isolated_muscle import load_muscle_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--fibers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    section = config["fiber_repair"]
    muscle, source_report = load_muscle_data(
        args.fibers or config["source_tet_asset"])
    surfaces = {
        "immutable_reference": load_surface(
            config["immutable_reference_asset"]),
        "invalid_tet_boundary": load_surface(
            config["invalid_tet_boundary_asset"], boundary_from_tets=True),
        "path_b": load_surface(config["path_b_surface"]),
        "c3": load_surface(args.surface),
    }
    frames = rotation_minimizing_frames(
        representative_bundle_path(muscle.fibers))
    report = diagnose_fibers(muscle.fibers, surfaces, frames, section)
    report["source_embedding_report"] = source_report
    report["fiber_semantics"] = {
        "required_volumetric_samples": int(sum(
            len(fiber.rest_points) for fiber in muscle.fibers)),
        "required_volumetric_fibers": len(muscle.fibers),
        "interpretation": (
            "load_muscle_data already excluded external skeleton controls and "
            "retained the longest contiguous volumetric run per fiber"),
        "excluded_source_controls_are_nonvolumetric": True,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "fiber_surface_inconsistency_report.json", report)
    rows = report["sample_rows"]
    if rows:
        points = np.asarray([row["sample_position"] for row in rows])
        cloud = np.column_stack((points, np.full((len(points), 3), 255)))
        np.savetxt(
            output / "outside_fiber_samples.ply", cloud,
            header=("ply\nformat ascii 1.0\nelement vertex %d\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\n"
                    "property uchar blue\nend_header" % len(points)),
            comments="", fmt=["%.17g", "%.17g", "%.17g", "%d", "%d", "%d"])
    write_obj_polylines(output / "outside_fiber_segments.obj", [], muscle.fibers)
    with (output / "fiber_surface_closest_segments.obj").open("w") as handle:
        for row in rows:
            for point in (row["sample_position"], row["closest_point"]):
                handle.write("v %.17g %.17g %.17g\n" % tuple(point))
        for index in range(len(rows)):
            handle.write(f"l {2 * index + 1} {2 * index + 2}\n")
    print(output / "fiber_surface_inconsistency_report.json")


if __name__ == "__main__":
    main()
