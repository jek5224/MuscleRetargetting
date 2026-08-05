"""Classify every sparse anchor chord against the accepted C3 surface."""
import argparse
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.fiber_repair import load_repaired_fibers, load_surface, write_json
from muscle_sim.fiber_routing import classify_straight_segment
from viewer.isolated_muscle import TetLocator


def write_chords(path, rows):
    with Path(path).open("w") as handle:
        vertex = 1
        for row in rows:
            if row["classification"] == "STRAIGHT_SEGMENT_VALID":
                continue
            handle.write("g fiber_%d_segment_%d\n" % (
                row["fiber_id"], row["source_segment"]))
            for point in (row["start_anchor"], row["end_anchor"]):
                handle.write("v %.17g %.17g %.17g\n" % tuple(point))
            handle.write(f"l {vertex} {vertex + 1}\n")
            vertex += 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--tet-mesh", required=True)
    parser.add_argument("--fibers")
    parser.add_argument("--anchor-candidate", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["fiber_path_routing"]
    surface = load_surface(args.surface)
    tet_path = Path(args.tet_mesh)
    if tet_path.is_dir():
        tet_path /= "optimized_tet_candidate.npz"
    tet_data = np.load(tet_path)
    vertices, tetrahedra = tet_data["vertices"], tet_data["tetrahedra"]
    locator = TetLocator(vertices, tetrahedra)
    anchor_path = Path(args.anchor_candidate)
    if anchor_path.is_dir():
        anchor_path /= "repaired_simulation_fibers.npz"
    repaired, _ = load_repaired_fibers(anchor_path)
    rows = []
    for fiber in repaired:
        points = np.asarray(fiber["points"])
        for segment, (start, end) in enumerate(zip(
                points[:-1], points[1:])):
            result = classify_straight_segment(
                surface, start, end,
                float(settings["interior_margin_m"]))
            samples = result["sampled_points"][::8]
            if not np.array_equal(
                    samples[-1], result["sampled_points"][-1]):
                samples = np.vstack((samples, result["sampled_points"][-1]))
            nearby_tets = sorted(set(
                int(value) for value in (
                    locator.locate(
                        point, tolerance=2e-8, candidates=32)[0]
                    for point in samples)
                if value >= 0))
            rows.append({
                "fiber_id": int(fiber["fiber_index"]),
                "source_segment": segment,
                "start_anchor": start, "end_anchor": end,
                "nearby_tets": nearby_tets,
                "nearby_sections": [],
                **{key: value for key, value in result.items()
                   if not key.startswith("sampled_")}})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    invalid = [row for row in rows
               if row["classification"] != "STRAIGHT_SEGMENT_VALID"]
    write_json(output / "invalid_source_segments.json", {
        "source_segment_count": len(rows),
        "invalid_segment_count": len(invalid), "segments": rows})
    write_chords(output / "invalid_segment_chords.obj", rows)
    crossing_points = []
    for row in invalid:
        for interval in row["outside_intervals"]:
            for alpha in (interval["alpha_first"], interval["alpha_last"]):
                crossing_points.append(
                    (1. - alpha) * np.asarray(row["start_anchor"])
                    + alpha * np.asarray(row["end_anchor"]))
    if crossing_points:
        cloud = np.asarray(crossing_points)
        np.savetxt(
            output / "segment_surface_crossings.ply", cloud,
            header=("ply\nformat ascii 1.0\nelement vertex %d\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "end_header" % len(cloud)), comments="")
    print(output / "invalid_source_segments.json")


if __name__ == "__main__":
    main()
