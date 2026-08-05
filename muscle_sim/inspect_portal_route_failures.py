"""Diagnose whether rejected routes failed by sequence or portal geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.fiber_repair import load_surface, signed_clearance, write_json


def write_obj(path, rows, fields):
    with Path(path).open("w") as handle:
        offset = 1
        for row in rows:
            handle.write("g fiber_%d_segment_%d\n" % (
                row["source_fiber_id"], row["source_segment_index"]))
            points = np.vstack([np.asarray(row[field]).reshape((-1, 3))
                                for field in fields])
            for point in points:
                handle.write("v %.17g %.17g %.17g\n" % tuple(point))
            handle.write("l " + " ".join(map(
                str, range(offset, offset + len(points)))) + "\n")
            offset += len(points)


def write_ply(path, points):
    points = np.asarray(points).reshape((-1, 3))
    with Path(path).open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property double x\nproperty double y\n"
                     "property double z\nend_header\n")
        for point in points:
            handle.write("%.17g %.17g %.17g\n" % tuple(point))


def classify(row, near, settings, endpoint_clearance):
    causes = []
    if row["route_length_relative_error"] > float(
            settings["maximum_route_length_relative_error"]):
        if len(row["selected_tet_path"]) > int(
                settings["bad_sequence_tet_threshold"]):
            causes.append("BAD_TET_SEQUENCE")
        else:
            causes.append("GOOD_SEQUENCE_BAD_PORTAL_POINTS")
    if row["minimum_clearance_m"] < float(settings["hard_clearance_m"]):
        causes.append("CLEARANCE_CONSTRAINED")
    key = (row["source_fiber_id"], row["source_segment_index"])
    if key in near:
        causes.append("BUNDLE_CONFLICT")
    if min(endpoint_clearance) < float(
               settings["anchor_boundary_diagnostic_m"]):
        causes.append("ANCHOR_NEAR_BOUNDARY")
    if not causes:
        causes.append("GOOD_SEQUENCE_BAD_PORTAL_POINTS")
    return causes[0] if len(causes) == 1 else "MULTIPLE_CAUSES", causes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--routing-summary", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["failure_analysis"]
    summary = Path(args.routing_summary)
    provenance_path = summary.with_name("fiber_route_provenance.json")
    data = json.loads(provenance_path.read_text())
    endpoints = np.asarray([
        point for row in data["source_segments"]
        for point in (row["start_anchor_position"], row["end_anchor_position"])])
    endpoint_clearance = signed_clearance(
        load_surface(config["c3_surface"]), endpoints)[0].reshape((-1, 2))
    near = set()
    for crossing in data["near_crossings"]:
        for label in ("first", "second"):
            fiber, simulation_segment = crossing[label]
            near.add((int(fiber), int(simulation_segment)))
    rows = []
    for source_index, source in enumerate(data["source_segments"]):
        failed = (
            source["route_length_relative_error"] > float(
                settings["maximum_route_length_relative_error"])
            or source["maximum_turn_degrees"] > float(
                settings["maximum_turn_degrees"])
            or source["minimum_clearance_m"] < float(
                settings["hard_clearance_m"]))
        if not failed:
            continue
        classification, causes = classify(
            source, near, settings, endpoint_clearance[source_index])
        optimized = np.vstack((
            source["start_anchor_position"], source["inserted_points"],
            source["end_anchor_position"]))
        crossings = source["shared_face_crossings"]
        initial = [
            np.asarray(row.get("initial_position", optimized[index + 1]))
            for index, row in enumerate(crossings)]
        deviation = np.linalg.norm(
            optimized - np.linspace(
                optimized[0], optimized[-1], len(optimized)), axis=1)
        rows.append({
            "source_fiber_id": source["source_fiber_id"],
            "source_segment_index": source["source_segment_index"],
            "original_anchor_endpoints": [
                source["start_anchor_position"], source["end_anchor_position"]],
            "selected_tet_sequence": source["selected_tet_path"],
            "selected_shared_face_sequence": [
                row["face_vertex_ids"] for row in crossings],
            "initial_face_crossing_points": initial,
            "initial_points_available": all(
                "initial_position" in row for row in crossings),
            "optimized_face_crossing_points": optimized[1:-1],
            "route_length_m": source["route_length_m"],
            "source_curve_estimate_length_m": source.get(
                "source_curve_prior_length_m", source["source_length_m"]),
            "maximum_turn_angle_degrees": source["maximum_turn_degrees"],
            "minimum_clearance_m": source["minimum_clearance_m"],
            "near_crossing_participation": (
                (source["source_fiber_id"], source["source_segment_index"])
                in near),
            "number_of_tets_traversed": len(source["selected_tet_path"]),
            "number_of_face_transitions": len(crossings),
            "maximum_deviation_from_source_curve_m": float(np.max(deviation)),
            "endpoint_tangent_error_degrees": source.get(
                "endpoint_tangent_error_degrees"),
            "failure_classification": classification,
            "contributing_causes": causes})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "portal_route_failure_report.json", {
        "muscle": args.muscle, "rejected_route_count": len(rows),
        "note": ("The v2 provenance overwrote initial portal locations; "
                 "initial_points_available marks this explicitly."),
        "routes": rows})
    write_obj(output / "bad_tet_sequences.obj", rows, [
        "original_anchor_endpoints"])
    write_ply(output / "bad_portal_points.ply", [
        point for row in rows for point in row[
            "optimized_face_crossing_points"]])
    with (output / "source_vs_route_curves.obj").open("w") as handle:
        offset = 1
        for row in rows:
            for label, points in (
                    ("source", row["original_anchor_endpoints"]),
                    ("route", [row["original_anchor_endpoints"][0]]
                     + row["optimized_face_crossing_points"]
                     + [row["original_anchor_endpoints"][1]])):
                handle.write("g %s_fiber_%d_segment_%d\n" % (
                    label, row["source_fiber_id"],
                    row["source_segment_index"]))
                for point in points:
                    handle.write("v %.17g %.17g %.17g\n" % tuple(point))
                handle.write("l " + " ".join(map(
                    str, range(offset, offset + len(points)))) + "\n")
                offset += len(points)
    print(output / "portal_route_failure_report.json")


if __name__ == "__main__":
    main()
