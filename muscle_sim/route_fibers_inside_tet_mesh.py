"""Route sparse Semitendinosus anchor chords through exact tet adjacency."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml

from muscle_sim.fiber_repair import (
    load_repaired_fibers, load_surface, signed_clearance, write_json)
from muscle_sim.fiber_routing import (
    ClearanceField, arc_length_weighted_tet_directions, assemble_fiber,
    build_tet_adjacency,
    classify_straight_segment, embed_routed_fiber, introduced_crossings,
    endpoint_tangent_errors, maximum_turn, route_length,
    route_source_segment, save_routed_fibers, source_curve_prior,
    tet_barycentric)
from muscle_sim.fiber_routing import traverse_straight_segment
from viewer.isolated_muscle import TetLocator, load_muscle_data


def write_polyline_obj(path, routes):
    with Path(path).open("w") as handle:
        offset = 1
        for route in routes:
            handle.write(f"g fiber_{route['fiber_index']}\n")
            for point in route["points"]:
                handle.write("v %.17g %.17g %.17g\n" % tuple(point))
            handle.write("l " + " ".join(map(
                str, range(offset, offset + len(route["points"])))) + "\n")
            offset += len(route["points"])


def write_points_ply(path, points):
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    with Path(path).open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property double x\nproperty double y\n"
                     "property double z\nend_header\n")
        for point in points:
            handle.write("%.17g %.17g %.17g\n" % tuple(point))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--tet-mesh", required=True)
    parser.add_argument("--fibers")
    parser.add_argument("--anchor-candidate", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = dict(config["fiber_path_routing"])
    candidate_path = Path(args.tet_mesh)
    if candidate_path.is_dir():
        candidate_path /= "optimized_tet_candidate.npz"
    data = np.load(candidate_path)
    vertices, tetrahedra = data["vertices"], data["tetrahedra"]
    surface = load_surface(config["c3_surface"])
    inspection = json.loads(Path(config["segment_inspection"]).read_text())
    classification_by_key = {
        (int(row["fiber_id"]), int(row["source_segment"])): row
        for row in inspection["segments"]}
    anchor_candidate_path = Path(args.anchor_candidate)
    if anchor_candidate_path.is_dir():
        anchor_candidate_path /= "repaired_simulation_fibers.npz"
    anchor_rows, _ = load_repaired_fibers(anchor_candidate_path)
    source_muscle, _ = load_muscle_data(
        args.fibers or config["source_tet_asset"])
    source_by_id = {
        fiber.fiber_index: fiber for fiber in source_muscle.fibers}
    repair_mapping_path = Path(args.anchor_candidate)
    if repair_mapping_path.is_file():
        repair_mapping_path = repair_mapping_path.parent
    repair_mapping = json.loads(
        (repair_mapping_path / "fiber_repair_mapping.json").read_text())
    repair_mapping_by_key = {
        (int(row["fiber_id"]), int(row["sample_id"])): row
        for row in repair_mapping}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    neighbors, neighbor_local = build_tet_adjacency(tetrahedra)
    np.savez_compressed(
        output / "tet_adjacency.npz", neighbors=neighbors,
        neighbor_local_faces=neighbor_local)
    centroids = vertices[tetrahedra].mean(axis=1)
    settings["_tet_centroids"] = centroids
    locator = TetLocator(vertices, tetrahedra)
    clearance = ClearanceField(surface)
    routes, embedded, segment_rows, anchor_mapping = [], [], [], []
    inserted_mapping, face_points = [], []
    clearance_samples = []
    candidate_stages = {
        name: {"candidate": name, "accepted": False, "segment_count": 0}
        for name in ("R0", "R1", "R2", "R3", "R4", "R5")}
    for anchor_row in anchor_rows:
        fiber_id = int(anchor_row["fiber_index"])
        print(f"routing fiber {fiber_id}", flush=True)
        anchors = np.asarray(anchor_row["points"])
        original = np.asarray(source_by_id[fiber_id].rest_points)
        prior = source_curve_prior(original, 32)
        per_segment = []
        for segment_id, (start, end) in enumerate(zip(
                anchors[:-1], anchors[1:])):
            classification = classification_by_key[(fiber_id, segment_id)]
            if classification["classification"] == "STRAIGHT_SEGMENT_VALID":
                try:
                    routed, crossings, validation = traverse_straight_segment(
                        vertices, tetrahedra, neighbors, locator, start, end,
                        fiber_id, segment_id)
                except ValueError:
                    # Degenerate edge/vertex crossings can make the analytic
                    # walk nonunique; exact face-A* is the deterministic
                    # fallback, not a surface-adjacency shortcut.
                    routed, crossings, validation = route_source_segment(
                        vertices, tetrahedra, neighbors, locator, clearance,
                        start, end, fiber_id, segment_id,
                        classification["classification"], settings, "R5")
            else:
                routed, crossings, validation = route_source_segment(
                    vertices, tetrahedra, neighbors, locator, clearance,
                    start, end, fiber_id, segment_id,
                    classification["classification"], settings, "R5")
            per_segment.append(routed)
            source_length = float(np.linalg.norm(end - start))
            prior_segment = prior[segment_id * 32:(segment_id + 1) * 32 + 1]
            source_curve_length = route_length(prior_segment)
            routed_length = route_length(routed.points)
            sampled = np.vstack([
                (1. - alpha) * routed.points[index]
                + alpha * routed.points[index + 1]
                for index in range(len(routed.points) - 1)
                for alpha in np.linspace(0., 1., 3)])
            clearance_first = len(clearance_samples)
            clearance_samples.extend(sampled)
            clearance_last = len(clearance_samples)
            row = {
                "source_fiber_id": fiber_id,
                "source_segment_index": segment_id,
                "classification": classification["classification"],
                "start_anchor_position": start,
                "end_anchor_position": end,
                "straight_chord_containment": classification,
                "selected_tet_path": routed.tet_path,
                "shared_face_crossings": crossings,
                "inserted_points": routed.points[1:-1],
                "inserted_sample_count": len(routed.points) - 2,
                "mandatory_shared_face_crossing_count": len(crossings),
                "route_length_m": routed_length,
                "source_length_m": source_length,
                "source_curve_prior_length_m": source_curve_length,
                "route_length_relative_error": abs(
                    routed_length - source_curve_length) / max(
                        source_curve_length, 1e-30),
                "_clearance_slice": [clearance_first, clearance_last],
                "maximum_turn_degrees": maximum_turn(routed.points),
                "endpoint_tangent_error_degrees": endpoint_tangent_errors(
                    routed.points, original, segment_id),
                "exact_containment": validation,
                "optimization_result": {
                    "method": "deterministic_projected_shared_face",
                    "endpoints_fixed": True,
                    "iterations": 30},
                "selected_candidate": "R5",
            }
            segment_rows.append(row)
            for point_index, (point, owner, parameter) in enumerate(zip(
                    routed.points[1:-1], routed.owner_tets[:-1],
                    routed.source_parameters[1:-1]), start=1):
                inserted_mapping.append({
                    "parent_fiber_id": fiber_id,
                    "source_segment_index": segment_id,
                    "normalized_parameter_along_source_segment": parameter,
                    "generation_method": "TET_FACE_ASTAR_R5",
                    "containing_tet": int(owner),
                    "barycentric_coordinates": tet_barycentric(
                        vertices, tetrahedra, int(owner), point),
                    "simulation_point_index_within_source_route": point_index,
                    "sample_role": "INSERTED_INTERIOR_SAMPLE"})
            face_points.extend([
                np.asarray(row["face_barycentric"]) @ vertices[
                    np.asarray(row["face_vertex_ids"], dtype=np.int32)]
                for row in crossings])
            for name in candidate_stages:
                candidate_stages[name]["segment_count"] += 1
        assembled = assemble_fiber(fiber_id, anchors, per_segment)
        routed_fiber, embedding = embed_routed_fiber(
            assembled, vertices, tetrahedra)
        assembled["embedding"] = embedding
        routes.append(assembled)
        embedded.append(routed_fiber)
        for index, (source, repaired) in enumerate(zip(original, anchors)):
            repair_row = repair_mapping_by_key[(fiber_id, index)]
            anchor_mapping.append({
                "fiber_id": fiber_id, "original_sample_index": index,
                "original_position": source,
                "repaired_anchor_position": repaired,
                "anchor_displacement_m": float(np.linalg.norm(
                    repaired - source)),
                "sample_role": ("REPAIRED_ORIGINAL_ANCHOR"
                                if not np.array_equal(source, repaired)
                                else "ORIGINAL_ANCHOR"),
                "source_semantic_role": repair_row["sample_role"],
                "anchor_repair_classification": repair_row["classification"],
                "anchor_repair_policy": repair_row["repair_policy"]})
    del settings["_tet_centroids"]
    print(f"clearance samples {len(clearance_samples)}", flush=True)
    clearance_array = np.asarray(clearance_samples)
    all_clearance = np.empty(len(clearance_array))
    for first in range(0, len(clearance_array), 4096):
        last = min(first + 4096, len(clearance_array))
        # Exact tet ownership already proves the points are inside.  Clearance
        # therefore needs only exact unsigned boundary distance, avoiding a
        # redundant winding-number query for 250k certified-interior samples.
        all_clearance[first:last] = trimesh.proximity.closest_point(
            surface, clearance_array[first:last])[1]
    for row in segment_rows:
        first, last = row.pop("_clearance_slice")
        row["minimum_clearance_m"] = float(np.min(
            all_clearance[first:last]))
    near_crossings = introduced_crossings(
        routes, float(settings["minimum_fiber_spacing_m"]))
    maximum_anchor = max(
        row["anchor_displacement_m"] for row in anchor_mapping)
    maximum_route_error = max(
        row["route_length_relative_error"] for row in segment_rows)
    minimum_clearance = min(
        row["minimum_clearance_m"] for row in segment_rows)
    maximum_turn_degrees = max(
        row["maximum_turn_degrees"] for row in segment_rows)
    maximum_inserted = max(
        row["inserted_sample_count"] for row in segment_rows)
    gates = {
        "maximum_anchor_displacement": maximum_anchor <= float(
            settings["maximum_anchor_displacement_m"]),
        "route_length": maximum_route_error <= float(
            settings["maximum_route_length_relative_error"]),
        "turn_angle": maximum_turn_degrees <= float(
            settings["maximum_turn_angle_degrees"]),
        "inserted_sample_cap": maximum_inserted <= int(
            settings["maximum_inserted_samples_per_source_segment"]),
        "exact_tet_containment": all(
            row["exact_containment"]["valid"] for row in segment_rows),
        "clearance": minimum_clearance >= float(
            settings["interior_margin_m"]),
        "no_introduced_near_crossings": not near_crossings,
        "embedding": all(
            row["embedding"]["maximum_reconstruction_error_m"] <= 1e-10
            and row["embedding"]["minimum_barycentric_weight"] >= -2e-9
            for row in routes),
    }
    accepted = all(gates.values())
    failure_classifications = []
    if not gates["route_length"]:
        failure_classifications.append("ROUTE_LENGTH_EXCESSIVE")
    if not gates["turn_angle"]:
        failure_classifications.append("ROUTE_CURVATURE_DISTORTION")
    if not gates["clearance"]:
        failure_classifications.append("CLEARANCE_FIELD_BLOCKS_ROUTE")
    if not gates["no_introduced_near_crossings"]:
        failure_classifications.append("BUNDLE_ORDERING_FAILURE")
    if max(max(row["endpoint_tangent_error_degrees"])
           for row in segment_rows) > float(
               settings["maximum_tangent_change_degrees"]):
        failure_classifications.append("ROUTE_TANGENT_DISTORTION")
    for name in candidate_stages:
        stage = candidate_stages[name]
        stage["description"] = {
            "R0": "original anchors and straight-chord diagnostics",
            "R1": "deterministic tet-graph routes",
            "R2": "shared-face source projection",
            "R3": "source-deviation and tangent-weighted A*",
            "R4": "bundle crossing/spacing validation",
            "R5": "adaptive within-tet refinement",
        }[name]
        stage["accepted"] = bool(accepted and name == "R5")
    provenance = {
        "accepted": accepted, "selected_candidate": "R5" if accepted else None,
        "source_fiber_asset": args.fibers or config["source_tet_asset"],
        "anchor_candidate": args.anchor_candidate,
        "accepted_tet_candidate": str(candidate_path),
        "configuration": settings, "candidate_stages": candidate_stages,
        "anchor_mapping": anchor_mapping,
        "inserted_sample_mapping": inserted_mapping,
        "source_segments": segment_rows,
        "near_crossings": near_crossings, "gates": gates,
        "failure_classifications": failure_classifications,
        "summary": {
            "fiber_count": len(routes),
            "source_segment_count": len(segment_rows),
            "inserted_sample_count": int(sum(
                len(route["points"]) - len(route["source_anchor_points"])
                for route in routes)),
            "maximum_anchor_displacement_m": maximum_anchor,
            "maximum_route_length_relative_error": maximum_route_error,
            "minimum_clearance_m": minimum_clearance,
            "maximum_turn_degrees": maximum_turn_degrees,
            "maximum_inserted_samples_per_source_segment": maximum_inserted,
        }}
    write_json(output / "fiber_route_summary.json", {
        "accepted": accepted, "selected_candidate": provenance[
            "selected_candidate"],
        "candidate_stages": candidate_stages, "gates": gates,
        "summary": provenance["summary"],
        "near_crossing_count": len(near_crossings)})
    write_json(output / "fiber_route_provenance.json", provenance)
    write_polyline_obj(output / "embedded_routed_fibers.obj", routes)
    write_polyline_obj(output / "repaired_anchor_polylines.obj", [
        {"fiber_index": row["fiber_index"],
         "points": row["source_anchor_points"]} for row in routes])
    write_polyline_obj(output / "original_anchor_polylines.obj", [
        {"fiber_index": fiber.fiber_index, "points": fiber.rest_points}
        for fiber in source_muscle.fibers])
    write_points_ply(
        output / "inserted_samples.ply",
        [row["points"][index] for row in routes
         for index, role in enumerate(row["roles"])
         if role == "INSERTED_INTERIOR_SAMPLE"])
    write_points_ply(output / "tet_face_crossing_points.ply", face_points)
    anisotropy, covered_length = arc_length_weighted_tet_directions(
        routes, vertices, tetrahedra)
    np.savez_compressed(
        output / "routed_tet_anisotropy.npz",
        directions=anisotropy, covered_arc_length_m=covered_length)
    with (output / "fiber_clearance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "fiber_id", "minimum_clearance_m", "inserted_samples",
            "traversed_tets", "embedded_route_length_m",
            "source_polyline_length_m"])
        writer.writeheader()
        for route in routes:
            rows = [row for row in segment_rows
                    if row["source_fiber_id"] == route["fiber_index"]]
            writer.writerow({
                "fiber_id": route["fiber_index"],
                "minimum_clearance_m": min(
                    row["minimum_clearance_m"] for row in rows),
                "inserted_samples": len(route["points"])
                - len(route["source_anchor_points"]),
                "traversed_tets": len(set(route[
                    "subsegment_owner_tets"].tolist())),
                "embedded_route_length_m": route["embedded_route_length"],
                "source_polyline_length_m": route["source_polyline_length"]})
    if not accepted:
        raise ValueError("no routed fiber candidate passed all hard gates")
    save_routed_fibers(
        output / "routed_simulation_fibers.npz",
        routes, embedded, provenance)
    print(output / "routed_simulation_fibers.npz")


if __name__ == "__main__":
    main()
