"""Generate refined-navigation guide paths and exact portal sequences."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import trimesh
import yaml

from muscle_sim.fiber_continuous import (
    build_navigation_domain, directional_navigation_search,
    extract_face_fans, guide_to_contracted_route, guide_to_tet_sequence,
    maximum_physical_turn)
from muscle_sim.fiber_portals import optimize_portals, source_segment_prior
from muscle_sim.fiber_portals import source_corridor
from muscle_sim.fiber_repair import (
    load_repaired_fibers, load_surface, polyline_tangents, write_json)
from muscle_sim.fiber_routing import (
    ClearanceField, build_tet_adjacency, maximum_turn, route_length,
    validate_exact_route)
from viewer.isolated_muscle import TetLocator, load_muscle_data


def write_obj(path, rows, field):
    with Path(path).open("w") as handle:
        offset = 1
        for row in rows:
            if field not in row:
                continue
            points = np.asarray(row[field])
            handle.write("g fiber_%d_segment_%d\n" % (
                row["fiber_id"], row["source_segment_index"]))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--tet-mesh", required=True)
    parser.add_argument("--fibers")
    parser.add_argument("--anchor-candidate", required=True)
    parser.add_argument("--corridors", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only-segment", help="optional FIBER:SEGMENT")
    parser.add_argument("--corridor-radius-m", type=float)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["continuous_sequence"]
    gates = config["route_gates"]
    portal_config = yaml.safe_load(Path(
        config["portal_optimizer_config"]).read_text())["portal_optimizer"]
    mesh_path = Path(args.tet_mesh)
    if mesh_path.is_dir():
        mesh_path /= "optimized_tet_candidate.npz"
    mesh = np.load(mesh_path)
    vertices, tetrahedra = mesh["vertices"], mesh["tetrahedra"]
    neighbors, _ = build_tet_adjacency(tetrahedra)
    locator = TetLocator(vertices, tetrahedra)
    anchor_path = Path(args.anchor_candidate)
    if anchor_path.is_dir():
        anchor_path /= "repaired_simulation_fibers.npz"
    repaired, _ = load_repaired_fibers(anchor_path)
    repaired_by_id = {
        int(row["fiber_index"]): np.asarray(row["points"]) for row in repaired}
    muscle, _ = load_muscle_data(
        args.fibers or config["source_tet_asset"])
    source_by_id = {
        int(fiber.fiber_index): np.asarray(fiber.rest_points)
        for fiber in muscle.fibers}
    corridor_path = Path(args.corridors)
    if corridor_path.is_dir():
        corridor_path /= "fiber_corridors.npz"
    corridors = np.load(corridor_path, allow_pickle=True)
    corridor_by_key = {
        tuple(map(int, key)): np.asarray(ids, dtype=np.int32)
        for key, ids in zip(corridors["keys"], corridors["tet_ids"])}
    only = (tuple(map(int, args.only_segment.split(":")))
            if args.only_segment else None)
    surface = load_surface(config["c3_surface"])
    clearance = ClearanceField(surface)
    rows, all_nodes = [], []
    for key in sorted(corridor_by_key):
        if only is not None and key != only:
            continue
        started = time.monotonic()
        fiber_id, segment = key
        corridor_tets = corridor_by_key[key]
        anchors, source = repaired_by_id[fiber_id], source_by_id[fiber_id]
        start, end = anchors[segment:segment + 2]
        start_tet = locator.locate(
            start, tolerance=2e-8, candidates=512)[0]
        end_tet = locator.locate(
            end, tolerance=2e-8, candidates=512)[0]
        prior = source_segment_prior(source, segment)
        if args.corridor_radius_m is not None:
            tet_clearance = clearance.approximate(
                vertices[tetrahedra].mean(axis=1))
            expanded, _ = source_corridor(
                vertices, tetrahedra, prior, tet_clearance,
                float(args.corridor_radius_m),
                float(settings["hard_clearance_m"]), start_tet, end_tet)
            corridor_tets = np.flatnonzero(expanded)
        try:
            domain = build_navigation_domain(
                vertices, tetrahedra, corridor_tets, prior, start, end,
                start_tet, end_tet, clearance.approximate, settings)
            nodes, edge_owners, guide_cost = directional_navigation_search(
                domain, settings)
            guide = domain.points[nodes]
            sequence = guide_to_tet_sequence(
                tetrahedra, neighbors, guide, edge_owners)
            portal = optimize_portals(
                vertices, tetrahedra, sequence, start, end, prior,
                clearance, portal_config)
            contracted_points, contracted_sequence, fan_events = (
                guide_to_contracted_route(
                    tetrahedra, neighbors, guide, edge_owners))
            contracted_exact = validate_exact_route(
                vertices, tetrahedra, contracted_sequence,
                contracted_points)
            portal_turn = maximum_turn(portal.points)
            contracted_turn = maximum_physical_turn(contracted_points)
            if contracted_exact["valid"] and contracted_turn < portal_turn:
                selected_points = contracted_points
                selected_sequence = contracted_sequence
                selected_candidate = "S5_CONTRACTED_GUIDE"
                selected_barycentric = []
            else:
                selected_points = portal.points
                selected_sequence = sequence
                selected_candidate = "S4_ANALYTICAL_PORTALS"
                selected_barycentric = portal.barycentric
            exact = validate_exact_route(
                vertices, tetrahedra, selected_sequence, selected_points)
            dense = np.vstack([
                (1. - alpha) * first + alpha * second
                for first, second in zip(
                    selected_points[:-1], selected_points[1:])
                for alpha in np.linspace(0., 1., 9)])
            distances = trimesh.proximity.closest_point(surface, dense)[1]
            minimum_index = int(np.argmin(distances))
            minimum_clearance = float(distances[minimum_index])
            length = route_length(selected_points)
            source_length = route_length(prior)
            length_error = abs(length - source_length) / max(
                source_length, 1e-30)
            turn = maximum_physical_turn(selected_points)
            route_directions = np.diff(selected_points, axis=0)
            physical = np.linalg.norm(route_directions, axis=1) > 1e-12
            route_directions = route_directions[physical]
            route_directions /= np.maximum(
                np.linalg.norm(route_directions, axis=1)[:, None], 1e-30)
            source_tangent = polyline_tangents(prior)
            tangent_error = np.degrees(np.arccos(np.clip([
                np.dot(route_directions[0], source_tangent[0]),
                np.dot(route_directions[-1], source_tangent[-1])], -1., 1.)))
            hard = {
                "exact_containment": bool(exact["valid"]),
                "route_length": length_error <= float(
                    gates["maximum_route_length_relative_error"]),
                "endpoint_tangent": float(np.max(tangent_error)) <= float(
                    gates["maximum_tangent_change_degrees"]),
                "maximum_turn": turn <= float(
                    gates["maximum_turn_angle_degrees"]),
                "clearance": minimum_clearance >= float(
                    gates["hard_clearance_m"]),
                "source_monotonicity": bool(np.min(np.diff(
                    domain.source_parameter[nodes]))
                    >= -float(settings["backward_parameter_tolerance"]))}
            row = {
                "fiber_id": fiber_id, "source_segment_index": segment,
                "accepted": all(hard.values()), "hard_gates": hard,
                "navigation_node_count": len(domain.points),
                "guide_node_count": len(nodes),
                "guide_cost": guide_cost, "guide_points": guide,
                "guide_edge_owner_tets": edge_owners,
                "selected_candidate": selected_candidate,
                "selected_tet_sequence": selected_sequence,
                "face_fans": extract_face_fans(
                    tetrahedra, selected_sequence,
                    int(settings["maximum_face_fan_size"])),
                "face_fan_contractions": fan_events,
                "uncontracted_portal_turn_degrees": portal_turn,
                "optimized_points": selected_points,
                "portal_barycentric": selected_barycentric,
                "route_length_m": length,
                "source_curve_length_m": source_length,
                "route_length_relative_error": length_error,
                "maximum_turn_degrees": turn,
                "endpoint_tangent_error_degrees": tangent_error,
                "minimum_clearance_m": minimum_clearance,
                "minimum_clearance_point": dense[minimum_index],
                "optimizer_success": portal.success,
                "optimizer_iterations": portal.iterations,
                "elapsed_seconds": time.monotonic() - started}
            if not row["accepted"]:
                if not hard["maximum_turn"]:
                    row["failure_classification"] = (
                        "GUIDE_TO_TET_SEQUENCE_DISTORTION")
                elif not hard["clearance"]:
                    row["failure_classification"] = (
                        "NO_CLEARANCE_AND_TURN_FEASIBLE_PATH")
                else:
                    row["failure_classification"] = (
                        "NO_CONTINUOUS_TURN_FEASIBLE_PATH")
            rows.append(row)
            all_nodes.extend(domain.points)
            print(f"continuous fiber {fiber_id} segment {segment}: "
                  f"{'PASS' if row['accepted'] else row['failure_classification']} "
                  f"L={length_error:.3f} turn={turn:.1f} "
                  f"clear={minimum_clearance:.3g}", flush=True)
        except ValueError as error:
            rows.append({
                "fiber_id": fiber_id, "source_segment_index": segment,
                "accepted": False, "failure_classification": str(error),
                "elapsed_seconds": time.monotonic() - started})
            print(f"continuous fiber {fiber_id} segment {segment}: {error}",
                  flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "accepted": bool(rows) and all(row["accepted"] for row in rows),
        "muscle": args.muscle, "segment_count": len(rows),
        "accepted_segment_count": sum(row["accepted"] for row in rows),
        "segments": rows}
    write_json(output / "continuous_sequence_summary.json", report)
    write_obj(output / "continuous_guide_paths.obj", rows, "guide_points")
    write_obj(output / "optimized_portal_routes.obj", rows, "optimized_points")
    write_ply(output / "refined_navigation_nodes.ply", all_nodes)
    if only == (0, 7):
        write_json(output / "representative_0_7_comparison.json", {
            "previous_anatomical_objective_turn_degrees": 74.93,
            "previous_pure_length_turn_degrees": 95.05,
            "new_result": rows[0]})
    if not report["accepted"]:
        raise ValueError(
            "continuous sequence candidate failed; no fiber asset published")
    print(output / "continuous_sequence_summary.json")


if __name__ == "__main__":
    main()
