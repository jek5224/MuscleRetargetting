"""Optimize source-aware shared-face portal routes inside saved corridors."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import trimesh
import yaml

from muscle_sim.fiber_portals import (
    deterministic_candidate_ranking, diverse_portal_sequences,
    optimize_portals, sample_polyline, simplify_tet_path,
    source_segment_prior)
from muscle_sim.fiber_repair import (
    load_repaired_fibers, load_surface, write_json)
from muscle_sim.fiber_routing import (
    ClearanceField, build_tet_adjacency, maximum_turn, route_length,
    validate_exact_route)
from viewer.isolated_muscle import TetLocator, load_muscle_data


def write_obj(path, rows):
    with Path(path).open("w") as handle:
        offset = 1
        for row in rows:
            if "points" not in row:
                continue
            points = np.asarray(row["points"])
            handle.write("g fiber_%d_segment_%d\n" % (
                row["fiber_id"], row["source_segment_index"]))
            for point in points:
                handle.write("v %.17g %.17g %.17g\n" % tuple(point))
            handle.write("l " + " ".join(map(
                str, range(offset, offset + len(points)))) + "\n")
            offset += len(points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--tet-mesh", required=True)
    parser.add_argument("--fibers")
    parser.add_argument("--anchor-candidate", required=True)
    parser.add_argument("--corridors", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--only-segment", help="optional diagnostic filter FIBER:SEGMENT")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    corridor_settings = config["fiber_corridor"]
    optimizer_settings = config["portal_optimizer"]
    gates = config["route_gates"]
    mesh_path = Path(args.tet_mesh)
    if mesh_path.is_dir():
        mesh_path /= "optimized_tet_candidate.npz"
    mesh = np.load(mesh_path)
    vertices, tetrahedra = mesh["vertices"], mesh["tetrahedra"]
    centroids = vertices[tetrahedra].mean(axis=1)
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
    corridor_data = np.load(corridor_path, allow_pickle=True)
    corridor_by_key = {
        tuple(map(int, key)): np.asarray(ids, dtype=np.int32)
        for key, ids in zip(corridor_data["keys"], corridor_data["tet_ids"])}
    surface = load_surface(config["c3_surface"])
    clearance = ClearanceField(surface)
    start_time = time.monotonic()
    total_budget = float(
        optimizer_settings["maximum_total_optimization_seconds"])
    rows = []
    only_key = (tuple(map(int, args.only_segment.split(":")))
                if args.only_segment else None)
    for key in sorted(corridor_by_key):
        if only_key is not None and key != only_key:
            continue
        fiber_id, segment = key
        if time.monotonic() - start_time > total_budget:
            rows.append({
                "fiber_id": fiber_id, "source_segment_index": segment,
                "accepted": False,
                "failure_classification": "RESOURCE_BUDGET_EXHAUSTED"})
            continue
        corridor_ids = corridor_by_key[key]
        if not len(corridor_ids):
            rows.append({
                "fiber_id": fiber_id, "source_segment_index": segment,
                "accepted": False,
                "failure_classification": "SOURCE_CORRIDOR_DISCONNECTED"})
            continue
        anchors = repaired_by_id[fiber_id]
        source = source_by_id[fiber_id]
        start, end = anchors[segment:segment + 2]
        start_tet = locator.locate(
            start, tolerance=2e-8, candidates=512)[0]
        end_tet = locator.locate(
            end, tolerance=2e-8, candidates=512)[0]
        allowed = np.zeros(len(tetrahedra), dtype=bool)
        allowed[corridor_ids] = True
        allowed[start_tet] = allowed[end_tet] = True
        prior = source_segment_prior(source, segment)
        sampled, parameter = sample_polyline(
            prior, max(route_length(prior) / 256., 1e-6))
        nearest = cKDTree(sampled).query(centroids[corridor_ids])[1]
        coordinate = np.zeros(len(tetrahedra))
        coordinate[corridor_ids] = parameter[nearest]
        sampled_tangent = np.gradient(sampled, axis=0)
        sampled_tangent /= np.maximum(
            np.linalg.norm(sampled_tangent, axis=1)[:, None], 1e-30)
        source_tangent = np.zeros_like(centroids)
        source_tangent[corridor_ids] = sampled_tangent[nearest]
        coordinate[start_tet], coordinate[end_tet] = 0., 1.
        source_tangent[start_tet] = sampled_tangent[0]
        source_tangent[end_tet] = sampled_tangent[-1]
        sequence_settings = dict(corridor_settings)
        sequences = diverse_portal_sequences(
            vertices, tetrahedra, neighbors, allowed, coordinate,
            start_tet, end_tet, sequence_settings, source_tangent)
        candidate_rows = []
        for sequence_index, sequence in enumerate(sequences):
            if len(sequence) - 1 > int(
                    optimizer_settings["maximum_portals_per_sequence"]):
                continue
            result = optimize_portals(
                vertices, tetrahedra, sequence, start, end, prior,
                clearance, optimizer_settings)
            if optimizer_settings.get("enable_corridor_shortcuts", True):
                simplified_points, simplified_path = simplify_tet_path(
                    vertices, tetrahedra, neighbors, locator,
                    result.points, sequence, allowed, clearance,
                    float(gates["hard_clearance_m"]), fiber_id, segment,
                    maximum_skip=64)
                if len(simplified_path) < len(sequence):
                    sequence = simplified_path
                    result = optimize_portals(
                        vertices, tetrahedra, sequence, start, end, prior,
                        clearance, optimizer_settings)
            exact = validate_exact_route(
                vertices, tetrahedra, sequence, result.points)
            dense = np.vstack([
                (1. - alpha) * first + alpha * second
                for first, second in zip(
                    result.points[:-1], result.points[1:])
                for alpha in np.linspace(0., 1., 9)])
            exact_clearance = float(np.min(trimesh.proximity.closest_point(
                surface, dense)[1]))
            length = route_length(result.points)
            source_length = route_length(prior)
            length_error = abs(length - source_length) / max(
                source_length, 1e-30)
            turn = maximum_turn(result.points)
            directions = np.diff(result.points, axis=0)
            directions /= np.maximum(
                np.linalg.norm(directions, axis=1)[:, None], 1e-30)
            source_directions = np.diff(prior, axis=0)
            source_directions /= np.maximum(
                np.linalg.norm(source_directions, axis=1)[:, None], 1e-30)
            tangent_errors = np.degrees(np.arccos(np.clip([
                np.dot(directions[0], source_directions[0]),
                np.dot(directions[-1], source_directions[-1])], -1., 1.)))
            hard = {
                "exact_containment": bool(exact["valid"]),
                "route_length": length_error <= float(
                    gates["maximum_route_length_relative_error"]),
                "maximum_turn": turn <= float(
                    gates["maximum_turn_angle_degrees"]),
                "endpoint_tangent": float(np.max(tangent_errors)) <= float(
                    gates["maximum_tangent_change_degrees"]),
                "clearance": exact_clearance >= float(
                    gates["hard_clearance_m"])}
            candidate_rows.append({
                "candidate": f"P{min(sequence_index + 1, 6)}",
                "tet_path": sequence, "points": result.points,
                "initial_portal_points": result.initial_points,
                "portal_barycentric": result.barycentric,
                "source_parameters": result.source_parameters,
                "portal_objective": result.objective,
                "optimizer_iterations": result.iterations,
                "optimizer_success": result.success,
                "route_length_m": length,
                "source_curve_length_m": source_length,
                "route_length_relative_error": length_error,
                "maximum_turn_degrees": turn,
                "endpoint_tangent_error_degrees": tangent_errors,
                "minimum_clearance_m": exact_clearance,
                "exact_containment": exact,
                "hard_gates": hard,
                "passes_hard_gates": all(hard.values()),
                "score": (length_error + turn / 180.
                          + float(np.max(tangent_errors)) / 180.
                          + max(0., float(gates["hard_clearance_m"])
                                - exact_clearance)
                          / float(gates["hard_clearance_m"]))})
        ranked = deterministic_candidate_ranking(candidate_rows)
        if not ranked:
            rows.append({
                "fiber_id": fiber_id, "source_segment_index": segment,
                "accepted": False,
                "failure_classification": "NO_CLEARANCE_FEASIBLE_SEQUENCE"})
            continue
        selected = ranked[0]
        selected.update({
            "fiber_id": fiber_id, "source_segment_index": segment,
            "accepted": selected["passes_hard_gates"],
            "sequence_candidate_count": len(candidate_rows)})
        if not selected["accepted"]:
            failures = []
            if not selected["hard_gates"]["route_length"]:
                failures.append("PORTAL_LENGTH_EXCESSIVE")
            if not selected["hard_gates"]["endpoint_tangent"]:
                failures.append("PORTAL_TANGENT_DISTORTION")
            if not selected["hard_gates"]["maximum_turn"]:
                failures.append("PORTAL_TURN_DISTORTION")
            if not selected["hard_gates"]["clearance"]:
                failures.append("NO_CLEARANCE_FEASIBLE_SEQUENCE")
            selected["failure_classification"] = (
                failures[0] if len(failures) == 1 else
                ("MULTIPLE_CAUSES" if failures else "PORTAL_LOCAL_MINIMUM"))
            selected["contributing_failures"] = failures
        rows.append(selected)
        print(f"portal fiber {fiber_id} segment {segment}: "
              f"{'PASS' if selected['accepted'] else selected['failure_classification']} "
              f"L={selected['route_length_relative_error']:.3f} "
              f"turn={selected['maximum_turn_degrees']:.1f} "
              f"clear={selected['minimum_clearance_m']:.3g}", flush=True)
    accepted = all(row.get("accepted", False) for row in rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "accepted": accepted, "muscle": args.muscle,
        "accepted_tet_asset": str(mesh_path),
        "source_fiber_asset": args.fibers or config["source_tet_asset"],
        "anchor_candidate": str(anchor_path),
        "elapsed_seconds": time.monotonic() - start_time,
        "segment_count": len(rows),
        "accepted_segment_count": sum(
            bool(row.get("accepted")) for row in rows),
        "segments": rows}
    write_json(output / "portal_route_summary.json", report)
    write_json(output / "fiber_route_provenance.json", report)
    write_obj(output / "portal_optimized_segments.obj", rows)
    if not accepted:
        raise ValueError(
            "portal candidate failed geometric gates; no fiber asset published")
    print(output / "portal_route_summary.json")


if __name__ == "__main__":
    main()
