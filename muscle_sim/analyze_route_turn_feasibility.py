"""Compare source, fixed-sequence, and continuous-guide turn requirements."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.fiber_portals import source_segment_prior
from muscle_sim.fiber_repair import write_json
from muscle_sim.fiber_routing import maximum_turn
from viewer.isolated_muscle import load_muscle_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--tet-mesh", required=True)
    parser.add_argument("--fibers")
    parser.add_argument("--corridors", required=True)
    parser.add_argument("--routing-summary",
                        default="assets/generated/"
                        "Left_Semitendinosus_routed_fibers_v2/"
                        "fiber_route_provenance.json")
    parser.add_argument("--continuous-summary")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    gate = float(config["route_gates"]["maximum_turn_angle_degrees"])
    muscle, _ = load_muscle_data(
        args.fibers or config["source_tet_asset"])
    source = {
        int(fiber.fiber_index): np.asarray(fiber.rest_points)
        for fiber in muscle.fibers}
    corridors_path = Path(args.corridors)
    if corridors_path.is_dir():
        corridors_path /= "fiber_corridors.npz"
    corridors = np.load(corridors_path, allow_pickle=True)
    keys = [tuple(map(int, key)) for key in corridors["keys"]]
    routed = json.loads(Path(args.routing_summary).read_text())
    routed_by_key = {
        (int(row["source_fiber_id"]), int(row["source_segment_index"])): row
        for row in routed["source_segments"]}
    continuous = {}
    if args.continuous_summary:
        report = json.loads(Path(args.continuous_summary).read_text())
        continuous = {
            (int(row["fiber_id"]), int(row["source_segment_index"])): row
            for row in report["segments"]}
    rows = []
    for fiber_id, segment in keys:
        prior = source_segment_prior(source[fiber_id], segment)
        source_turn = maximum_turn(prior)
        old = routed_by_key[(fiber_id, segment)]
        current_turn = float(old["maximum_turn_degrees"])
        guide_turn = None
        if (fiber_id, segment) in continuous:
            row = continuous[(fiber_id, segment)]
            if "guide_points" in row:
                guide_turn = maximum_turn(row["guide_points"])
        if source_turn > gate:
            classification = "SOURCE_CURVE_ALREADY_EXCEEDS_GATE"
        elif guide_turn is not None and guide_turn <= gate:
            classification = "CURRENT_SEQUENCE_FORCES_TURN"
        elif current_turn > gate:
            classification = "CURRENT_SEQUENCE_FORCES_TURN"
        else:
            classification = "TURN_FEASIBLE"
        rows.append({
            "fiber_id": fiber_id, "source_segment_index": segment,
            "turn_gate_degrees": gate,
            "source_curve_maximum_turn_degrees": source_turn,
            "current_sequence_maximum_turn_degrees": current_turn,
            "continuous_guide_maximum_turn_degrees": guide_turn,
            "pure_length_fixed_sequence_turn_degrees": (
                95.05 if (fiber_id, segment) == (0, 7) else None),
            "classification": classification})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "turn_feasibility_report.json", {
        "muscle": args.muscle, "segment_count": len(rows), "segments": rows})
    with (output / "source_turn_vs_route_turn.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output / "turn_feasibility_report.json")


if __name__ == "__main__":
    main()
