"""Enumerate bounded alternative tet paths for extracted face fans."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.fiber_continuous import (
    enumerate_local_tet_paths, extract_face_fans)
from muscle_sim.fiber_repair import write_json
from muscle_sim.fiber_routing import build_tet_adjacency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tet-mesh", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["face_fan_contraction"]
    mesh_path = Path(args.tet_mesh)
    if mesh_path.is_dir():
        mesh_path /= "optimized_tet_candidate.npz"
    mesh = np.load(mesh_path)
    tetrahedra = mesh["tetrahedra"]
    neighbors, _ = build_tet_adjacency(tetrahedra)
    sequence_path = Path(args.sequences)
    if sequence_path.is_dir():
        sequence_path /= "continuous_sequence_summary.json"
    data = json.loads(sequence_path.read_text())
    rows = []
    for source in data["segments"]:
        if "selected_tet_sequence" not in source:
            continue
        path = list(map(int, source["selected_tet_sequence"]))
        fans = extract_face_fans(
            tetrahedra, path, int(settings["maximum_fan_tets"]))
        for index, fan in enumerate(fans):
            first, last = (
                fan["first_tet_index"], fan["last_tet_index"])
            allowed = path[first:last + 1]
            alternatives = enumerate_local_tet_paths(
                neighbors, allowed, path[first], path[last],
                int(settings["maximum_alternative_paths"]),
                int(settings["maximum_fan_tets"]))
            rows.append({
                "fiber_id": source["fiber_id"],
                "source_segment_index": source["source_segment_index"],
                "fan_index": index, **fan,
                "original_path": allowed,
                "alternative_paths": alternatives,
                "shorter_alternative_exists": any(
                    len(value) < len(allowed) for value in alternatives)})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "face_fan_contraction_report.json", {
        "fan_count": len(rows), "fans": rows})
    with (output / "sequence_candidate_ranking.csv").open(
            "w", newline="") as handle:
        fields = ["fiber_id", "source_segment_index", "fan_index",
                  "shorter_alternative_exists"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(output / "face_fan_contraction_report.json")


if __name__ == "__main__":
    main()
