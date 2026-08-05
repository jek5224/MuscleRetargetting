"""Inspect and rank complete Semitendinosus surface representations."""
import argparse
import json
from pathlib import Path

import pickle
import yaml

from muscle_sim.full_surface import (
    classify_loops, complete_topology, inspect_candidates)
from muscle_sim.local_remeshing import load_raw_mesh
from viewer.isolated_muscle import (
    extract_boundary_faces, load_muscle_data, orient_tetrahedra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = inspect_candidates(
        config["reference_candidates"], config["current_tet_asset"])
    (output / "reference_candidate_ranking.json").write_text(json.dumps(
        rows, indent=2))
    selected = Path(rows[0]["path"])
    raw = load_raw_mesh(selected)
    vertices, faces = raw["vertices"], raw["render_faces"]
    topology = complete_topology(vertices, faces)
    (output / "reference_surface_topology.json").write_text(json.dumps(
        topology, indent=2))
    current = load_raw_mesh(config["current_tet_asset"])
    current_tets = orient_tetrahedra(
        current["vertices"], current["tetrahedra"])
    current_faces = extract_boundary_faces(current_tets)
    current_topology = complete_topology(current["vertices"], current_faces)
    try:
        current_muscle, _ = load_muscle_data(config["current_tet_asset"])
        loops = classify_loops(
            current["vertices"], current_topology["boundary_loops"],
            current_muscle.attachment_patches, current_muscle.fibers)
    except ValueError:
        loops = classify_loops(
            current["vertices"], current_topology["boundary_loops"])
    current_topology["classified_loops"] = loops
    (output / "current_tet_boundary_topology.json").write_text(json.dumps(
        current_topology, indent=2))
    print(rows[0]["path"])


if __name__ == "__main__":
    main()
