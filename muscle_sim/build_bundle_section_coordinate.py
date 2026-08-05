"""CLI for fiber-bundle longitudinal coordinates and RMF frames."""
import argparse
import json
from pathlib import Path

import yaml

from muscle_sim.cross_sections import (
    assign_bundle_coordinates, representative_bundle_path,
    rotation_minimizing_frames)
from muscle_sim.local_remeshing import json_ready, load_raw_mesh
from viewer.isolated_muscle import load_muscle_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    muscle, _ = load_muscle_data(config["current_tet_asset"])
    path = representative_bundle_path(muscle.fibers)
    frames = rotation_minimizing_frames(path)
    reference = load_raw_mesh(args.reference)
    coordinates = assign_bundle_coordinates(
        reference["vertices"], frames)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "bundle_longitudinal_coordinate.json").write_text(json.dumps(
        json_ready({
            "path": path, "frames": frames,
            "reference_surface_coordinates": coordinates,
        }), indent=2))


if __name__ == "__main__":
    main()
