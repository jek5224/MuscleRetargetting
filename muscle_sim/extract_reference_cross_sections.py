"""CLI for reference and Path B fiber-normal cross sections."""
import argparse
import json
from pathlib import Path

import trimesh
import yaml

from muscle_sim.cross_sections import (
    build_section_data, representative_bundle_path,
    rotation_minimizing_frames)
from muscle_sim.local_remeshing import json_ready, load_raw_mesh
from viewer.isolated_muscle import load_muscle_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-surface", default=(
        "assets/generated/Left_Semitendinosus_manifold_surface_implicit/"
        "implicit_reconstruction_candidate.obj"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    muscle, _ = load_muscle_data(config["current_tet_asset"])
    frames = rotation_minimizing_frames(
        representative_bundle_path(muscle.fibers))
    raw = load_raw_mesh(config["selected_reference_asset"])
    reference = trimesh.Trimesh(
        raw["vertices"], raw["render_faces"], process=False)
    current = trimesh.load(args.base_surface, force="mesh", process=False)
    sections = build_section_data(
        reference, current, frames, config["cross_section_constraints"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference_sections.json").write_text(json.dumps(
        json_ready(sections), indent=2))


if __name__ == "__main__":
    main()
