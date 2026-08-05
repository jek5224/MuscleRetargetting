"""Reconstruct a manifold simulation surface from the chosen reference."""
import argparse
from pathlib import Path

import yaml

from muscle_sim.full_surface import reconstruct_from_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    reconstruct_from_reference(
        config["selected_reference_asset"], config["current_tet_asset"],
        config["diagnostic_dir"], args.output, config)
    print(Path(args.output) / "repaired_simulation_surface.obj")


if __name__ == "__main__":
    main()
