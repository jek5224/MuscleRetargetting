"""Explicit CLI for Path B implicit full-surface reconstruction."""
import argparse
from pathlib import Path

import yaml

from muscle_sim.full_surface import implicit_voxel_reconstruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    implicit_voxel_reconstruction(
        config["selected_reference_asset"], args.output, config)


if __name__ == "__main__":
    main()
