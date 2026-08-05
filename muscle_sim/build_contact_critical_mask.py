"""CLI for hybrid reconstruction masks."""
import argparse
from pathlib import Path
import yaml

from muscle_sim.hybrid_surface import build_masks_and_context, write_mask_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--contact-report")
    parser.add_argument("--base-surface", default=(
        "assets/generated/Left_Semitendinosus_manifold_surface_implicit/"
        "implicit_reconstruction_candidate.obj"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    context = build_masks_and_context(
        args.base_surface, config["selected_reference_asset"], config)
    write_mask_outputs(context, args.output)


if __name__ == "__main__":
    main()
