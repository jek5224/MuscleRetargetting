"""CLI for B0-B3 hybrid contact-constrained surface fitting."""
import argparse
from pathlib import Path
import yaml

from muscle_sim.hybrid_surface import (
    build_masks_and_context, fit_candidates, write_mask_outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-surface", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    context = build_masks_and_context(
        args.base_surface, args.reference, config)
    write_mask_outputs(context, Path(args.output) / "masks")
    fit_candidates(context, config, args.output)


if __name__ == "__main__":
    main()
