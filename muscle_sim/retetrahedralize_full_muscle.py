"""CLI for full manifold-surface tetrahedralization."""
import argparse
import json
from pathlib import Path

import yaml

from muscle_sim.full_retetrahedralization import build_derived_asset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    try:
        asset, _ = build_derived_asset(args.surface, config, args.output)
    except ValueError as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "retetrahedralization_failure.json").write_text(
            json.dumps({
                "accepted": False,
                "failure_classification":
                    "THIN_REGION_TET_QUALITY_FAILURE",
                "reason": str(error),
                "surface": args.surface,
                "hard_minimum_dihedral_degrees":
                    config["quality_thresholds"][
                        "hard_minimum_dihedral_degrees"],
                "target_minimum_dihedral_degrees":
                    config["quality_thresholds"][
                        "target_minimum_dihedral_degrees"],
                "metadata_transfer_performed": False,
                "isolated_validation_performed": False,
                "experiment_F_performed": False,
            }, indent=2))
        raise
    print(asset)


if __name__ == "__main__":
    main()
