"""CLI for deterministic fixed-boundary local retetrahedralization."""
import argparse
from pathlib import Path

import yaml

from muscle_sim.local_remeshing import run_repair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tet-dir", default="tet")
    parser.add_argument("--diagnostic-dir", default=(
        ".bake_outputs/gracilis_semitendinosus_rest_fix_full_patch_final"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    source = Path(args.tet_dir) / f"{args.muscle}_tet.npz"
    asset, _ = run_repair(source, config, args.diagnostic_dir, args.output)
    print(asset)


if __name__ == "__main__":
    main()
