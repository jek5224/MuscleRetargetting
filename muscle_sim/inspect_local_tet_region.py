"""CLI for selecting and exporting a local tetrahedral repair cavity."""
import argparse
from pathlib import Path

import yaml

from muscle_sim.local_remeshing import (
    load_raw_mesh, save_debug_region, select_region)
from viewer.isolated_muscle import (
    attachment_patches_from_tet, extract_boundary_faces, load_muscle_data,
    orient_tetrahedra)


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
    path = Path(args.tet_dir) / f"{args.muscle}_tet.npz"
    raw = load_raw_mesh(path)
    vertices = raw["vertices"]
    tetrahedra = orient_tetrahedra(vertices, raw["tetrahedra"])
    muscle, _ = load_muscle_data(path)
    selection = select_region(
        vertices, tetrahedra, extract_boundary_faces(tetrahedra), config,
        args.diagnostic_dir, args.muscle,
        attachment_patches_from_tet(raw), muscle.fibers)
    save_debug_region(args.output, vertices, tetrahedra, selection)
    print(f"selected {len(selection.selected_tets)} tetrahedra")


if __name__ == "__main__":
    main()
