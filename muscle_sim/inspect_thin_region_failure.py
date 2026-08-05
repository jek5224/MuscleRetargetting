"""CLI for exact rejected-volume thin/sliver mapping."""
import argparse
import json
from pathlib import Path

import meshio
import numpy as np
import trimesh
import yaml

from muscle_sim.layered_meshing import failure_map
from muscle_sim.local_remeshing import json_ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True)
    parser.add_argument("--tet-failure", required=True)
    parser.add_argument("--candidate", default=(
        "assets/generated/Left_Semitendinosus_cross_section_retet/"
        "rejected_tet_candidate.npz"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    report, vertices, tetrahedra, failed = failure_map(
        args.candidate, args.surface, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "thin_region_failure_map.json").write_text(json.dumps(
        json_ready(report), indent=2))
    meshio.write(
        output / "thin_region_failure_tets.vtu",
        meshio.Mesh(vertices, [("tetra", tetrahedra[failed])],
                    cell_data={"minimum_dihedral_degrees": [
                        np.asarray([row["minimum_dihedral_degrees"]
                                    for row in report["failed_tets"]])] }))
    surface = trimesh.load(args.surface, force="mesh", process=False)
    face_ids = sorted({row["nearest_surface_triangle"]
                       for row in report["failed_tets"]})
    trimesh.Trimesh(
        vertices=surface.vertices, faces=surface.faces[face_ids],
        process=False).export(output / "thin_region_failure_surface.ply")


if __name__ == "__main__":
    main()
