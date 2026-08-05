"""Apply deterministic interior flips to isolated rejected slivers."""
import argparse
import json
from pathlib import Path

import numpy as np

from muscle_sim.layered_meshing import optimize_isolated_sliver
from muscle_sim.local_remeshing import (
    json_ready, quality_summary, tet_quality)
from viewer.isolated_muscle import extract_boundary_faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = np.load(args.candidate)
    vertices = source["vertices"]
    tetrahedra = source["tetrahedra"]
    quality = tet_quality(vertices, tetrahedra)
    bad = np.where(quality["minimum_dihedral_degrees"] < 2.)[0]
    operations = []
    current = tetrahedra
    for tet_id in bad:
        # IDs remain valid only for the first isolated operation. Recompute
        # worst ID after every topology change.
        current_quality = tet_quality(vertices, current)
        worst = int(np.argmin(
            current_quality["minimum_dihedral_degrees"]))
        result = optimize_isolated_sliver(vertices, current, worst)
        if result is None:
            break
        current, operation = result
        operations.append(operation)
    final_quality = tet_quality(vertices, current)
    original_boundary = {
        tuple(sorted(map(int, face)))
        for face in extract_boundary_faces(tetrahedra)}
    final_boundary = {
        tuple(sorted(map(int, face)))
        for face in extract_boundary_faces(current)}
    report = {
        "operations": operations,
        "quality_before": quality_summary(quality),
        "quality_after": quality_summary(final_quality),
        "outer_boundary_exactly_preserved":
            original_boundary == final_boundary,
        "accepted": bool(
            np.min(final_quality["minimum_dihedral_degrees"]) >= 2.
            and original_boundary == final_boundary),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "topology_optimization_provenance.json").write_text(
        json.dumps(json_ready(report), indent=2))
    if report["accepted"]:
        np.savez_compressed(
            output / "optimized_tet_candidate.npz",
            vertices=vertices, tetrahedra=current,
            boundary_faces=extract_boundary_faces(current),
            **final_quality)
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
