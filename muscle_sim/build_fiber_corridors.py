"""Build narrow, adaptively connected source-fiber tet corridors."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.fiber_portals import expand_corridor
from muscle_sim.fiber_repair import (
    load_repaired_fibers, load_surface, write_json)
from muscle_sim.fiber_routing import ClearanceField, build_tet_adjacency
from viewer.isolated_muscle import TetLocator, load_muscle_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--tet-mesh", required=True)
    parser.add_argument("--fibers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["fiber_corridor"]
    mesh_path = Path(args.tet_mesh)
    if mesh_path.is_dir():
        mesh_path /= "optimized_tet_candidate.npz"
    mesh = np.load(mesh_path)
    vertices, tetrahedra = mesh["vertices"], mesh["tetrahedra"]
    neighbors, _ = build_tet_adjacency(tetrahedra)
    locator = TetLocator(vertices, tetrahedra)
    muscle, _ = load_muscle_data(
        args.fibers or config["source_tet_asset"])
    anchor_path = Path(config["anchor_candidate"])
    repaired, _ = load_repaired_fibers(anchor_path)
    repaired_by_id = {
        int(row["fiber_index"]): np.asarray(row["points"])
        for row in repaired}
    inspection = json.loads(Path(config["segment_inspection"]).read_text())
    invalid_by_fiber = {}
    for row in inspection["segments"]:
        if row["classification"] != "STRAIGHT_SEGMENT_VALID":
            invalid_by_fiber.setdefault(int(row["fiber_id"]), []).append(
                int(row["source_segment"]))
    clearance_field = ClearanceField(load_surface(config["c3_surface"]))
    tet_clearance = clearance_field.approximate(
        vertices[tetrahedra].mean(axis=1))
    masks, rows = [], []
    for fiber in muscle.fibers:
        source_points = np.asarray(fiber.rest_points)
        points = repaired_by_id[int(fiber.fiber_index)]
        for segment in invalid_by_fiber.get(int(fiber.fiber_index), []):
            start = locator.locate(
                points[segment], tolerance=2e-6, candidates=512)[0]
            end = locator.locate(
                points[segment + 1], tolerance=2e-6, candidates=512)[0]
            if start < 0 or end < 0:
                raise ValueError("ANCHOR_NEAR_BOUNDARY")
            first, last = max(0, segment - 1), min(
                len(source_points), segment + 3)
            local_source = source_points[first:last]
            try:
                allowed, fields, radius, attempts = expand_corridor(
                    vertices, tetrahedra, neighbors, local_source,
                    tet_clearance, start, end, settings)
                failure = None
            except ValueError as error:
                allowed = np.zeros(len(tetrahedra), dtype=bool)
                radius, attempts, failure = None, [], str(error)
            masks.append(np.flatnonzero(allowed).astype(np.int32))
            rows.append({
                "fiber_id": int(fiber.fiber_index),
                "source_segment_index": segment,
                "start_tet": int(start), "end_tet": int(end),
                "minimum_connected_radius_m": radius,
                "corridor_tet_count": int(allowed.sum()),
                "expansion_attempts": attempts,
                "failure_classification": failure,
                "minimum_corridor_clearance_m": (
                    float(np.min(tet_clearance[allowed]))
                    if np.any(allowed) else None)})
            print(f"corridor fiber {fiber.fiber_index} segment {segment}: "
                  f"{allowed.sum()} tets"
                  + (f" at {radius:.6g} m" if radius else f" {failure}"),
                  flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "fiber_corridors.npz",
        tet_ids=np.asarray(masks, dtype=object),
        keys=np.asarray([[row["fiber_id"], row["source_segment_index"]]
                         for row in rows], dtype=np.int32),
        tet_clearance=tet_clearance)
    write_json(output / "fiber_corridor_summary.json", {
        "muscle": args.muscle, "accepted_tet_asset": str(mesh_path),
        "source_fibers": args.fibers or config["source_tet_asset"],
        "settings": settings, "fibers": rows})
    print(output / "fiber_corridors.npz")


if __name__ == "__main__":
    main()
