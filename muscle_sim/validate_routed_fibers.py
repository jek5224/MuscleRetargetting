"""Independent exact-containment validation of routed fibers."""
import argparse
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.fiber_repair import load_surface, signed_clearance, write_json
from muscle_sim.fiber_routing import (
    load_routed_fibers, maximum_turn, validate_exact_route)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--fibers", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["fiber_path_routing"]
    candidate = Path(args.muscle)
    if candidate.is_dir():
        candidate /= "optimized_tet_candidate.npz"
    data = np.load(candidate)
    vertices, tetrahedra = data["vertices"], data["tetrahedra"]
    fiber_path = Path(args.fibers)
    if fiber_path.is_dir():
        fiber_path /= "routed_simulation_fibers.npz"
    routes, embedded = load_routed_fibers(fiber_path)
    surface = load_surface(config["c3_surface"])
    rows = []
    for route, fiber in zip(routes, embedded):
        exact = validate_exact_route(
            vertices, tetrahedra, route["subsegment_owner_tets"],
            route["points"])
        reconstruction = np.einsum(
            "ni,nij->nj", fiber.barycentric,
            vertices[tetrahedra[fiber.tet_ids]])
        clearance = signed_clearance(surface, route["points"])[0]
        rest_lengths = np.linalg.norm(
            np.diff(route["points"], axis=0), axis=1)
        rows.append({
            "fiber_id": route["fiber_index"], "exact_containment": exact,
            "maximum_reconstruction_error_m": float(np.max(np.linalg.norm(
                reconstruction - route["points"], axis=1))),
            "minimum_clearance_m": float(np.min(clearance)),
            "maximum_turn_degrees": maximum_turn(route["points"]),
            "zero_prestress_rest_lengths": bool(np.allclose(
                rest_lengths, fiber.rest_segment_lengths,
                atol=1e-13, rtol=1e-12))})
    accepted = all(
        row["exact_containment"]["valid"]
        and row["maximum_reconstruction_error_m"] <= 1e-10
        and row["minimum_clearance_m"] >= float(
            settings["interior_margin_m"])
        and row["zero_prestress_rest_lengths"] for row in rows)
    report = {"accepted": accepted, "fiber_count": len(rows), "fibers": rows}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "validation_report.json", report)
    if not accepted:
        raise ValueError("routed fibers failed independent validation")
    print(output / "validation_report.json")


if __name__ == "__main__":
    main()
