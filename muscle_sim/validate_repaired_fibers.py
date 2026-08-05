"""Independent geometry and embedding validation for repaired fibers."""
import argparse
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.fiber_repair import (
    fiber_shape_metrics, load_repaired_fibers, load_surface, write_json)
from viewer.isolated_muscle import load_muscle_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--fibers", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    source, _ = load_muscle_data(config["source_tet_asset"])
    fiber_path = Path(args.fibers)
    if fiber_path.is_dir():
        fiber_path /= "repaired_simulation_fibers.npz"
    repaired, embedded = load_repaired_fibers(fiber_path)
    metrics = fiber_shape_metrics(
        source.fibers, repaired, load_surface(config["c3_surface"]),
        config["fiber_repair"])
    metrics["embedded_fiber_count"] = 0 if embedded is None else len(embedded)
    metrics["all_samples_embedded"] = bool(
        embedded is not None and len(embedded) == len(source.fibers)
        and all(np.all(fiber.tet_ids >= 0) for fiber in embedded))
    metrics["zero_prestress_rest_lengths"] = bool(
        embedded is not None and all(np.allclose(
            fiber.rest_segment_lengths,
            np.linalg.norm(np.diff(fiber.rest_points, axis=0), axis=1),
            atol=1e-12, rtol=1e-10) for fiber in embedded))
    metrics["accepted"] = bool(
        metrics["accepted"] and metrics["all_samples_embedded"]
        and metrics["zero_prestress_rest_lengths"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "validation_report.json", metrics)
    if not metrics["accepted"]:
        raise ValueError("repaired fibers failed validation")
    print(output / "validation_report.json")


if __name__ == "__main__":
    main()
