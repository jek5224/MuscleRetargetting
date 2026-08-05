"""Generate and hard-gate F0--F4 repaired simulation-fiber candidates."""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.cross_sections import (
    representative_bundle_path, rotation_minimizing_frames)
from muscle_sim.fiber_repair import (
    diagnose_fibers, embed_repaired_fibers, fiber_shape_metrics,
    load_surface, repair_fibers, save_fibers, write_json,
    write_obj_polylines)
from viewer.isolated_muscle import load_muscle_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--tet-mesh", required=True)
    parser.add_argument("--fibers")
    parser.add_argument("--sections")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidates", nargs="+", default=["F0", "F1", "F2", "F3", "F4"])
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["fiber_repair"]
    muscle, _ = load_muscle_data(args.fibers or config["source_tet_asset"])
    c3 = load_surface(args.surface)
    surfaces = {
        "immutable_reference": load_surface(
            config["immutable_reference_asset"]),
        "invalid_tet_boundary": load_surface(
            config["invalid_tet_boundary_asset"], boundary_from_tets=True),
        "path_b": load_surface(config["path_b_surface"]), "c3": c3}
    frames = rotation_minimizing_frames(
        representative_bundle_path(muscle.fibers))
    diagnosis = diagnose_fibers(
        muscle.fibers, surfaces, frames, settings)
    tet_data = np.load(
        Path(args.tet_mesh) if Path(args.tet_mesh).is_file()
        else Path(args.tet_mesh) / "optimized_tet_candidate.npz")
    vertices, tetrahedra = tet_data["vertices"], tet_data["tetrahedra"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    accepted = []
    for name in args.candidates:
        repaired, mapping = repair_fibers(
            muscle.fibers, c3, frames, diagnosis, settings, name)
        metrics = fiber_shape_metrics(
            muscle.fibers, repaired, c3, settings)
        embedded, embedding = embed_repaired_fibers(
            repaired, vertices, tetrahedra)
        metrics["embedding"] = embedding
        metrics["all_samples_embedded"] = not embedding["outside_samples"]
        metrics["accepted"] = bool(
            metrics["accepted"] and metrics["all_samples_embedded"])
        candidate_dir = output / name
        candidate_dir.mkdir(exist_ok=True)
        save_fibers(
            candidate_dir / "repaired_simulation_fibers.npz",
            repaired, embedded)
        write_json(candidate_dir / "fiber_repair_mapping.json", mapping)
        write_json(candidate_dir / "fiber_repair_summary.json", metrics)
        candidates.append({"candidate": name, **metrics})
        if metrics["accepted"]:
            accepted.append((name, repaired, embedded, mapping, metrics))
    if not accepted:
        write_json(output / "candidate_comparison.json", candidates)
        raise ValueError("no fiber repair candidate passed all hard gates")
    best = min(accepted, key=lambda item: (
        item[4]["maximum_displacement_m"],
        item[4]["RMS_displacement_m"], item[0]))
    name, repaired, embedded, mapping, metrics = best
    save_fibers(output / "repaired_simulation_fibers.npz", repaired, embedded)
    write_obj_polylines(
        output / "original_and_repaired_fibers.obj",
        repaired, muscle.fibers)
    write_json(output / "fiber_repair_mapping.json", mapping)
    write_json(output / "fiber_repair_summary.json", {
        "accepted": True, "selected_candidate": name,
        "diagnosis": diagnosis, "metrics": metrics})
    write_json(output / "candidate_comparison.json", candidates)
    with (output / "section_fiber_distribution_before.csv").open(
            "w", newline="") as before_file, (
            output / "section_fiber_distribution_after.csv").open(
                "w", newline="") as after_file:
        headers = ["fiber_id", "sample_id", "x", "y", "z"]
        before_writer, after_writer = (
            csv.DictWriter(before_file, fieldnames=headers),
            csv.DictWriter(after_file, fieldnames=headers))
        before_writer.writeheader()
        after_writer.writeheader()
        repaired_by_id = {row["fiber_index"]: row for row in repaired}
        for fiber in muscle.fibers:
            after = repaired_by_id[fiber.fiber_index]["points"]
            for sample_id, (old, new) in enumerate(zip(
                    fiber.rest_points, after)):
                before_writer.writerow(dict(
                    zip(headers, [fiber.fiber_index, sample_id, *old])))
                after_writer.writerow(dict(
                    zip(headers, [fiber.fiber_index, sample_id, *new])))
    print(output / "repaired_simulation_fibers.npz")


if __name__ == "__main__":
    main()
