"""Independent validation for a derived retetrahedralized muscle asset."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from muscle_sim.local_remeshing import (
    json_ready, load_raw_mesh, quality_summary, tet_quality,
    validate_closed_boundary)
from viewer.isolated_muscle import (
    attachment_targets, bind_attachment_patches, deformation_gradients,
    load_muscle_data, precompute_energy, reconstruct_fiber,
    total_energy_gradient)


def validate(path, config):
    raw = load_raw_mesh(path)
    muscle, embedding_report = load_muscle_data(path)
    quality = tet_quality(muscle.vertices, muscle.tetrahedra)
    topology = validate_closed_boundary(
        muscle.vertices, muscle.surface_faces)
    sorted_tets = np.sort(muscle.tetrahedra, axis=1)
    duplicate_tets = len(sorted_tets) - len(np.unique(sorted_tets, axis=0))
    precomputed = precompute_energy(muscle)
    isolated_config = yaml.safe_load(
        Path("config/isolated_muscle.yaml").read_text())
    energy, gradient, terms = total_energy_gradient(
        muscle.vertices, muscle, precomputed, isolated_config)
    reconstructed_error = 0.0
    for fiber in muscle.fibers:
        reconstructed_error = max(reconstructed_error, float(np.max(
            np.linalg.norm(reconstruct_fiber(
                fiber, muscle.vertices, muscle.tetrahedra)
                - fiber.rest_points, axis=1))))
    angle = 0.37
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    rigid = muscle.vertices @ rotation.T + [0.01, -0.02, 0.03]
    rigid_energy, _, rigid_terms = total_energy_gradient(
        rigid, muscle, precomputed, isolated_config)
    moved = muscle.vertices.copy()
    provenance_path = Path(path).parent / "retetrahedralization_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text())
        if provenance_path.exists() else {})
    new_region = np.asarray([
        int(key) for key, value in provenance.get(
            "mapping", {}).get("new_tet_source_region", {}).items()
        if value == "local_cavity"], dtype=np.int32)
    if len(new_region):
        center = np.mean(muscle.vertices[
            muscle.tetrahedra[new_region[0]]], axis=0)
        vertex = int(np.argmin(np.linalg.norm(
            muscle.vertices - center, axis=1)))
        moved[vertex] += [1e-5, 0.0, 0.0]
    perturb_energy, perturb_gradient, perturb_terms = total_energy_gradient(
        moved, muscle, precomputed, isolated_config)
    report = {
        "asset": str(path),
        "topology": {
            **topology,
            "duplicate_tet_count": duplicate_tets,
            "positive_orientation": bool(np.all(
                quality["signed_volume"] > 0.0)),
        },
        "rest_quality": quality_summary(quality),
        "elastic_rest_test": {
            "elastic_energy": energy,
            "elastic_residual": float(np.linalg.norm(gradient)),
            "minimum_J": terms["minimum_J"],
            "maximum_J_error": float(np.max(np.abs(
                np.linalg.det(deformation_gradients(
                    muscle.vertices, muscle, precomputed)) - 1.0))),
            "fiber_reconstruction_error": reconstructed_error,
            "embedding_report": embedding_report,
        },
        "rigid_motion_test": {
            "elastic_energy": rigid_energy,
            "minimum_J": rigid_terms["minimum_J"],
            "volume_ratio": rigid_terms["volume_ratio"],
        },
        "local_perturbation_test": {
            "elastic_energy": perturb_energy,
            "finite_force": bool(np.all(np.isfinite(perturb_gradient))),
            "minimum_J": perturb_terms["minimum_J"],
        },
    }
    thresholds = config["quality_thresholds"]
    report["passed"] = bool(
        topology["watertight"] and topology["manifold"]
        and report["topology"]["positive_orientation"]
        and not duplicate_tets
        and report["rest_quality"]["minimum_dihedral_degrees"]
        >= float(thresholds["hard_minimum_dihedral_degrees"])
        and abs(energy) < 1e-9
        and np.linalg.norm(gradient) < 1e-7
        and reconstructed_error <= float(config["fiber_transfer"][
            "maximum_reconstruction_error_m"]))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True,
                        help="derived asset directory or tet file")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.muscle)
    if path.is_dir():
        candidates = sorted(path.glob("*_tet.npz"))
        if len(candidates) != 1:
            raise ValueError("derived directory must contain one *_tet.npz")
        path = candidates[0]
    config = yaml.safe_load(Path(args.config).read_text())
    report = validate(path, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "repaired_mesh_validation.json").write_text(
        json.dumps(json_ready(report), indent=2))
    print(json.dumps({"passed": report["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
