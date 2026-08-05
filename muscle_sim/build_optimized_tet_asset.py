"""Publish a hard-gate-passing topology-optimized tet candidate."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import trimesh
import yaml

from muscle_sim.full_retetrahedralization import (
    contact_correspondence, render_embedding, transfer_attachment_faces)
from muscle_sim.fiber_repair import load_repaired_fibers
from muscle_sim.local_remeshing import (
    active_contact_points, json_ready, quality_summary, tet_quality)
from viewer.isolated_muscle import (
    EmbeddedFiber, TetLocator, build_tet_fiber_directions,
    extract_boundary_faces, load_muscle_data)


def reembed(fibers, vertices, tetrahedra, tolerance):
    locator = TetLocator(vertices, tetrahedra)
    output, outside, maximum_error = [], [], 0.0
    minimum_weight = np.inf
    for fiber in fibers:
        ids, weights = [], []
        for sample_id, point in enumerate(fiber.rest_points):
            tet_id, bary, score = locator.locate(
                point, tolerance=tolerance, candidates=512)
            if tet_id < 0:
                outside.append([fiber.fiber_index, sample_id, score])
            ids.append(tet_id)
            weights.append(bary)
            minimum_weight = min(minimum_weight, float(np.min(bary)))
        ids, weights = np.asarray(ids), np.asarray(weights)
        valid = ids >= 0
        if np.any(valid):
            reconstructed = np.einsum(
                "ni,nij->nj", weights[valid],
                vertices[tetrahedra[ids[valid]]])
            maximum_error = max(maximum_error, float(np.max(np.linalg.norm(
                reconstructed - fiber.rest_points[valid], axis=1))))
        output.append(EmbeddedFiber(
            fiber.stream_index, fiber.fiber_index, ids, weights,
            fiber.rest_points.copy(), fiber.rest_segment_lengths.copy()))
    return output, {
        "fiber_count": len(fibers), "outside_samples": outside,
        "maximum_reconstruction_error": maximum_error,
        "minimum_barycentric_weight": minimum_weight,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fiber-override")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    data = np.load(args.candidate)
    vertices, tetrahedra = data["vertices"], data["tetrahedra"]
    quality = tet_quality(vertices, tetrahedra)
    hard = float(config["layered_meshing"]["quality"][
        "hard_minimum_dihedral_degrees"])
    if np.min(quality["minimum_dihedral_degrees"]) < hard:
        raise ValueError("candidate does not pass hard tet quality")
    surface = trimesh.load(args.surface, force="mesh", process=False)
    surface_vertices = np.asarray(surface.vertices)
    surface_faces = np.asarray(surface.faces)
    if not np.allclose(
            vertices[:len(surface_vertices)], surface_vertices,
            atol=1e-12, rtol=0.):
        raise ValueError("C3 outer vertices changed")
    boundary = extract_boundary_faces(tetrahedra)
    if ({tuple(sorted(map(int, face))) for face in boundary}
            != {tuple(sorted(map(int, face))) for face in surface_faces}):
        raise ValueError("C3 outer faces changed")
    current_muscle, _ = load_muscle_data(config["current_tet_asset"])
    attachment_rows, cap_face_ids = transfer_attachment_faces(
        surface_vertices, surface_faces, current_muscle, 0.0005)
    if args.fiber_override:
        _, fibers = load_repaired_fibers(args.fiber_override)
        if fibers is None:
            raise ValueError("fiber override has no tet embeddings")
        fiber_report = {
            "fiber_count": len(fibers), "outside_samples": [],
            "maximum_reconstruction_error": float(max(np.max(np.linalg.norm(
                np.einsum("ni,nij->nj", fiber.barycentric,
                          vertices[tetrahedra[fiber.tet_ids]])
                - fiber.rest_points, axis=1)) for fiber in fibers)),
            "minimum_barycentric_weight": float(min(
                np.min(fiber.barycentric) for fiber in fibers)),
            "source": str(args.fiber_override)}
    else:
        fibers, fiber_report = reembed(
            current_muscle.fibers, vertices, tetrahedra, 2e-6)
    if fiber_report["outside_samples"]:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        boundary_mesh = trimesh.Trimesh(
            vertices=vertices, faces=boundary, process=False)
        outside_rows = []
        by_key = {(fiber.fiber_index, sample_id): point
                  for fiber in current_muscle.fibers
                  for sample_id, point in enumerate(fiber.rest_points)}
        for fiber_id, sample_id, score in fiber_report["outside_samples"]:
            point = by_key[(fiber_id, sample_id)]
            _, distance, face_id = trimesh.proximity.closest_point(
                boundary_mesh, point.reshape(1, 3))
            outside_rows.append({
                "fiber_index": fiber_id, "sample_index": sample_id,
                "point": point, "minimum_barycentric_score": score,
                "distance_to_new_boundary": float(distance[0]),
                "nearest_new_boundary_face": int(face_id[0]),
            })
        fiber_report["outside_sample_diagnostics"] = outside_rows
        (output / "fiber_transfer_failure.json").write_text(json.dumps(
            json_ready(fiber_report), indent=2))
        raise ValueError(
            f"{len(fiber_report['outside_samples'])} fiber samples outside "
            "optimized tet mesh")
    directions, valid_directions = build_tet_fiber_directions(
        vertices, tetrahedra, fibers)
    old_vertices = current_muscle.vertices
    old_faces = current_muscle.surface_faces
    contacts = active_contact_points(
        Path(config["diagnostic_dir"]) / "active_contact_multipliers.json",
        config["muscle_name"], old_vertices, old_faces)
    correspondence = contact_correspondence(
        old_vertices, old_faces, vertices, boundary, contacts, .005)
    reference_raw = current_muscle.raw
    derived = dict(reference_raw)
    derived.update({
        "vertices": vertices, "tetrahedra": tetrahedra,
        "faces": surface_faces, "render_faces": surface_faces,
        "sim_faces": boundary, "surface_face_count": len(surface_faces),
        "cap_face_indices": cap_face_ids,
        "anchor_vertices": np.unique(np.concatenate([
            row["vertex_ids"] for row in attachment_rows])),
        "cap_attachments": np.asarray([
            [int(row["vertex_ids"][0]), 0, row["end_type"], 0, 0]
            for row in attachment_rows], dtype=np.int32),
        "_transferred_attachment_patches": attachment_rows,
        "_transferred_fiber_embeddings": fibers,
        "_simulation_fiber_embeddings": fibers,
        "_material_region_labels": np.zeros(
            len(tetrahedra), dtype=np.int32),
    })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    asset = output / "L_Semitendinosus_tet.npz"
    with asset.open("wb") as handle:
        pickle.dump(derived, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(
        output / "fiber_embeddings.npz",
        fibers=np.asarray(fibers, dtype=object),
        directions=directions, valid=valid_directions)
    (output / "attachment_labels.json").write_text(json.dumps(
        json_ready(attachment_rows), indent=2))
    (output / "contact_surface_correspondence.json").write_text(json.dumps(
        json_ready(correspondence), indent=2))
    provenance = {
        "accepted": True,
        "strategy": "M4_direct_TetGen_plus_single_interior_2_to_3_flip",
        "layered_region_constructed": False,
        "reason_layering_not_used": (
            "exact failure map found one isolated surface-diagonal sliver, "
            "not a thin-region cluster"),
        "quality": quality_summary(quality),
        "outer_C3_surface_exact": True,
        "attachment_transfer": attachment_rows,
        "fiber_transfer": fiber_report,
        "contact_state": {
            "old_primitive_ids_valid": False,
            "multipliers_reset_required": True,
            "active_set_reset_required": True,
        },
    }
    (output / "layered_meshing_provenance.json").write_text(json.dumps(
        json_ready(provenance), indent=2))
    print(asset)


if __name__ == "__main__":
    main()
