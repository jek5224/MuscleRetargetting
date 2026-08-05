#!/usr/bin/env python3
"""Classify primitive-level rest penetrations and attachment compatibility."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml

from core.dartHelper import buildFromInfo, saveSkeletonInfo
from muscle_sim.rest_contact import (
    attachment_patch_report, directed_bone_constraints,
    directed_muscle_constraints, duplicate_diagnostics,
    summarize_constraints)
from muscle_sim.run_multimuscle import (
    BONE_BODY, boundary_ring_exclusion, capture_bones,
    load_bone_rest_surfaces)
from viewer.isolated_muscle import (
    bind_attachment_patches, load_muscle_data, precompute_energy)
from viewer.multimuscle import MultiMuscleSystem, MuscleBody


def ready(value):
    if isinstance(value, dict):
        return {key: ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_rest_system(config, tet_dir, muscle_overrides=None):
    with Path("config/isolated_muscle.yaml").open() as handle:
        isolated = yaml.safe_load(handle)
    bodies = []
    muscle_overrides = muscle_overrides or {}
    for name in config["muscles"]["selected"]:
        muscle, _ = load_muscle_data(
            muscle_overrides.get(
                name, Path(tet_dir) / f"{name}_tet.npz"))
        body = MuscleBody(
            muscle, precompute_energy(
                muscle, float(isolated["fiber"]["nearby_radius"])),
            copy.deepcopy(isolated))
        bodies.append(body)
    index = {body.name: i for i, body in enumerate(bodies)}
    pairs = [
        (index[first], index[second])
        for first, second in config["contact_pairs"]]
    system = MultiMuscleSystem(bodies, pairs)
    for body in bodies:
        boundary_ring_exclusion(
            body, int(config["contact"]["attachment_exclusion_rings"]))
    skeleton_info, root_name, _, _, _, _ = saveSkeletonInfo(
        "data/zygote_skel.xml")
    skeleton = buildFromInfo(skeleton_info, root_name)
    skeleton.setPositions(np.zeros(skeleton.getNumDofs()))
    collision_names = sorted({
        bone for body in bodies
        for bone in config["muscles"]["collision_bones"][body.name]})
    attachment_names = sorted({
        patch.bone_name for body in bodies
        for patch in body.muscle.attachment_patches})
    transforms = capture_bones(
        skeleton, sorted(set(
            attachment_names + [BONE_BODY[name]
                                for name in collision_names])))
    for body in bodies:
        bind_attachment_patches(
            body.muscle.attachment_patches,
            body.muscle.vertices, transforms)
    bones, _ = load_bone_rest_surfaces(
        skeleton, collision_names)
    system.bones = bones
    system.body_bones = {
        body.name: config["muscles"]["collision_bones"][body.name]
        for body in bodies}
    return system


def export_constraint_points(path, constraints):
    points = np.asarray([
        constraint["closest_point"] for constraint in constraints],
        dtype=np.float64)
    colors = {
        "FREE_FREE": [50, 220, 80, 255],
        "CONSTRAINED_FREE": [255, 210, 30, 255],
        "FREE_CONSTRAINED": [255, 140, 20, 255],
        "CONSTRAINED_CONSTRAINED_SAME_BONE": [80, 150, 255, 255],
        "CONSTRAINED_CONSTRAINED_DIFFERENT_BONES": [255, 30, 30, 255],
    }
    if len(points):
        cloud = trimesh.points.PointCloud(
            points, colors=np.asarray([
                colors[item["classification"]] for item in constraints]))
        cloud.export(path)


def run(args):
    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    system = build_rest_system(config, args.tet_dir)
    positions = [
        body.muscle.vertices.copy() for body in system.bodies]
    constraints = []
    for first_id, second_id in system.contact_pairs:
        first, second = system.bodies[first_id], system.bodies[second_id]
        constraints.extend(directed_muscle_constraints(
            first, second, positions[first_id], positions[second_id],
            config))
        constraints.extend(directed_muscle_constraints(
            second, first, positions[second_id], positions[first_id],
            config))
    for body, vertices in zip(system.bodies, positions):
        for bone_name in system.body_bones[body.name]:
            constraints.extend(directed_bone_constraints(
                body, vertices, bone_name, system.bones[bone_name],
                config))
    duplicates = duplicate_diagnostics(constraints)
    report = summarize_constraints(constraints)
    report["constraints"] = constraints
    report["symmetric_duplicate_diagnostics"] = {
        "near_reverse_constraint_count": len(duplicates),
        "groups": duplicates,
    }
    with (output / "initial_contact_feasibility_report.json").open(
            "w") as handle:
        json.dump(ready(report), handle, indent=2)
    muscle_targets = {body.name for body in system.bodies}
    muscle_constraints = [
        item for item in constraints
        if item["target_body"] in muscle_targets]
    bone_constraints = [
        item for item in constraints
        if item["target_body"] not in muscle_targets]
    def maxima(items):
        return {
            "constraint_count": len(items),
            "maximum_exact_depth": max(
                [item["exact_penetration_depth"] for item in items]
                or [0.0]),
            "maximum_solver_proxy_penetration": max(
                [item["solver_proxy_penetration"] for item in items]
                or [0.0]),
        }
    with (output / "penetration_definition_report.json").open(
            "w") as handle:
        json.dump({
            "exact_signed_gap": (
                "g=-closest_surface_depth for an initially contained "
                "vertex; feasible means g>=0"),
            "solver_proxy": (
                "max(0, thickness-g), hence inside depth + thickness"),
            "initial_muscle_muscle": maxima(muscle_constraints),
            "initial_muscle_bone": maxima(bone_constraints),
            "previous_failed_solve_3.30mm": (
                "measured on a deformed failed line-search/optimizer state; "
                "it is not the original rest-state exact depth"),
            "unit": "metres",
        }, handle, indent=2)
    patches = []
    for body, vertices in zip(system.bodies, positions):
        patches.extend(attachment_patch_report(
            body, vertices, system.bones))
    for patch in patches:
        overlap_ids = []
        for constraint in constraints:
            source_match = (
                constraint["source_body"] == patch["muscle"]
                and constraint["source_attachment_patch_id"]
                == patch["patch_id"])
            target_match = (
                constraint["target_body"] == patch["muscle"]
                and patch["patch_id"]
                in constraint["target_attachment_patch_ids"])
            if source_match or target_match:
                overlap_ids.append(constraint["constraint_id"])
        patch["overlapping_contact_constraint_ids"] = overlap_ids
        patch["overlapping_contact_count"] = len(overlap_ids)
    with (output / "attachment_patch_inspection.json").open("w") as handle:
        json.dump(ready(patches), handle, indent=2)
    np.savez_compressed(
        output / "attachment_patch_proposals.npz",
        proposals=np.asarray(patches, dtype=object))
    patch_points, patch_colors = [], []
    palette = np.asarray([
        [230, 40, 40, 255], [40, 120, 240, 255],
        [40, 210, 90, 255], [230, 180, 30, 255],
        [180, 50, 220, 255], [20, 210, 210, 255]],
        dtype=np.uint8)
    body_by_name = {body.name: body for body in system.bodies}
    for index, patch in enumerate(patches):
        body = body_by_name[patch["muscle"]]
        patch_points.extend(
            body.muscle.vertices[patch["current_patch"]])
        patch_colors.extend(
            [palette[index % len(palette)]] * len(patch["current_patch"]))
    trimesh.points.PointCloud(
        np.asarray(patch_points), colors=np.asarray(patch_colors)).export(
            output / "attachment_patches.ply")
    export_constraint_points(
        output / "constraint_classes.ply", constraints)
    print(json.dumps({
        "constraints": report["constraint_count"],
        "maximum_exact_depth":
            report["maximum_exact_initial_penetration"],
        "maximum_solver_proxy":
            report["maximum_initial_solver_proxy_penetration"],
        "categories": {
            key: value["count"]
            for key, value in report["categories"].items()},
        "near_reverse_duplicates": len(duplicates),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/pes_anserinus_multimuscle.yaml")
    parser.add_argument("--tet-dir", default="tet")
    parser.add_argument(
        "--output", default=".bake_outputs/rest_feasibility")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
